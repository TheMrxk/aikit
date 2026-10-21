"""统一 AI 客户端（OpenAI 兼容协议，httpx 实现）。

对外主入口：
    from aikit import AI
    ai = AI()                         # 从 env/配置文件自动解析
    ai.chat("你好")                    # 普通对话
    ai.chat_json(prompt, schema_hint) # 强制 JSON
    ai.stream(messages)               # 流式

设计：
- 全部走 OpenAI 兼容的 /chat/completions，主流厂商 + coding plan + 中转站通用；
- 401/403 不重试（鉴权问题重试无用）；429/5xx/网络错误指数退避重试；
- 配置了多个通道时自动故障转移（默认通道优先）；
- chat_json 用「提示词约束 + 响应格式 + 容错提取 + 重试」保证拿到 JSON；
- 任何异常都不回显密钥。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator

import httpx

from .config import ConfigStore
from .errors import (
    AIKitError, AllProvidersFailed, AuthError, JSONModeError, NetworkError,
    ProtocolError, ProviderError, RateLimitError,
)
from .models import ChatResult, GenParams, Message, ModelConfig, Usage

_CHAT_PATH = "/chat/completions"


class _OneClient:
    """单个通道的轻量封装。"""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(cfg.extra_headers)
        self._http = httpx.AsyncClient(
            base_url=cfg.base_url, timeout=cfg.timeout, headers=headers)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, payload: dict) -> dict:
        try:
            r = await self._http.post(_CHAT_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise NetworkError(f"请求超时（{self.cfg.timeout}s）") from exc
        except httpx.HTTPError as exc:
            raise NetworkError(f"网络错误：{type(exc).__name__}") from exc

        if r.status_code in (401, 403):
            raise AuthError(f"鉴权失败 HTTP {r.status_code}（请检查 API Key）",
                            r.status_code)
        if r.status_code == 429:
            retry_after = None
            try:
                retry_after = float(r.headers.get("retry-after"))
            except (TypeError, ValueError):
                pass
            raise RateLimitError("触发限流（429）", retry_after)
        if r.status_code >= 500:
            raise ProviderError(f"服务端错误 HTTP {r.status_code}",
                                r.status_code, _safe_body(r))
        if r.status_code >= 400:
            # 400 类通常是参数/模型名错误，不重试
            raise ProviderError(
                f"请求被拒 HTTP {r.status_code}：{_safe_body(r)[:200]}",
                r.status_code, _safe_body(r))
        try:
            return r.json()
        except ValueError as exc:
            raise ProtocolError("响应不是合法 JSON") from exc

    async def raw_completion(self, payload: dict) -> dict:
        return await self._post(payload)


def _safe_body(r: httpx.Response) -> str:
    try:
        return r.text[:500]
    except Exception:  # noqa: BLE001
        return ""


def _build_payload(cfg: ModelConfig, messages: list[dict],
                   gen: GenParams | None, force_json: bool,
                   json_schema: dict | None) -> dict:
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
    }
    if gen:
        payload.update(gen.to_dict())
    if force_json:
        # 主流厂商都支持 response_format=json_object；不支持的会忽略或 400（被捕获后降级提示）
        payload["response_format"] = {"type": "json_object"}
    if json_schema and cfg.extra_body.get("supports_json_schema"):
        payload["response_format"] = {
            "type": "json_schema", "json_schema": json_schema}
    if cfg.extra_body:
        for k, v in cfg.extra_body.items():
            if k != "supports_json_schema":
                payload[k] = v
    return payload


def _parse_choice(data: dict) -> tuple[str, str, Usage]:
    try:
        choice = data["choices"][0]
        msg = choice.get("message", {})
        text = msg.get("content") or ""
        finish = choice.get("finish_reason") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ProtocolError(f"响应结构异常：{str(data)[:200]}") from exc
    return text, str(finish), Usage.from_raw(data.get("usage"))


class AI:
    """统一入口。

    参数：
      config       显式传入 ConfigStore；否则自动新建（读 AIKIT_CONFIG / env）
      auto_failover 多通道是否自动故障转移，默认 True
      max_retries  单通道内对可重试错误的最大重试次数
    """

    def __init__(self, config: ConfigStore | None = None,
                 auto_failover: bool = True, max_retries: int = 2):
        self.store = config or ConfigStore()
        self.auto_failover = auto_failover
        self.max_retries = max_retries

    # ---------- 上下文管理 ----------
    async def __aenter__(self) -> "AI":
        return self

    async def __aexit__(self, *exc) -> None:
        pass

    # ---------- 核心：带重试 + 故障转移 ----------
    async def _run(self, fn, *args, **kwargs):
        channels = self.store.resolve_all_enabled()
        if not channels:
            self.store.resolve()  # 抛出标准 ConfigError
        errors: list[tuple[str, BaseException]] = []
        for idx, (name, cfg) in enumerate(channels):
            is_last = (idx == len(channels) - 1)
            client = _OneClient(cfg)
            try:
                return await self._with_retries(
                    client, fn, cfg, *args,
                    allow_failover=(self.auto_failover and not is_last),
                    **kwargs)
            except (AuthError, ProviderError) as exc:
                # 401/鉴权 或明确 4xx 参数错：不重试本通道；可转移则换下一家
                errors.append((name, exc))
                if not (self.auto_failover and not is_last):
                    raise
            except (RateLimitError, NetworkError) as exc:
                errors.append((name, exc))
                if not (self.auto_failover and not is_last):
                    raise
            finally:
                await client.aclose()
        raise AllProvidersFailed(errors)

    async def _with_retries(self, client: _OneClient, fn, cfg: ModelConfig,
                            *args, allow_failover: bool, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await fn(client, cfg, *args, **kwargs)
            except (RateLimitError, NetworkError, ProviderError) as exc:
                last_exc = exc
                # 429 尊重 Retry-After；否则指数退避
                wait = getattr(exc, "retry_after", None)
                if wait is None:
                    wait = min(2 ** attempt, 8)
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(float(wait))
        assert last_exc is not None
        raise last_exc

    # ---------- 对外方法 ----------
    async def chat(
        self,
        messages: Any,
        *,
        system: str | None = None,
        gen: GenParams | None = None,
        channel: str | None = None,
    ) -> ChatResult:
        """普通对话。messages 可为字符串、dict 列表或 Message 列表。"""
        if channel:
            return await self._run_on(channel, self._do_chat,
                                      self._norm(messages, system), gen)
        return await self._run(self._do_chat, self._norm(messages, system), gen)

    async def _run_on(self, channel: str, fn, *args):
        cfg = self.store.resolve(channel)
        client = _OneClient(cfg)
        try:
            return await fn(client, cfg, *args)
        finally:
            await client.aclose()

    async def _do_chat(self, client: _OneClient, cfg: ModelConfig,
                       msgs: list[dict], gen: GenParams | None) -> ChatResult:
        data = await client.raw_completion(_build_payload(cfg, msgs, gen, False, None))
        text, finish, usage = _parse_choice(data)
        return ChatResult(text=text, model=data.get("model", cfg.model),
                          provider=cfg.provider, finish_reason=finish,
                          usage=usage, raw=data)

    async def stream(
        self,
        messages: Any,
        *,
        system: str | None = None,
        gen: GenParams | None = None,
        channel: str | None = None,
    ) -> AsyncIterator[str]:
        """流式输出，逐段 yield 文本增量。流式不做通道间重试（已开始返回）。"""
        cfg = self.store.resolve(channel)
        msgs = self._norm(messages, system)
        headers = {"Authorization": f"Bearer {cfg.api_key}",
                   "Content-Type": "application/json",
                   "Accept": "text/event-stream"}
        headers.update(cfg.extra_headers)
        payload = _build_payload(cfg, msgs, gen, False, None)
        payload["stream"] = True
        async with httpx.AsyncClient(base_url=cfg.base_url,
                                     timeout=cfg.timeout, headers=headers) as http:
            try:
                async with http.stream("POST", _CHAT_PATH, json=payload) as r:
                    if r.status_code in (401, 403):
                        raise AuthError("鉴权失败，请检查 API Key", r.status_code)
                    if r.status_code >= 400:
                        body = (await r.aread()).decode("utf-8", "ignore")
                        raise ProviderError(f"HTTP {r.status_code}：{body[:200]}",
                                            r.status_code)
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            break
                        try:
                            d = json.loads(chunk)
                            delta = d["choices"][0].get("delta", {})
                            piece = delta.get("content")
                            if piece:
                                yield piece
                        except (ValueError, KeyError, IndexError):
                            continue
            except httpx.HTTPError as exc:
                raise NetworkError(f"流式网络错误：{type(exc).__name__}") from exc

    async def chat_json(
        self,
        messages: Any,
        *,
        system: str | None = None,
        schema_hint: str = "",
        gen: GenParams | None = None,
        channel: str | None = None,
        retries: int = 2,
    ) -> Any:
        """要求模型返回 JSON，自动解析为 Python 对象。

        通过 system 强约束 + response_format=json_object + 容错提取，
        多次尝试直到拿到合法 JSON。schema_hint 描述期望结构（自然语言或 JSON 样例）。
        """
        sys_parts = ["你只能输出严格的 JSON，不要输出任何解释、Markdown 代码块或额外文字。"]
        if schema_hint:
            sys_parts.append(f"JSON 结构要求：{schema_hint}")
        sys_text = (system + "\n" if system else "") + "\n".join(sys_parts)
        msgs = self._norm(messages, sys_text)
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                if channel:
                    res = await self._run_on(channel, self._do_chat, msgs,
                                             gen or GenParams(temperature=0.1))
                else:
                    res = await self._run(self._do_chat, msgs,
                                          gen or GenParams(temperature=0.1))
            except ProviderError as exc:
                # 个别厂商不支持 response_format，退化为纯提示词约束
                if "response_format" in str(exc.body or "") or exc.status_code == 400:
                    msgs2 = self._norm(messages, sys_text)
                    res = await self._run_json_no_format(channel, msgs2, gen)
                else:
                    raise
            parsed = _extract_json(res.text)
            if parsed is not None:
                return parsed
            last = JSONModeError("模型输出无法解析为 JSON")
            # 追加纠错指令再来一次
            msgs = msgs + [{"role": "assistant", "content": res.text[:500]},
                           {"role": "user", "content": "上一条不是合法 JSON。请只输出 JSON 本体。"}]
        raise JSONModeError(str(last))

    async def _run_json_no_format(self, channel, msgs, gen):
        # 临时构造不带 response_format 的调用
        async def _plain(client, cfg, m, g):
            data = await client.raw_completion(_build_payload(cfg, m, g, False, None))
            text, finish, usage = _parse_choice(data)
            return ChatResult(text=text, model=data.get("model", cfg.model),
                              provider=cfg.provider, raw=data)
        if channel:
            return await self._run_on(channel, _plain, msgs, gen or GenParams(temperature=0.1))
        return await self._run(_plain, msgs, gen or GenParams(temperature=0.1))

    async def ping(self, channel: str | None = None) -> dict:
        """连接测试：发一句最小请求，返回 {ok, model, latency_ms, error}。"""
        import time
        t0 = time.time()
        try:
            res = await self.chat("ping", gen=GenParams(max_tokens=4),
                                  channel=channel)
            return {"ok": True, "model": res.model, "provider": res.provider,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "reply": res.text[:20]}
        except AIKitError as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": int((time.time() - t0) * 1000)}

    @staticmethod
    def _norm(messages: Any, system: str | None) -> list[dict]:
        if isinstance(messages, str):
            msgs = [Message("user", messages).to_dict()]
        elif isinstance(messages, list):
            msgs = [Message.normalize(m).to_dict() for m in messages]
        else:
            msgs = [Message("user", str(messages)).to_dict()]
        if system:
            msgs.insert(0, Message("system", system).to_dict())
        return msgs


def _extract_json(text: str) -> Any:
    """从模型输出中容错提取 JSON：先直接 parse，再剥离 ```json 代码块，再抓首个 {...}/[...]。"""
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except ValueError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        j = s.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except ValueError:
                continue
    return None
