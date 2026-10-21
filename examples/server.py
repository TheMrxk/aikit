"""aikit 独立测试台（零依赖 FastAPI 演示 + 接入参考）。

用途：
  1. 验证 aikit 模块本体（配置多通道 / 测试连接 / 对话 / JSON 模式）；
  2. 作为「在你自己项目里嵌入 AI 设置页」的后端参考实现。

启动：
  pip install fastapi uvicorn
  python -m examples.server          # 默认 0.0.0.0:8899
  AIKIT_WEB_PORT=9000 python -m examples.server

数据：写入 AIKIT_CONFIG（默认 ~/.aikit/config.json），密钥 0600 存储，
接口只回脱敏 key_hint，不回明文。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from aikit import AI, ConfigStore, StoredChannel, __version__  # noqa: E402
from aikit.presets import ALL_PRESETS  # noqa: E402

PORT = int(os.environ.get("AIKIT_WEB_PORT", "8899"))
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="aikit 测试台")
store = ConfigStore()  # 读 AIKIT_CONFIG / 环境变量


class ChannelIn(BaseModel):
    name: str
    provider: str = "custom"
    api_key: str = ""          # 空表示保留原 key（编辑时不回显）
    model: str = ""
    base_url: str = ""
    label: str = ""
    make_default: bool = False
    enabled: bool = True


# ---------- 页面 ----------
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/presets")
async def presets() -> dict:
    return {"ok": True, "presets": ALL_PRESETS}


@app.get("/api/config")
async def get_config() -> dict:
    """返回脱敏后的配置视图（绝不回明文 key）。"""
    view = store.public_view()
    return {"ok": True, **view}


@app.post("/api/channels")
async def upsert(ch: ChannelIn) -> dict:
    name = ch.name.strip()
    if not name:
        return JSONResponse({"ok": False, "error": "通道名必填"}, status_code=400)
    existing = store.channels.get(name)
    api_key = ch.api_key.strip()
    if not api_key:
        # 编辑场景：前端不回显 key，留空则保留旧 key
        if existing:
            api_key = existing.api_key
    if not api_key:
        return JSONResponse({"ok": False, "error": "API Key 必填"}, status_code=400)
    channel = StoredChannel(
        api_key=api_key, provider=ch.provider, model=ch.model.strip(),
        base_url=ch.base_url.strip(), label=ch.label.strip() or name,
        enabled=ch.enabled,
    )
    store.upsert_channel(name, channel, make_default=ch.make_default)
    store.save()
    return {"ok": True, "config": store.public_view()}


@app.post("/api/channels/{name}/default")
async def set_default(name: str) -> dict:
    try:
        store.set_default(name)
        store.save()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


@app.post("/api/channels/{name}/toggle")
async def toggle(name: str) -> dict:
    ch = store.channels.get(name)
    if not ch:
        return JSONResponse({"ok": False, "error": "通道不存在"}, status_code=404)
    ch.enabled = not ch.enabled
    store.save()
    return {"ok": True, "enabled": ch.enabled}


@app.delete("/api/channels/{name}")
async def delete(name: str) -> dict:
    store.remove_channel(name)
    store.save()
    return {"ok": True}


@app.post("/api/test")
async def test_connection(body: dict) -> dict:
    """测试连接。body: {channel?: 已存通道名} 或内联 {provider,api_key,model,base_url}。"""
    try:
        if body.get("channel"):
            result = await AI(config=store, auto_failover=False).ping(body["channel"])
        else:
            # 临时通道不落盘
            tmp = ConfigStore(path=_tmp_path(), use_env=False)
            tmp.upsert_channel("_test", StoredChannel(
                api_key=body.get("api_key", ""),
                provider=body.get("provider", "custom"),
                model=body.get("model", ""),
                base_url=body.get("base_url", "")))
            result = await AI(config=tmp, auto_failover=False).ping("_test")
        return {"ok": result["ok"], "result": result}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "result": {"ok": False, "error": str(exc)}}


_TMP_FILE: str | None = None


def _tmp_path() -> str:
    global _TMP_FILE
    import tempfile
    if not _TMP_FILE:
        fd, _TMP_FILE = tempfile.mkstemp(prefix="aikit_", suffix=".json")
        os.close(fd)
    return _TMP_FILE


class ChatIn(BaseModel):
    message: str
    system: str = ""
    channel: str | None = None
    json_mode: bool = False
    schema_hint: str = ""


@app.post("/api/chat")
async def chat(c: ChatIn) -> dict:
    try:
        ai = AI(config=store, auto_failover=bool(not c.channel))
        if c.json_mode:
            data = await ai.chat_json(
                c.message, system=c.system or None,
                schema_hint=c.schema_hint, channel=c.channel)
            return {"ok": True, "json": data}
        res = await ai.chat(c.message, system=c.system or None,
                            channel=c.channel)
        return {"ok": True, "text": res.text, "model": res.model,
                "provider": res.provider,
                "usage": {"prompt": res.usage.prompt_tokens,
                          "completion": res.usage.completion_tokens}}
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=200)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "version": __version__}


if __name__ == "__main__":
    import uvicorn
    print(f"aikit 测试台 v{__version__} → http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
