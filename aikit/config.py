"""配置层：多源合并 + 持久化 + key 脱敏。

三个配置来源（优先级从高到低）：
  1. 编程式显式传入 / 前台页面保存（写入 JSON 文件或应用自己的库）
  2. JSON 配置文件（默认 ~/.aikit/config.json，可用 AIKIT_CONFIG 指定）
  3. 环境变量（服务端 .env 部署最常用，全员零配置）

环境变量约定：
  AIKIT_PROVIDER   厂商 id（如 deepseek/xiaomi/custom）
  AIKIT_BASE_URL   自定义/覆盖 base_url
  AIKIT_API_KEY    密钥
  AIKIT_MODEL      模型名
  AIKIT_TIMEOUT    超时秒
  AIKIT_EXTRA_MODEL 备用模型，逗号分隔

JSON 文件结构（也作为前台设置页的持久化格式）：
  {
    "default": "deepseek",
    "providers": {
      "deepseek": {"api_key": "sk-...", "model": "deepseek-chat",
                   "base_url": "...（可省略，用预设）", "label": "主力"},
      "custom-x": {"provider": "custom", "api_key": "...", "base_url": "...", "model": "..."}
    }
  }
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .models import ModelConfig
from .presets import base_url_for, default_model, get_preset

DEFAULT_CONFIG_PATH = os.environ.get(
    "AIKIT_CONFIG", str(Path.home() / ".aikit" / "config.json"))


def mask_key(key: str) -> str:
    """脱敏展示：保留头4尾4，中间打码。空/过短整体打码，绝不回显明文。"""
    if not key:
        return ""
    s = str(key)
    if len(s) <= 10:
        return "*" * len(s)
    return f"{s[:4]}{'*' * max(4, len(s) - 8)}{s[-4:]}"


@dataclass
class StoredChannel:
    """一个已配置通道（持久化形态）。"""

    api_key: str = ""
    provider: str = "custom"
    model: str = ""
    base_url: str = ""
    label: str = ""
    timeout: float = 60.0
    models: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_key": self.api_key, "provider": self.provider,
            "model": self.model, "base_url": self.base_url,
            "label": self.label, "timeout": self.timeout,
            "models": self.models, "enabled": self.enabled,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StoredChannel":
        return StoredChannel(
            api_key=str(d.get("api_key") or ""),
            provider=str(d.get("provider") or "custom"),
            model=str(d.get("model") or ""),
            base_url=str(d.get("base_url") or ""),
            label=str(d.get("label") or ""),
            timeout=float(d.get("timeout") or 60.0),
            models=list(d.get("models") or []),
            enabled=bool(d.get("enabled", True)),
        )


class ConfigStore:
    """配置仓库：内存 + JSON 文件持久化 + 环境变量兜底。"""

    def __init__(self, path: str | None = None, use_env: bool = True):
        self.path = path or DEFAULT_CONFIG_PATH
        self.use_env = use_env
        self.channels: dict[str, StoredChannel] = {}
        self.default: str = ""
        self.load()

    # ---------- 持久化 ----------
    def load(self) -> "ConfigStore":
        # 以磁盘为唯一真相源：每次重载先清空内存，避免文件被删/改名后
        # 仍残留旧通道（否则开通检测会一直返回已配置）。
        self.channels = {}
        self.default = ""
        if os.path.exists(self.path):
            try:
                text = Path(self.path).read_text(encoding="utf-8").strip()
                if not text:
                    return self            # 空文件视为空配置（mkstemp 临时文件等）
                raw = json.loads(text)
                self.default = str(raw.get("default") or "")
                for name, d in (raw.get("providers") or {}).items():
                    self.channels[name] = StoredChannel.from_dict(d)
            except (ValueError, OSError) as exc:
                raise ConfigError(f"配置文件损坏 {self.path}: {exc}") from exc
        return self

    def save(self, path: str | None = None) -> str:
        path = path or self.path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "default": self.default,
            "providers": {k: v.to_dict() for k, v in self.channels.items()},
        }
        # 0600：含密钥，仅属主可读写
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    # ---------- 通道管理（前台设置页用） ----------
    def upsert_channel(self, name: str, channel: StoredChannel,
                       make_default: bool = False) -> None:
        # 补预设默认值
        if not channel.base_url:
            channel.base_url = base_url_for(channel.provider)
        if not channel.model:
            channel.model = default_model(channel.provider)
        self.channels[name] = channel
        if make_default or not self.default:
            self.default = name

    def remove_channel(self, name: str) -> None:
        self.channels.pop(name, None)
        if self.default == name:
            self.default = next(iter(self.channels), "")

    def set_default(self, name: str) -> None:
        if name not in self.channels:
            raise ConfigError(f"通道不存在：{name}")
        self.default = name

    # ---------- 解析为可用的 ModelConfig ----------
    def _env_channel(self) -> StoredChannel | None:
        if not self.use_env or not os.environ.get("AIKIT_API_KEY"):
            return None
        provider = os.environ.get("AIKIT_PROVIDER", "custom")
        return StoredChannel(
            api_key=os.environ["AIKIT_API_KEY"],
            provider=provider,
            model=os.environ.get("AIKIT_MODEL", default_model(provider)),
            base_url=os.environ.get("AIKIT_BASE_URL", base_url_for(provider)),
            timeout=float(os.environ.get("AIKIT_TIMEOUT", "60")),
            models=[m for m in os.environ.get("AIKIT_EXTRA_MODEL", "").split(",") if m],
            label="env",
        )

    def resolve(self, name: str | None = None) -> ModelConfig:
        """解析单个通道为 ModelConfig。name 为空取默认；env 优先级最高（当未显式指定）。"""
        if name is None:
            env = self._env_channel()
            if env:
                return self._to_model("env", env)
            name = self.default
        ch = self.channels.get(name)
        if ch is None:
            raise ConfigError(
                "未配置可用的 AI 通道（缺少 api_key/base_url/model）。"
                "可在设置页配置或设置 AIKIT_API_KEY 等环境变量。")
        if not ch.enabled:
            raise ConfigError(f"通道 {name} 已禁用")
        return self._to_model(name, ch)

    def resolve_all_enabled(self) -> list[tuple[str, ModelConfig]]:
        """用于故障转移：默认通道在前，其余启用通道随后。"""
        env = self._env_channel()
        out: list[tuple[str, ModelConfig]] = []
        if env:
            out.append(("env", self._to_model("env", env)))
        order = [self.default] + [k for k in self.channels if k != self.default]
        seen = set()
        for n in order:
            ch = self.channels.get(n)
            if ch and ch.enabled and n not in seen:
                seen.add(n)
                out.append((n, self._to_model(n, ch)))
        return out

    @staticmethod
    def _to_model(name: str, ch: StoredChannel) -> ModelConfig:
        base = ch.base_url or base_url_for(ch.provider)
        model = ch.model or default_model(ch.provider)
        if not ch.api_key or not base or not model:
            raise ConfigError(
                f"通道 {name} 配置不完整（需要 api_key/base_url/model）")
        return ModelConfig(
            api_key=ch.api_key, base_url=base.rstrip("/"),
            model=model, provider=ch.provider,
            models=ch.models, timeout=ch.timeout or 60.0,
            label=ch.label or name,
        )

    # ---------- 给设置页的安全视图（不含明文 key） ----------
    def public_view(self) -> dict[str, Any]:
        return {
            "default": self.default,
            "providers": [
                {"id": p["id"], "name": p["name"],
                 "default_model": (p.get("models") or [""])[0],
                 "base_url": p.get("base_url", ""),
                 "docs": p.get("docs", ""), "notes": p.get("notes", "")}
                for p in __import__("aikit.presets", fromlist=["ALL_PRESETS"]).ALL_PRESETS
            ],
            "channels": {
                name: {
                    "provider": ch.provider,
                    "model": ch.model,
                    "base_url": ch.base_url or base_url_for(ch.provider),
                    "label": ch.label, "enabled": ch.enabled,
                    "has_key": bool(ch.api_key), "key_hint": mask_key(ch.api_key),
                    "preset_name": (get_preset(ch.provider) or {}).get("name", ch.provider),
                }
                for name, ch in self.channels.items()
            },
        }
