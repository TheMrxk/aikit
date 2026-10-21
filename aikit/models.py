"""aikit 数据模型：消息、模型配置、生成参数、返回结构。

全部是 dataclass，不依赖 pydantic，保证模块在任何 Python 3.9+ 环境零阻力集成。
对外也允许直接传 dict（Client 内部会归一化）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterator, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """一条对话消息。OpenAI 兼容协议的最小通用结构。"""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @staticmethod
    def normalize(x: Any) -> "Message":
        if isinstance(x, Message):
            return x
        if isinstance(x, dict):
            return Message(role=x.get("role", "user"), content=str(x.get("content", "")))
        # 裸字符串当 user
        return Message(role="user", content=str(x))


@dataclass
class ModelConfig:
    """一个可用模型通道的完整配置。

    - 标准 OpenAI 兼容服务只需要 api_key + base_url + model；
    - provider 仅用于预设填充与日志展示，自定义端点用 "custom"；
    - extra_headers/extra_body 透传给后端，兼容 coding plan / 中转的特殊字段。
    """

    api_key: str = ""
    base_url: str = ""
    model: str = ""
    provider: str = "custom"
    # 可选的备用模型名列表（主名不存在时尝试）
    models: list[str] = field(default_factory=list)
    timeout: float = 60.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    # 该通道的显示别名（多通道故障转移时区分）
    label: str = ""

    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass
class GenParams:
    """生成参数。None 表示不传给后端，使用其默认值。"""

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop: list[str] | str | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @staticmethod
    def from_raw(d: dict | None) -> "Usage":
        d = d or {}
        return Usage(
            prompt_tokens=int(d.get("prompt_tokens") or 0),
            completion_tokens=int(d.get("completion_tokens") or 0),
            total_tokens=int(d.get("total_tokens") or 0),
        )


@dataclass
class ChatResult:
    """一次非流式对话的返回。"""

    text: str
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)


StreamChunk = Iterator[str]  # 流式逐段文本
