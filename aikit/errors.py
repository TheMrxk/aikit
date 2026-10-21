"""aikit 异常体系。

设计原则：
- 调用方只需 catch AIKitError 即可兜住所有库内异常；
- 细分类型便于区分「配置问题 / 鉴权问题 / 限流 / 网络 / 协议 / 内容」并决定是否重试；
- 所有异常都不携带明文密钥（日志/上报安全）。
"""
from __future__ import annotations

from typing import Any


class AIKitError(Exception):
    """所有 aikit 异常的基类。"""


class ConfigError(AIKitError):
    """配置缺失或非法（未配置 provider、base_url/model 为空、配置文件损坏等）。

    这类错误重试无意义，应提示用户去配置。
    """


class AuthError(AIKitError):
    """鉴权失败（HTTP 401/403、key 无效或欠费）。不应自动重试。"""

    def __init__(self, message: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(AIKitError):
    """触发限流（HTTP 429）。可退避后重试。"""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderError(AIKitError):
    """服务端错误（HTTP 5xx / 网关错误）。可重试。"""

    def __init__(self, message: str, status_code: int = 500, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class NetworkError(AIKitError):
    """网络层错误（连接超时、DNS、断连）。可重试。"""


class ProtocolError(AIKitError):
    """响应不符合预期（非 JSON、缺字段、SSE 流异常等）。"""


class JSONModeError(AIKitError):
    """要求 JSON 输出但模型始终未返回可解析 JSON（重试预算耗尽）。"""


class AllProvidersFailed(AIKitError):
    """配置了多个 provider 做故障转移，但全部失败。

    errors: [(provider_name, exception), ...]，保留每一家的失败原因。
    """

    def __init__(self, errors: list[tuple[str, BaseException]]):
        self.errors = errors
        detail = "; ".join(f"{name}: {type(exc).__name__}" for name, exc in errors)
        super().__init__(f"所有模型通道均失败：{detail}")
