"""aikit —— 通用 AI 接入层。

一个独立、零业务依赖、OpenAI 兼容的统一模型封装，目标是让任何项目
装上即用，不必逐家厂商对接、不必让每个终端用户配置。

快速开始
--------
服务端 .env 配置（全员零配置）::

    AIKIT_PROVIDER=deepseek
    AIKIT_API_KEY=sk-xxxx
    AIKIT_MODEL=deepseek-chat

代码::

    import asyncio
    from aikit import AI

    async def main():
        ai = AI()
        print((await ai.chat("你好，一句话自我介绍")).text)
        data = await ai.chat_json(
            "从这句话提取姓名和城市：我叫张三，住在沈阳",
            schema_hint='{"name": str, "city": str}')
        print(data)

    asyncio.run(main())

也可以用 ConfigStore 在前台设置页配置多通道、设默认、做连接测试。
"""
from .client import AI
from .config import ConfigStore, StoredChannel, mask_key
from .errors import (
    AIKitError, AllProvidersFailed, AuthError, ConfigError, JSONModeError,
    NetworkError, ProtocolError, ProviderError, RateLimitError,
)
from .models import ChatResult, GenParams, Message, ModelConfig, Usage
from .presets import ALL_PRESETS, get_preset

__version__ = "0.1.0"

__all__ = [
    "AI",
    "ConfigStore",
    "StoredChannel",
    "mask_key",
    "Message",
    "ModelConfig",
    "GenParams",
    "ChatResult",
    "Usage",
    "ALL_PRESETS",
    "get_preset",
    "AIKitError",
    "ConfigError",
    "AuthError",
    "RateLimitError",
    "ProviderError",
    "NetworkError",
    "ProtocolError",
    "JSONModeError",
    "AllProvidersFailed",
    "__version__",
]
