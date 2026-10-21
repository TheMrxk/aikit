"""最小示例：环境变量配置后直接运行。

export AIKIT_PROVIDER=deepseek
export AIKIT_API_KEY=sk-xxxx
python examples/quickstart.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aikit import AI, ConfigError, GenParams  # noqa: E402


async def main() -> None:
    ai = AI()
    # 1) 连接测试
    ping = await ai.ping()
    print("ping:", ping)
    if not ping["ok"]:
        print("未配置或 key 无效：请先设置 AIKIT_API_KEY 或用测试台配置。")
        return

    # 2) 普通对话
    r = await ai.chat("用一句话解释什么是 RAG",
                      gen=GenParams(max_tokens=100))
    print("\n对话：", r.text)
    print(f"模型 {r.model} ｜ tokens {r.usage.prompt}+{r.usage.completion}")

    # 3) JSON 模式（结构化抽取/答题）
    data = await ai.chat_json(
        "单选题：下列哪个是检索增强生成？A RAG B SQL C HTTP。只输出 JSON。",
        schema_hint='{"answer": "A/B/C", "reason": "不超过20字"}')
    print("\nJSON：", data)

    # 4) 流式
    print("\n流式：", end="", flush=True)
    async for piece in ai.stream("用10个字形容秋天", gen=GenParams(max_tokens=40)):
        print(piece, end="", flush=True)
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ConfigError as exc:
        print("配置错误：", exc)
