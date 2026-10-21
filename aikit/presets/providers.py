"""厂商预设表。

这是 aikit「降低配置门槛」的核心：把各家 OpenAI 兼容端点的 base_url、
默认模型、申请地址预先填好，用户只需选厂商 + 填 api_key。

字段说明：
  base_url  指向 /v1（OpenAI 兼容根），client 会拼 /chat/completions
  models    该厂商常用模型（第一个为默认推荐）
  docs      申请/获取 key 的地址
  notes     踩坑提示（如 coding plan 专用端点、区域差异）

绝大多数主流厂商与 coding plan / 中转站都暴露同一套 OpenAI 兼容协议，
新增厂商通常只需追加一条预设即可，无需改任何调用代码。
自定义 / 中转站：用 provider="custom"，直接填它给的 base_url 与 model。
"""
from __future__ import annotations

# 国内厂商（稳定、低延迟、合规友好）
XIAOMI_MIMO = {
    "id": "xiaomi",
    "name": "小米 MiMo",
    "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
    "models": ["mimo-v2-pro", "mimo-v2.5-pro", "mimo-coding"],
    "docs": "https://www.xiaomimimo.com",
    "notes": "token-plan 套餐必须用 token-plan-cn.xiaomimimo.com，api.xiaomimimo.com/v1 不通用（已真机踩坑）。",
}

DEEPSEEK = {
    "id": "deepseek",
    "name": "DeepSeek 深度求索",
    "base_url": "https://api.deepseek.com/v1",
    "models": ["deepseek-chat", "deepseek-reasoner"],
    "docs": "https://platform.deepseek.com/api_keys",
    "notes": "性价比高，选择题/JSON 稳定；deepseek-reasoner 为推理模型。",
}

MOONSHOT = {
    "id": "moonshot",
    "name": "月之暗面 Kimi",
    "base_url": "https://api.moonshot.cn/v1",
    "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k", "kimi-k2-0905-preview"],
    "docs": "https://platform.moonshot.cn/console/api-keys",
    "notes": "",
}

DASHSCOPE = {
    "id": "dashscope",
    "name": "阿里云百炼 通义",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen2.5-72b-instruct"],
    "docs": "https://bailian.console.aliyun.com/?apiKey=1#/api-key",
    "notes": "务必用 compatible-mode 路径才是 OpenAI 兼容协议。",
}

ZHIPU = {
    "id": "zhipu",
    "name": "智谱 GLM",
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "models": ["glm-4-plus", "glm-4-flash", "glm-4.5", "glm-4-air"],
    "docs": "https://open.bigmodel.cn/usercenter/apikeys",
    "notes": "glm-4-flash 免费。",
}

MINIMAX = {
    "id": "minimax",
    "name": "MiniMax",
    "base_url": "https://api.minimaxi.com/v1",
    "models": ["MiniMax-M2", "MiniMax-Text-01", "abab6.5s-chat"],
    "docs": "https://platform.minimaxi.com/user-center/basic-information/interface-key",
    "notes": "",
}

DOUBAO = {
    "id": "doubao",
    "name": "火山引擎 豆包",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "models": ["doubao-pro-32k", "doubao-pro-128k", "doubao-1.5-pro-32k"],
    "docs": "https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey",
    "notes": "标准推理用 /api/v3；火山 coding plan 见 volcengine-coding 预设。",
}

# Coding Plan 专用端点（订阅制，key 与标准端点不同）
VOLC_CODING = {
    "id": "volcengine-coding",
    "name": "火山 Coding Plan（方舟）",
    "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
    "models": ["ark-code-latest", "doubao-seed-evolving", "deepseek-v4-pro",
               "kimi-k3", "glm-5.3", "minimax-m3"],
    "docs": "https://console.volcengine.com/ark",
    "notes": "coding plan 专用路径 /api/coding/v3，与标准 /api/v3 不同；一个 key 可用多家模型。",
}

KIMI_CODING = {
    "id": "kimi-coding",
    "name": "Kimi Coding Plan",
    "base_url": "https://api.moonshot.cn/anthropic",
    "models": ["kimi-k2-0905", "kimi-k3"],
    "docs": "https://platform.moonshot.cn",
    "notes": "该端点为 Anthropic 协议，非 OpenAI 兼容；当前版本暂不支持，请改用 moonshot 预设。",
}

# 国际厂商
OPENAI = {
    "id": "openai",
    "name": "OpenAI",
    "base_url": "https://api.openai.com/v1",
    "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-3.5-turbo", "o4-mini"],
    "docs": "https://platform.openai.com/api-keys",
    "notes": "国内服务器通常需可访问的网络或中转。",
}

ANTHROPIC = {
    "id": "anthropic",
    "name": "Anthropic Claude",
    "base_url": "https://api.anthropic.com/v1",
    "models": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-latest", "claude-3-haiku-20240307"],
    "docs": "https://console.anthropic.com/settings/keys",
    "notes": "Anthropic 使用自有协议（/v1/messages），非 OpenAI 兼容；当前版本未适配，预留。",
}

# 统一路由（一个 key 调几百家模型）
OPENROUTER = {
    "id": "openrouter",
    "name": "OpenRouter（统一路由）",
    "base_url": "https://openrouter.ai/api/v1",
    "models": ["anthropic/claude-sonnet-4", "openai/gpt-4o", "deepseek/deepseek-chat",
               "google/gemini-2.5-pro", "moonshotai/kimi-k2"],
    "docs": "https://openrouter.ai/keys",
    "notes": "model 必须带 厂商/模型 前缀；可作为统一聚合入口。",
}

CUSTOM = {
    "id": "custom",
    "name": "自定义 / 中转站（OpenAI 兼容）",
    "base_url": "",
    "models": [],
    "docs": "",
    "notes": "填中转站或自建服务给的 base_url（通常以 /v1 结尾）与 model，key 按对方要求。",
}

# 顺序即前端展示顺序
ALL_PRESETS = [
    XIAOMI_MIMO, DEEPSEEK, MOONSHOT, DASHSCOPE, ZHIPU, MINIMAX, DOUBAO,
    VOLC_CODING, OPENAI, OPENROUTER, CUSTOM,
]

PRESET_MAP = {p["id"]: p for p in ALL_PRESETS}


def get_preset(provider_id: str) -> dict | None:
    return PRESET_MAP.get(provider_id)


def default_model(provider_id: str) -> str:
    p = PRESET_MAP.get(provider_id)
    if p and p.get("models"):
        return p["models"][0]
    return ""


def base_url_for(provider_id: str) -> str:
    p = PRESET_MAP.get(provider_id)
    return (p or {}).get("base_url", "")
