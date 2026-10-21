# aikit · 通用 AI 接入层

一个独立、零业务依赖的 Python 模块，用同一套接口调用**任何 OpenAI 兼容的大模型**——
主流厂商、Coding Plan、中转站、自建服务。目标：**任何程序装上就能用 AI，不必逐家对接、不必让每个用户配置。**

- 🔌 一套接口通吃：小米 MiMo、DeepSeek、Kimi、通义、智谱、MiniMax、豆包、OpenAI、OpenRouter、火山 Coding Plan、任意中转站
- ⚙️ 三种配置来源：环境变量（服务端全员零配置）／ JSON 文件 ／ 前台设置页
- 🧩 专为「答题/结构化」准备的 `chat_json`：强约束 + 容错提取 + 重试，稳出 JSON
- 🌊 流式 `stream`、普通 `chat`、连接测试 `ping`
- 🛡 自动重试（429/5xx/网络，指数退避）+ 多通道故障转移
- 🔐 密钥 0600 存储、只回脱敏、日志不带 key
- 🪶 核心只依赖 `httpx`；测试台另需 fastapi/uvicorn

---

## 安装

```bash
# 方式一：本地可编辑安装（推荐，方便跟随更新）
cd aikit && pip install -e .

# 带 Web 测试台
pip install -e ".[web]"

# 方式二：直接拷进项目（核心无第三方框架依赖，除 httpx）
cp -r aikit /your-project/aikit
```

---

## 30 秒上手

### 1. 配置（三选一或叠加）

**A. 环境变量 / .env（部署时配一次，所有用户零配置）**
```bash
export AIKIT_PROVIDER=deepseek
export AIKIT_API_KEY=sk-xxxx
export AIKIT_MODEL=deepseek-chat        # 不填则用该厂商默认模型
```

**B. JSON 文件**（默认 `~/.aikit/config.json`，可用 `AIKIT_CONFIG` 改路径）
```json
{
  "default": "main",
  "providers": {
    "main": { "provider": "deepseek", "api_key": "sk-xxxx", "model": "deepseek-chat" },
    "backup": { "provider": "xiaomi", "api_key": "sk-yyyy", "model": "mimo-v2-pro" }
  }
}
```

**C. 代码 / 前台页面**（见下方「在 Web 应用里嵌入设置页」）

### 2. 调用

```python
import asyncio
from aikit import AI, GenParams

async def main():
    ai = AI()                                  # 自动从 env/文件读配置

    # 普通对话
    r = await ai.chat("用一句话解释什么是 RAG")
    print(r.text, "| tokens:", r.usage.total_tokens)

    # 强制 JSON（答题、抽取、分类神器）
    data = await ai.chat_json(
        "RAG 属于以下哪类技术？A 检索增强 B 图形渲染 C 操作系统",
        schema_hint='{"answer": "A/B/C/D", "reason": "简短理由"}')
    print(data["answer"])

    # 流式
    async for piece in ai.stream("写一首关于沈阳的短诗"):
        print(piece, end="", flush=True)

asyncio.run(main())
```

---

## 核心 API

| 方法 | 说明 |
|---|---|
| `AI().chat(messages, system=, gen=, channel=)` | 对话，返回 `ChatResult{text, model, provider, usage, ...}` |
| `AI().chat_json(messages, schema_hint=, retries=)` | 强约束 JSON，自动解析为 Python 对象 |
| `AI().stream(messages, ...)` | async 生成器，逐段产出文本 |
| `AI().ping(channel=)` | 连接测试 `{ok, model, latency_ms, error}` |
| `ConfigStore().upsert_channel(...)` | 保存通道（前台设置页用） |
| `ConfigStore().public_view()` | 脱敏配置视图（给前端，key 只回 `sk-1****cdef`） |

`messages` 支持：裸字符串、`[{"role","content"}]` dict、`Message` 对象。
生成参数用 `GenParams(temperature=0.1, max_tokens=500, top_p=...)`，传 `None` 的字段不下发。

### 同步代码里用
模块是 async 的；同步环境包一层：
```python
import asyncio
text = asyncio.run(AI().chat("你好")).text
```

---

## 在 Web 应用里嵌入「AI 设置页」

让用户在前台选厂商、贴 Key、点测试，完全不用懂 base_url：

```python
from aikit import AI, ConfigStore, StoredChannel

store = ConfigStore()                       # 路径用 AIKIT_CONFIG 指定到你的数据目录

# 保存用户填的通道（provider 选预设，base_url/model 自动补）
store.upsert_channel("main", StoredChannel(
    api_key=req.api_key, provider=req.provider, model=req.model,
    base_url=req.base_url), make_default=True)
store.save()

# 业务里直接用（前台 key 优先于服务端 env）
answer = await AI(config=store).chat_json(question, schema_hint=...)

# 给前端的安全视图（不含明文 key）
safe = store.public_view()
# 厂商预设表（填充下拉与默认值）：from aikit.presets import ALL_PRESETS
```

完整可抄的后端 + 前端见 [`examples/server.py`](examples/server.py) 与 [`web/index.html`](web/index.html)，
启动后是一个现成的配置/测试台：

```bash
python -m examples.server        # http://localhost:8899
```

---

## 配置优先级与故障转移

- 单次调用若不指定 `channel`：**环境变量通道优先**，其次配置文件默认通道，再其余启用通道。
- `auto_failover=True`（默认）时，主通道遇到 401/429/5xx/网络错误会**自动切到下一个启用通道**；
  全挂抛 `AllProvidersFailed`（含每家原因）。指定 `channel=` 则只用该通道、不转移。
- 重试：429（尊重 `Retry-After`）/5xx/网络错误指数退避，默认每通道重试 2 次；401/参数错不重试。

---

## 预设厂商

见 `aikit/presets/providers.py`，含 base_url / 默认模型 / 申请地址 / 踩坑提示。已内置：

小米 MiMo · DeepSeek · Kimi · 阿里通义（compatible-mode）· 智谱 GLM · MiniMax ·
火山豆包 · **火山 Coding Plan（/api/coding/v3，一个 key 调多家模型）** · OpenAI ·
OpenRouter（统一路由）· 自定义/中转站。

> 增加厂商通常只需在预设表加一条，**无需改任何调用代码**。
> 自定义 OpenAI 兼容端点：provider 选 `custom`，填它给的 `base_url`（一般以 `/v1` 结尾）、`model`、`api_key` 即可。

### 已知坑（已写进预设 notes）
- 小米 token-plan 必须用 `token-plan-cn.xiaomimimo.com`，`api.xiaomimimo.com/v1` 不通。
- 阿里百炼要走 `/compatible-mode/v1` 才是 OpenAI 兼容协议。
- 火山 coding plan 与标准推理端点路径不同（`/api/coding/v3` vs `/api/v3`）。
- Anthropic / Kimi-coding 的 anthropic 端点是**非 OpenAI 协议**，当前版本不适配（已在预设标注），后续可加协议适配层。

---

## 异常处理

```python
from aikit import AuthError, RateLimitError, ProviderError, NetworkError, ConfigError, AllProvidersFailed

try:
    await AI().chat("hi")
except ConfigError:        # 没配置 → 引导用户去设置页
except AuthError:          # key 错/欠费 → 提示更换
except RateLimitError:     # 限流（可退避）
except ProviderError:      # 4xx/5xx
except AllProvidersFailed: # 多通道全挂
```

---

## 安全

- 配置文件创建为 `0600`（仅属主可读写）。
- `public_view()` / `mask_key()` 只回 `sk-1****cdef`，接口绝不返回明文 key。
- 异常与内部请求不带密钥；建议业务日志也只记 provider/model，不记 key。

---

## 测试

```bash
python tests/test_core.py     # 或 pytest tests/ -q
```

单测不发起真实网络请求（JSON 提取、多源配置、脱敏、payload、错误分类等）。

---

## 路线图

- [ ] 首个真实业务接入：ccb-course-bot 随机抽题卷的 AI 整卷作答 + 项目内「AI 设置」页
- [ ] Anthropic Messages 协议适配层
- [ ] 响应 JSON Schema（structured outputs）按厂商能力自动降级
- [ ] 用量统计 / 成本估算 / per-channel 速率与冷却
- [ ] 同步便捷封装（`AI.chat_sync`）、embeddings、图片多模态接口

## License

MIT
