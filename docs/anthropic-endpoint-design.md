# WebAI-to-API Anthropic Messages 端点设计纪要

> 日期：2026-05-20
> 状态：待实施

## 背景

WebAI-to-API 目前只支持 OpenAI Chat Completions 格式（`/v1/chat/completions`）。Claude Code 等 Anthropic 生态工具需要 Anthropic Messages API 格式（`/v1/messages`），无法直接使用 WebAI。

## 目标

新增 Anthropic Messages 兼容端点，使 Claude Code 能通过 WebAI 调用 Gemini Web。

## 现有目录结构

```text
E:\git-project\WebAI-to-API\
├── config.conf                              — Cookie/代理配置
├── config.models.json                       — 模型注册表
├── pyproject.toml                           — 项目依赖
├── src/
│   ├── app/
│   │   ├── main.py                          — FastAPI app 入口，注册路由
│   │   ├── config.py                        — 读 config.conf
│   │   ├── logger.py                        — 日志
│   │   ├── endpoints/
│   │   │   ├── chat.py                      — /v1/models + /v1/chat/completions + /translate + /v1/gems
│   │   │   ├── gemini.py                    — /gemini + /gemini-chat
│   │   │   ├── google_generative.py         — Google Generative AI 格式端点
│   │   │   └── init.py
│   │   ├── services/
│   │   │   ├── gemini_client.py             — Gemini 客户端初始化/获取单例
│   │   │   └── session_manager.py           — /translate 用的 session manager
│   │   └── utils/
│   │       └── browser.py                   — 从浏览器读 cookie
│   ├── models/
│   │   └── gemini.py                        — ModelRegistry、DailyModelDiscovery、MyGeminiClient、resolve_model_name
│   ├── schemas/
│   │   └── request.py                       — OpenAIChatRequest、GeminiRequest、GoogleGenerativeRequest 的 Pydantic model
│   └── run.py                               — uvicorn 启动入口
├── scripts/
│   └── discover-gemini-web-models.mjs       — 自动发现 Gemini Web 模型
├── tests/
│   ├── test_model_registry.py               — 模型注册表 + 每日发现 + 未知模型测试
│   ├── test_chat_cleanup.py                 — 聊天临时会话清理测试
│   ├── test_logger.py                       — 日志测试
│   └── test_browser_cdp_port.py             — CDP 端口测试
└── logs/                                    — 运行日志（gitignore）
```

## 新增文件

### `src/app/endpoints/anthropic.py`

新路由 `POST /v1/messages`，放在 `endpoints/` 下与 `chat.py`、`gemini.py`、`google_generative.py` 同级。

职责：
- **请求转换**：Anthropic Messages → OpenAI Chat Completions
  - `system` 顶层字段 → `role: system` 消息
  - `tool_use` content block → `tool_calls`
  - `tool_result` content block → `role: tool` 消息
  - `max_tokens` → 透传
  - `stream: true/false` → 透传
- **响应转换**：OpenAI 格式 → Anthropic 格式
  - 非流式：`choices[0].message` → Anthropic `content` blocks + `usage`
  - 流式：OpenAI SSE → Anthropic SSE（`message_start` / `content_block_delta` / `message_delta` / `message_stop`）
  - 停止原因：`stop` → `end_turn`，`tool_calls` → `tool_use`
- **模型校验**：复用 `ModelRegistry`，未知模型返回 Anthropic 格式错误
- **每日刷新**：调用 `refresh_models_if_needed()`

### `src/schemas/anthropic_request.py`

Anthropic Messages 请求的 Pydantic model，与 `request.py`（OpenAI schema）同级。

字段：
- `model`, `messages`, `system`, `max_tokens`, `tools`, `stream`
- messages 里的 content block 类型（text / tool_use / tool_result）

### `tests/test_anthropic_endpoint.py`

测试覆盖：
- 非流式请求 → 正常回复
- 流式请求 → SSE 格式正确
- tool_use 场景
- 未知模型 → Anthropic 格式错误
- system 消息转换

## 改动文件

### `src/app/main.py`

加两行：
```python
from app.endpoints import anthropic
app.include_router(anthropic.router)
```

## 不改的文件

```text
src/app/endpoints/chat.py       — 现有 OpenAI 端点不动
src/app/endpoints/gemini.py     — 不动
src/app/endpoints/google_generative.py — 不动
src/models/gemini.py            — ModelRegistry 已支持，不动
src/schemas/request.py          — 现有 OpenAI schema 不动
scripts/                        — 不动
config.models.json              — 不动
```

## 参考实现

One API（songquanpeng/one-api，Go）：
- `relay/adaptor/anthropic/` — Anthropic 请求/响应转换
- `relay/adaptor/openai/` — OpenAI 标准格式
- `relay/constant/` — 模型类型常量

## 核心转换逻辑

### 请求方向：Anthropic → OpenAI

```text
Anthropic                          OpenAI
─────────────────────────────────   ─────────────────────────────────
system: "..."                   →   messages[0] = {role: "system", content: "..."}
messages[i].content = [             messages[i] =
  {type: "text", text: "..."}  →     {role: "user"/"assistant", content: "..."}
  {type: "tool_use", ...}      →     {role: "assistant", tool_calls: [{...}]}
  {type: "tool_result", ...}   →     {role: "tool", content: "...", tool_call_id: "..."}
]
tools: [{name, description,     →   tools: [{type: "function", function: {name, description, parameters}}]
  input_schema: {...}}]
max_tokens: 8192                →   max_tokens: 8192
stream: true                    →   stream: true
```

### 响应方向：OpenAI → Anthropic（非流式）

```text
OpenAI                             Anthropic
─────────────────────────────────   ─────────────────────────────────
choices[0].message.content     →   content: [{type: "text", text: "..."}]
choices[0].finish_reason=stop  →   stop_reason: "end_turn"
choices[0].finish_reason=      →   stop_reason: "tool_use"
  tool_calls
usage.prompt_tokens            →   usage.input_tokens
usage.completion_tokens        →   usage.output_tokens
```

### 响应方向：OpenAI → Anthropic（流式）

```text
OpenAI SSE                         Anthropic SSE
─────────────────────────────────   ─────────────────────────────────
(首条)                          →   event: message_start
                                     data: {type: "message", id, role: "assistant", content: [], model, usage}
choices[0].delta.content       →   event: content_block_start
                                     data: {type: "content_block_start", index: 0, content_block: {type: "text", text: ""}}
                                  → event: content_block_delta
                                     data: {type: "content_block_delta", index: 0, delta: {type: "text_delta", text: "..."}}
choices[0].delta.tool_calls    →   event: content_block_start
                                     data: {type: "content_block_start", index: N, content_block: {type: "tool_use", id, name}}
                                  → event: content_block_delta
                                     data: {type: "content_block_delta", index: N, delta: {type: "input_json_delta", partial_json: "..."}}
choices[0].finish_reason       →   event: message_delta
                                     data: {type: "message_delta", delta: {stop_reason: "end_turn"/"tool_use"}, usage: {output_tokens: N}}
data: [DONE]                   →   event: message_stop
```

## 风险点

1. **流式 SSE 格式**：Anthropic 流式事件结构比 OpenAI 复杂，Claude Code 对此要求严格，最容易出 bug
2. **tool_use / tool_result**：Anthropic 多 content block 模式和 OpenAI `tool_calls` 结构差异大，嵌套场景容易漏
3. **Claude Code 容错**：对响应格式可能很严格，一点偏差就报错，需要实跑验证
4. **thinking/extended thinking**：Anthropic 有 `thinking` content block 类型，Claude Code 开 thinking 时可能发送/期望此类型，需确认是否要处理

## 实施步骤

1. 先加非流式端点 + 基本流式，用 curl 手动验证格式正确
2. 再接 Claude Code 实际跑，看报什么错再修
3. 补充 tool_use 场景测试
