# Gemini 工具调用兼容修复

修复 AstrBot 使用 Google GenAI 适配器连接部分 Gemini 兼容端点时，工具调用后持续返回以下错误的问题：

```text
400 antigravity executor: invalid Gemini function call history:
functionResponse.name "call_*" does not match functionCall.name "..."
```

## 原因

AstrBot 4.26.6 会保存 OpenAI 风格的工具历史：函数名位于 assistant 的
`tool_calls[].function.name`，工具结果只保存 `tool_call_id`。Google GenAI
适配器在工具结果没有 `name` 时，会错误地把 `call_*`、`call-*` 或
`toolu_*` 调用 ID 当成 `functionResponse.name`。

官方 Gemini API 的部分模型过去不返回调用 ID，因此这个问题不一定出现；
OpenRouter 使用 OpenAI 协议，按 ID 关联工具结果也是正确行为。一些 Gemini
兼容端点会返回非空调用 ID，从而暴露这个兼容性问题。

## 修复方式

插件加载时只对 AstrBot 的 `ProviderGoogleGenAI._prepare_conversation()` 安装
可逆补丁。在每次 Gemini 请求构造前，插件根据 assistant 工具调用建立：

```text
tool_call_id -> function.name
```

然后仅在发送给 Gemini 的临时消息副本中补全工具结果的 `name`。插件：

- 不修改会话数据库；
- 不改变 OpenRouter、Grok 等 OpenAI 兼容 provider；
- 支持并行工具调用；
- 卸载或停用时恢复 AstrBot 原方法；
- 能修复仍保留完整 assistant/tool 配对的旧历史。

## 安装

在 AstrBot WebUI 的插件管理中，通过仓库地址安装：

```text
https://github.com/LYB926/astrbot_plugin_fix_gemini_toolcall
```

安装后重启或重载插件，并对曾经报错的会话执行一次：

```text
/new
```

若日志出现以下信息，说明补丁已加载：

```text
[fix_gemini_toolcall] Gemini 工具调用历史兼容补丁已安装。
```

## 限制

如果上下文压缩已经删除了 assistant 工具调用，只留下孤立的 tool 消息，插件
无法可靠推断函数名，会记录警告并建议 `/new`。插件使用 AstrBot 的内部 provider
方法实现热修复；AstrBot 上游修复该问题后，应停用本插件并重新验证。

## 开发与测试

目标运行环境：AstrBot 4.26.x、Python 3.12。

```bash
pytest -q
ruff check .
ruff format --check .
```

可选真实端点测试不会把 API Key 写入文件：

```bash
GEMINI_API_KEY='...' \
GEMINI_API_BASE='https://your-endpoint.example' \
GEMINI_MODEL='gemini-3.5-flash' \
python scripts/live_test.py
```

## License

[MIT](LICENSE)
