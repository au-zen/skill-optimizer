# Hermes Configuration Resolution

## opencode-zen Provider Rewrite

When `_resolve_hermes_default_model()` detects `provider=opencode-zen` or `opencode` in `api_base`, it automatically rewrites to OpenRouter direct connection:

- **model**: `deepseek-v4-flash-free` -> `deepseek/deepseek-v4-flash:free` (OpenRouter免费模型，$0)
- **api_base**: `https://openrouter.ai/api/v1`
- **api_key**: resolved via `_resolve_api_key('')` - 空 hint，避免被 deepseek 过滤逻辑限死到过期 DEEPSEEK_API_KEY，选择 OPENROUTER_API_KEY 或 SKILLOPT_API_KEY
- **provider**: `opencode-zen` -> `openrouter`

### Why OpenRouter free model?

1. opencode-zen 端点是 session gateway，不接受外部 API key
2. `DEEPSEEK_API_KEY` 可能过期/额度用完
3. 用户偏好：**仅使用免费模型**，不上付费模型
4. 免费额度 50次/天/账号，超额返回 429
5. OpenRouter 上唯一免费 DeepSeek 模型是 `deepseek/deepseek-v4-flash:free`；`deepseek/deepseek-chat:free` 不存在(404)
6. Key 可写入 `.hermes/.env` 作为 `SKILLOPT_API_KEY`，shell env var 优先级更高

### Why not DeepSeek direct?

The initial fix went to DeepSeek direct (`api.deepseek.com/v1` + `deepseek-chat`), but:
1. `DEEPSEEK_API_KEY` may be expired or quota-exhausted
2. Calling `_resolve_api_key('deepseek-chat')` with a 'deepseek' hint filters out OPENROUTER_API_KEY
3. User preference is OpenRouter free model, not DeepSeek direct

### Key Resolution with opencode-zen fallback

```
_hermes_config reads: provider=opencode-zen, model=deepseek-v4-flash-free
    ↓
opencode-zen detected -> rewrite to OpenRouter FREE model
    |
    model=deepseek/deepseek-v4-flash:free, api_base=https://openrouter.ai/api/v1
    |
    _resolve_api_key('') -> no deepseek hint -> picks OPENROUTER_API_KEY (not DEEPSEEK_API_KEY)
    ↓
from_hermes_config receives: provider=openrouter, api_key=sk-or-v1-...
```

### OpenRouter Free Model Availability

Only `deepseek/deepseek-v4-flash:free` is free among DeepSeek models. `deepseek/deepseek-chat:free` does NOT exist (404). Other strong free models:
- `nousresearch/hermes-3-llama-3.1-405b:free`
- `nvidia/nemotron-3-super-120b-a12b:free`
- `google/gemma-4-31b-it:free`

Check available free models:
```
curl -H "Authorization: Bearer $OPENR...EY" https://openrouter.ai/api/v1/models | jq '.data[] | select(.pricing.prompt == "0") | .id'
```

### OpenRouter Model Naming

OpenRouter requires fully-qualified model IDs. Short names like `deepseek-chat` return 400: "Model ID 'deepseek-chat' is ambiguous".

### Key Persistence

When no shell env var is set for SKILLOPT_API_KEY or OPENROUTER_API_KEY, `_resolve_api_key()` falls back to reading `~/.hermes/.env` directly (raw file open, not os.environ). This bypasses the `***` masking in terminal output. Shell env vars take priority over .env file values.
