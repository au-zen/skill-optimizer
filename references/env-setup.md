# SkillOpt .env 配置 — 3 层 Fallback 链

## 文件位置

```
~/.hermes/skills/software-development/skill-optimizer/.env
```

## 自动加载机制

`scripts/llm_client.py` 在模块导入时自动从技能根目录读取 `.env`：

```python
# 手动解析，无需 python-dotenv 依赖
_skill_root = Path(__file__).resolve().parent.parent  # scripts/../ = 技能根目录
_env_path = _skill_root / ".env"
# 读取 → 解析 KEY=VALUE → os.environ[key] = value  (override 模式)
```

- 使用 **override 模式**（`os.environ[key] = value`），非 `setdefault`
- 确保父进程 shell env var **不会干扰** optskill 的 LLM 配置
- 支持 `#` 注释、引号剥离、空行跳过
- 不从 `~/.hermes/.env` 或 `~/.hermes/credentials/` 读取任何 key

## 推荐配置（3 层 Fallback）

```bash
# ══════════ Tier 1: opencode-zen (最高优先级) ══════════
SKILLOPT_API_KEY=sk-zN6...Pvfc
SKILLOPT_MODEL=deepseek-v4-flash-free
SKILLOPT_API_BASE=https://opencode.ai/zen/v1

# ══════════ Tier 2: DeepSeek 直连 (可选 fallback) ══════════
SKILLOPT_API_KEY_2=sk-...
SKILLOPT_MODEL_2=deepseek-v4-flash
SKILLOPT_API_BASE_2=https://api.deepseek.com/v1

# ══════════ Tier 3: OpenRouter 免费模型 (可选 fallback) ══════════
SKILLOPT_API_KEY_3=sk-or-...
SKILLOPT_MODEL_3=google/gemma-4-31b-it:free
SKILLOPT_API_BASE_3=https://openrouter.ai/api/v1
```

**至少需要 Tier 1**，Tier 2/3 可选。只配 Tier 1 则无 fallback 保护。Tier 2/3 各自有回退检测：Tier 2 无 `SKILLOPT_API_KEY_2` 时尝试 `DEEPSEEK_API_KEY` 环境变量；Tier 3 无 `SKILLOPT_API_KEY_3` 时尝试 `OPENROUTER_API_KEY` / `OPENROUTER_API_KEY_2`。

## Fallback 行为

`OptimizerLLMClient.from_hermes_config()` 读取全部 3 层，`chat()` 方法按序尝试：

| 错误类型 | 行为 |
|----------|------|
| Timeout, Connection error, DNS failure | Fallthrough → 下一层 |
| 429 (Rate limit) | Fallthrough → 下一层 |
| 5xx (Server error) | Fallthrough → 下一层 |
| 400 (Bad request), 401/403 (Auth), 404 (Not found) | 直接报错，不 fallthrough |

## 注意

- **opencode 的 deepseek-v4-flash-free 是推理模型**：max_tokens 不足时 `content` 可能为空（所有 token 用于推理输出），建议至少给 200 max_tokens
- **OpenRouter 上的 deepseek 免费模型常 429 限流**：Tier 3 推荐用 gemma 等非 deepseek 免费模型
- **DeepSeek 直连 key**（`sk-...` 格式）与 opencode key（`sk-z...` 格式）是不同系统的 key，不可混用

## 优先级

Override 模式：**.env 文件覆盖所有**，shell env var 不生效。
这是刻意设计——确保 optskill 的 LLM 配置完全独立于进程环境。

## 与 Hermes 关系

| 组件 | 是否读取 | 说明 |
|------|----------|------|
| `~/.hermes/config.yaml` | ❌ 不读 | 已删除 `_read_hermes_config()` |
| `~/.hermes/.env` | ❌ 不读 | optskill 独立管理 |
| `~/.hermes/credentials/` | ❌ 不读 | optskill 独立管理 |
| 技能目录 `.env` | ✅ 自动加载（override 模式） | 唯一 key 来源 |
| Shell env var | ❌ 被覆盖 | .env 赋值覆盖父进程变量 |

## 调试

```bash
# 验证 .env 是否加载，查看 3 层链
cd ~/.hermes/skills/software-development/skill-optimizer
python3 -c "
import os
# 清除所有 API_KEY 环境变量以测试纯 .env 加载
for k in list(os.environ):
    if 'API_KEY' in k or 'SKILLOPT' in k or 'DEEPSEEK' in k:
        del os.environ[k]
from scripts.llm_client import _resolve_all_providers
providers = _resolve_all_providers()
print(f'Providers ({len(providers)} tiers):')
for p in providers:
    print(f'  {p[\"label\"]:20s}  model={p[\"model\"]}  key_len={len(p[\"api_key\"])}')
# 测试 API 直连
from scripts.llm_client import OptimizerLLMClient
client = OptimizerLLMClient.from_hermes_config()
result = client.chat([{'role': 'user', 'content': 'Reply: OK'}], max_tokens=200)
print(f'API result: {result}')
"
```
