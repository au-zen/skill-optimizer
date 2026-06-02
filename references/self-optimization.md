# 自优化指南：让 skill-optimizer 优化自己

已执行三轮自优化：

| 轮次 | 日期 | 轨迹数 | Accepted | Rejected | 结果 |
|------|------|--------|----------|----------|------|
| v1 | 2026-05-30 | 10 | 3 | 3 | +2 规则 |
| v2 | 2026-05-31 | 11 | 4 | 5 | +4 规则，v1.2.0→v1.3.0 |
| v3 | 2026-05-31 | 10 | 3 | 3 | +1 规则，v1.3.0→v1.4.0。Hermes skill_* 工具集成轨迹 |
| v4 | 2026-05-31 | 10 | 3 | 3 | +1 规则（hermes-agent 上下文规则），v1.4.0→v1.5.0。真实 Hermes Agent skill_* 工具轨迹 |

**关键发现：** 自优化引擎可以成功对自己产生有效改进。两次运行均产出可用的 candidate edits，但某些编辑（Markdown 表格相关）需人工审核。

## 流程

### 1. 收集轨迹数据

**推荐方法：通过真实 Hermes Agent 工具采集**

详见 `references/hermes-traj-collection.md`。核心思路：直接用 Hermes Agent 的 `skill_view`、`skills_list`、`skill_manage`、`read_file` 等真实工具执行任务，将工具的真实输出包装为 Trajectory JSON。

**备选方法（Python 脚本模拟）：**

创建测试任务脚本，覆盖技能的核心功能：

- 模块导入测试
- 结构验证测试（有效文档 / 缺 frontmatter / 未闭合代码块 / 缺 heading）
- API 客户端创建和调用测试
- CLI help 测试
- 编辑操作测试（增/删/改，含组合测试）
- Trajectory 阈值测试
- 自优化流程测试（状态持久化、候选查看、指标查看）
- 空轨迹处理测试

→ 每条轨迹应有 `score` 字段（1.0=通过, 0.0=失败）
→ 最少 10 条（8 训练 + 2 验证），越多越有效

### 2. 拆分为 train + sel

```python
import json, random
random.seed(42)
with open('all_trajs.json') as f:
    trajs = json.load(f)
random.shuffle(trajs)
split = int(len(trajs) * 0.2)
val = trajs[:split]
train = trajs[split:]
with open('train_trajs.json', 'w') as f: json.dump(train, f)
with open('sel_trajs.json', 'w') as f: json.dump(val, f)
```

### 3. 运行自优化

```bash
cd ~/.hermes/skills/software-development/skill-optimizer
rm -rf selfopt_work

# 默认模式：使用预收集轨迹（manual scoring）
uv run python3 run.py optimize \
  --skill ./SKILL.md \
  --train-trajs /tmp/train_trajs.json \
  --sel-trajs /tmp/sel_trajs.json \
  --epochs 2 --steps 2 \
  --budget-init 3 --budget-final 1 \
  --work-dir ./selfopt_work

# hermes_auto 模式：用 score_skill.py 实时评分
# （需先在 optimizer_cli.py 添加 --rollout-mode 参数）
uv run python3 run.py optimize \
  --skill ./SKILL.md \
  --train-trajs /tmp/train_trajs.json \
  --sel-trajs /tmp/sel_trajs.json \
  --epochs 2 --steps 2 \
  --rollout-mode hermes_auto \
  --work-dir ./selfopt_work

# 查看结果
diff SKILL.md best_skill.md   # 检查改动
```

### 4. 审核 Candidate 编辑

**关键步骤：不要盲目 `cp best_skill.md SKILL.md`。**

LLM 生成的 candidate edits 可能损坏复杂 Markdown 结构：

| 风险 | 示例 | 后果 |
|------|------|------|
| 表格损坏 | 管道符 `|` 被转义为 `\|` | 表格渲染断裂 |
| `\n` 转义 | 用 `\\n` 替代真实的换行 | 多行内容挤成一行 |
| 代码块丢失缩进 | 嵌套 ` ``` ` 被破坏 | 高亮失效 |
| 多余空格 | 行尾多余空格 | diff 噪声 |

**审核流程：**

1. 运行 `diff SKILL.md best_skill.md` 确认改动
2. 如果改动涉及 Markdown 表格：**不要直接替换**，而是手动提取候选编辑中的规则文本，人工修正格式后插入 SKILL.md
3. 检查 candidate 输出的 JSON：`cat selfopt_work/candidates/candidate_e0s0.md` — 看 edits 的 `content` 字段是否存在 `\n` 或 `|` 异常
4. 如果 candidate 包含破坏性编辑，只提取合理的规则手工应用，跳过损坏的部分
5. 重新跑一遍轨迹收集确认所有测试仍然通过
6. version bump

### 5. 清理临时文件

每次自优化后清理：

```bash
rm -rf selfopt_work selfopt_v2 train_trajs.json sel_trajs.json _collect_trajs.py
```

## 已知 Pitfalls

### argparse 默认值不一致
`--optimizer-model` 和 `--optimizer-api-base` 的默认值在 argparse default、`cmd_optimize` 内 fallback 两处定义。改了一个漏另一个会导致模型走错。
**修复：** 依赖 argparse default 作为唯一真相源，`cmd_optimize` 中 `args.optimizer_model or "deepseek-chat"` 已无害化——但 `or` 只有在 argparse 设了 default 且用户未传参时才不生效。

### Prompt 模板花括号
`str.format(**kwargs)` 遇到 JSON 样例的 `{` `}` 时抛出 `KeyError`。
**修复：** `load_prompt()` 改用 `content.replace("{" + key + "}", str(value))`，只替换已知 key。

### argpar default值不一致
`--optimizer-model` 和 `--optimizer-api-base` 的默认值在 argparse default、`cmd_optimize` 内 fallback 两处定义。改了一个漏另一个会导致模型走错。  
**修复：** 依赖 argparse default 作为唯一真相源，`cmd_optimize` 中 `args.optimizer_model or \"deepseek-chat\"` 已无害化——但 `or` 只有在 argparse 设了 default 且用户未传参时才不生效。

### Prompt 模板花括号
`str.format(**kwargs)` 遇到 JSON 样例的 `{` `}` 时抛出 `KeyError`。  
**修复：** `load_prompt()` 改用 `content.replace(\"{\" + key + \"}\", str(value))`，只替换已知 key。

### 401 不触发 provider fallback
`llm_client.py` 中 401（Unauthorized）默认不在 `_FALLBACK_STATUSES` 中。Tier 1（opencode-zen）返回 401 时，LLM client 直接抛异常，不会尝试 Tier 2/3。  
**修复：** 将 `401` 加入 `_FALLBACK_STATUSES` 集合，或确保 Tier 1 的 SKILLOPT_API_KEY 有效。

### 缓存未覆盖 sel 轨迹
`_cached_rollout` 只缓存 train split，sel 轨迹触发 `FileNotFoundError`。  
**修复：** 用 `_cached_task_map = {t.task_id: t for t in cached_all}` 覆盖所有 split。

### LLM 生成的 edits 可能损坏 Markdown 结构
自优化 v2 中，LLM 生成的 candidate 将 Pitfalls 表格中的 `|` 转义为 `\` 并用 `\\n` 代替换行，导致表格完全损坏。优化器的结构验证只检查 frontmatter/代码块/section heading，不校验表格完整性。
**缓解：** 不要直接 `cp best_skill.md SKILL.md`。手动检查 candidate edits 中的 Markdown 表格、列表、代码块。提取合理规则人工修正格式后插入。如果表格损坏但规则合理，手动重写该规则并跳过 LLM 生成的表格结构。

### `--sel` 参数被解析为 task_id 列表
`--sel 0.2` 被 CLI 解析为 task_id 而非比例值。预收集轨迹时必须使用 `--train-trajs` + `--sel-trajs` 传入独立文件，无法用比例在线采样。
**修复：** 已固定为 `--train-trajs` / `--sel-trajs` 双文件模式。不要尝试 `--sel 0.2`。

## 结果检查清单

- [ ] `best_skill.md` 生成且结构完全（无表格损坏、无转义异常）
- [ ] accepted edits ≥ 1
- [ ] 新增规则合理且可操作
- [ ] 验证集分数未下降
- [ ] 候选编辑的人工审核完成（尤其是表格/代码块）
- [ ] version 已 bump
- [ ] 重新跑轨迹收集保证回归测试通过