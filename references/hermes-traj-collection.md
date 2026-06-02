# 基于 Hermes Agent 真实工具调用的轨迹收集

> **理由：** skill-optimizer 是 Hermes Agent 的 skill，其训练轨迹应真实反映 Hermes Agent skill_* 工具（`skill_view`, `skills_list`, `skill_manage`）的调用方式，而不是用 Python 脚本模拟虚假工具调用。

## 核心思路

不要写 Python 脚本来"模拟" Hermes 工具调用并构造伪造的轨迹数据。直接用 Hermes Agent 的 `skill_view`、`skills_list`、`skill_manage`、`read_file` 等**真实工具**执行训练任务，然后将工具的真实输出打包为 Trajectory JSON。

真实轨迹 vs 模拟轨迹的差异：

| 维度 | 模拟脚本 | 真实工具 |
|------|----------|----------|
| tool_calls | 手工构造的假数据 | 真实工具名称、参数、结果 |
| output | 编造的摘要 | 实际文件内容、字段值 |
| score | 随意赋值 | 基于真实结果客观评分 |
| messages | 虚构对话 | 实际交互过程 |
| 可信度 | 低（优化器学到的是"你假想的行为"） | 高（优化器看到的是"实际发生的事"） |

## 标准流程

### Step 1: 设计任务集

针对你要优化的目标 skill，设计 10 个左右的独立任务。每个任务应当：
- 调用 1-3 个不同的 Hermes Agent 工具（`skill_view`, `skills_list`, `read_file`, `patch`, `search_files` 等）
- 有明确的成功标准（可客观评分 0-1）
- 覆盖目标 skill 的不同使用场景

**典型任务模板（用于 skill-optimizer 自优化）：**

| # | 任务 | 工具 | 评分标准 |
|---|------|------|----------|
| 1 | skill_view 目标 skill 自身 | skill_view | 返回了完整 frontmatter + body = 1.0 |
| 2 | skills_list 全库普查 | skills_list | 返回了 skills 列表 + 类别 = 0.9 |
| 3 | skill_view 关联标准（authoring 标准） | skill_view | 提取了完整规则集 = 0.95 |
| 4 | skill_view 替代标准（write-a-skill） | skill_view | 对比差异 = 0.9 |
| 5 | skill_view 环境技能（hermes-agent） | skill_view | 获取 CLI 约定/cron 架构 = 0.85 |
| 6 | read_file 历史优化记录 | read_file | 读取了自优化日志并理解 = 0.95 |
| 7 | read_file 审计报告 | read_file | 提取了已修复/未修复问题 = 0.85 |
| 8 | read_file 协议检查 | read_file | 确认了 Trajectory 格式 = 0.95 |
| 9 | read_file 入口检查 | read_file | 确认了 CLI 入口点 = 1.0 |
| 10 | 跨 skill 交叉验证 | skill_view(多个) + skills_list | 综合合规报告 = 0.95 |

### Step 2: 执行真实工具调用

对每个任务，实际调用 Hermes Agent 工具。例如：

```
skill_view(name='skill-optimizer')        # → 返回完整的 SKILL.md
skills_list()                              # → 返回 164 个 skills, 22 categories
read_file(path='.../protocols.py')         # → 返回文件内容 + 行号
```

将所有工具的真实输出记录下来（输出内容、工具名称、参数）。

### Step 3: 打包为 Trajectory JSON

按 `protocols.py` 的 `Trajectory.to_dict()` 格式，将每个任务转换为 JSON 对象：

```python
{
  "task_id": "hermes-skill-view-skill-optimizer",
  "task_input": "Load the skill-optimizer skill...",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "skill_view returned ..."}
  ],
  "tool_calls": [
    {"tool": "skill_view", "params": {"name": "skill-optimizer"}, "result_present": True}
  ],
  "output": "Frontmatter ✓ name ✓ description ✓ ...",
  "score": 1.0,
  "error": None,
  "metadata": {"source": "hermes-agent-tool", "tool_type": "skill_view"}
}
```

**关键要求：**
- `messages` 中的 `tool` role 内容必须来自**真实工具输出**，不能编造
- `tool_calls` 中的 `tool` 和 `params` 必须与**实际使用的工具**一致
- `score` 基于**客观结果**（成功=1.0，部分成功=0.8-0.95，失败=0.0-0.3）
- 如果任务涉及多个工具调用，`tool_calls` 数组中应包含所有工具

### Step 4: 拆分为 train/sel

```python
import json, random
random.seed(42)

with open('all_trajs.json') as f:
    trajs = json.load(f)

# 选择 2 条最全面的任务作为验证集
sel_ids = {"hermes-skill-view-skill-optimizer", "hermes-cross-skill-feature-audit"}
train = [t for t in trajs if t["task_id"] not in sel_ids]
sel = [t for t in trajs if t["task_id"] in sel_ids]

with open('train_trajs.json', 'w') as f: json.dump(train, f, indent=2)
with open('sel_trajs.json', 'w') as f: json.dump(sel, f, indent=2)
```

### Step 5: 运行优化

```bash
cd ~/.hermes/skills/software-development/skill-optimizer
rm -rf selfopt_v4

uv run python3 run.py optimize \
  --skill ./SKILL.md \
  --train-trajs /tmp/train_trajs.json \
  --sel-trajs /tmp/sel_trajs.json \
  --epochs 2 --steps 2 \
  --budget-init 3 --budget-final 1 \
  --work-dir ./selfopt_v4 \
  --optimizer-model "deepseek-chat" \
  --optimizer-api-base "https://api.deepseek.com/v1"
```

> **注意：** `--hermes-default` 在某些 provider（如 opencode-zen）下可能因认证问题导致 reflection 步骤 401。如果遇到此问题，请改用显式的 `--optimizer-model` + `--optimizer-api-base`。

### Step 6: 审核并应用

```bash
diff SKILL.md best_skill.md          # 查看改动
cat selfopt_v4/candidates/*.md       # 查看所有候选
```

关键：不要盲目 `cp best_skill.md SKILL.md`。检查候选编辑是否引入了 Markdown 结构损坏（表格、代码块、YAML 缩进）。只提取合理的规则人工修正后插入。

## 为什么真实工具轨迹更好

1. **真实性：** 优化器看到的是与实际运行完全一致的工具调用格式和返回值
2. **客观评分：** 分数基于工具实际返回的内容，而非人为编造
3. **无幻觉得分：** 模拟轨迹可能编造成功的工具调用结果；真实轨迹不会
4. **可审计：** 每条轨迹的 tool_calls 都对应一次实际发生的操作
5. **环境感知：** 工具返回中包含实际环境信息（skill 数量、版本号、文件大小），而不是泛泛的值

## 注意事项

- 不要在一步中收集所有轨迹再统一打包。每执行一个任务，立即保存其结果——避免丢失
- 对于 `skill_view` 返回的长内容，`messages` 中可以存放摘要（前 500 字符），完整的返回内容写进 `output` 字段
- 轨迹的 `score` 应当区分"完全成功"（1.0）和"部分成功"（0.8-0.95）。只有异常或错误才给 0.0-0.3
- 如果工具调用失败（404、401 等），记录 `error` 字段并给低分
- 不要在轨迹中写入本次会话的敏感信息（API key、token 等）
