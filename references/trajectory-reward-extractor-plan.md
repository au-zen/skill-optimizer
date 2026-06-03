# Trajectory Reward Extractor 接入方案

## 1. 背景与修正后的判断

基于 review 对 Hermes 官方文档的重新解读，本项目的关键方向应从“优化 `SKILL.md` 文本质量”修正为“基于真实 Hermes 轨迹优化工具使用行为”。Hermes Trajectory Format 已经在批量运行器轨迹中提供 `completed`、`api_calls`、`toolsets_used`、`tool_stats`、`tool_error_counts` 等字段；Hermes Tools Runtime 也说明工具不是简单的 `LLM -> Tool` 调用，而是经由 registry、toolset resolution、dispatch、environment 的统一运行时。

因此，Skill Optimizer 不应再把 `score_skill.py` 的结构化文档评分作为主导 reward。`score_skill.py` 可以继续作为候选 `SKILL.md` 的静态安全检查或辅助质量门，但 candidate selection 的主 reward 应来自 Hermes Runtime 写入的 trajectory 信号。

## 2. 当前项目差距

当前仓库已经具备自优化闭环的雏形：

```text
rollout
  -> reflection
  -> candidate edit
  -> validation gate
  -> accepted/rejected buffer
```

但核心 scoring 仍偏向人工/静态路线：

- `scripts/protocols.py` 的 `Trajectory` 已有 `score` 字段，但没有表达 Hermes 原生字段，例如 `completed`、`api_calls`、`toolsets_used`、`tool_stats`、`tool_error_counts`。
- `scripts/rollout.py` 的 `hermes_auto` 分支仍通过 `score_skill.py` 评价候选 skill 文档结构，而不是读取 Hermes trajectory reward。
- `references/hermes-traj-collection.md` 强调真实工具轨迹，但其示例仍要求手工写入 `score`，还没有定义从官方 Hermes trajectory 字段抽取 reward 的标准层。

由此可见，真正缺口不是 `delegate_task` 是否能返回 `success_rate`，而是缺少一个 `Trajectory Reward Extractor`：把 Hermes 官方轨迹中的 runtime 信号稳定转换为 `Trajectory.score`，并让 engine 的 baseline、reflection、validation gate 都使用这个 score。

## 3. 目标架构

建议将优化链路调整为：

```text
Hermes Runtime
  -> trajectory_samples.jsonl / failed_trajectories.jsonl
  -> reward_extractor.py
  -> SkillOpt Trajectory(score + reward_components)
  -> reflection
  -> candidate SKILL.md
  -> candidate rollout
  -> validation gate
```

其中：

- Hermes Runtime 负责 state logging。
- `reward_extractor.py` 负责 reward shaping。
- `score_skill.py` 退居为静态文档质量辅助检查。
- `engine.py` 继续负责优化循环，但其 selection score 应优先来自 trajectory reward。

## 4. Reward Extractor 设计

### 4.1 输入

支持两类输入：

1. Hermes 官方 JSONL：
   - `trajectory_samples.jsonl`
   - `failed_trajectories.jsonl`
   - batch runner 自定义输出 JSONL
2. SkillOpt 现有 JSON 数组：
   - `train_trajs.json`
   - `sel_trajs.json`

### 4.2 输出

输出仍保持 SkillOpt `Trajectory` 兼容格式，但补充 reward 细节：

```json
{
  "task_id": "...",
  "task_input": "...",
  "messages": [],
  "tool_calls": [],
  "output": "...",
  "score": 0.82,
  "error": null,
  "metadata": {
    "reward_source": "hermes_trajectory",
    "reward_components": {
      "completed": 1.0,
      "tool_success_rate": 0.9,
      "tool_efficiency": 0.7,
      "error_penalty": 1.0,
      "answer_judge": 0.5
    },
    "hermes": {
      "api_calls": 7,
      "toolsets_used": ["code_tools", "file_tools"],
      "tool_stats": {},
      "tool_error_counts": {}
    }
  }
}
```

### 4.3 默认 reward 公式

第一版建议使用可解释、可配置的线性 reward：

```python
reward = (
    0.50 * completed
  + 0.20 * tool_success_rate
  + 0.10 * tool_efficiency
  + 0.10 * error_penalty
  + 0.10 * answer_judge
)
```

组件定义：

| 组件 | 默认权重 | 来源 | 计算方式 |
|------|----------|------|----------|
| `completed` | 0.50 | `completed` | `1.0` if true else `0.0` |
| `tool_success_rate` | 0.20 | `tool_stats` | `sum(success) / max(sum(count), 1)` |
| `tool_efficiency` | 0.10 | `api_calls`, `tool_stats` | 对过多 API/tool 调用做温和衰减 |
| `error_penalty` | 0.10 | `tool_error_counts`, `tool_stats.failure` | `1 - normalized_error_rate` |
| `answer_judge` | 0.10 | 可选 LLM/规则 judge | 未配置时给中性值 `0.5` |

### 4.4 Tool Policy 诊断

Extractor 不只输出总分，还应输出可供 reflection 使用的诊断：

```json
{
  "tool_policy_diagnostics": {
    "high_failure_tools": ["terminal"],
    "overused_tools": ["terminal"],
    "unused_expected_toolsets": ["file_tools"],
    "api_call_budget_exceeded": true,
    "suggestion": "Skill should constrain terminal usage and prefer read_file for static inspection tasks."
  }
}
```

这样 reflection 的问题定位会从“文档格式不合格”转向“工具选择策略不稳定”。

## 5. 代码改造计划

### Phase 1: 新增 reward_extractor.py

新增 `scripts/reward_extractor.py`：

- `extract_reward(entry: dict, config: RewardConfig) -> RewardResult`
- `hermes_entry_to_trajectory(entry: dict, config: RewardConfig) -> Trajectory`
- `load_hermes_jsonl(path: str) -> list[Trajectory]`
- `convert_hermes_jsonl(input_paths, output_path)` CLI 辅助函数

最低实现要求：

- 兼容 Hermes batch runner 字段缺失场景。
- 对 CLI/交互式格式只有 `completed` 而没有 `tool_stats` 的样本，给出合理默认值。
- 所有 reward component 写入 `metadata.reward_components`。
- 原始 Hermes 字段保留到 `metadata.hermes`，便于审计。

### Phase 2: 扩展 protocols.py

保持 `Trajectory.score` 不变，避免破坏已有 engine；但建议在 `metadata` 中规范化以下字段：

```text
metadata.reward_source
metadata.reward_components
metadata.tool_policy_diagnostics
metadata.hermes
```

如需更强类型，可增加可选 dataclass：

```python
@dataclass
class RewardResult:
    score: float
    components: dict[str, float]
    diagnostics: dict[str, Any]
```

### Phase 3: CLI 接入

在 `scripts/optimizer_cli.py` 增加命令：

```bash
uv run python3 run.py extract-rewards \
  --input trajectory_samples.jsonl failed_trajectories.jsonl \
  --output train_trajs.json \
  --weights completed=0.5,tool_success_rate=0.2,tool_efficiency=0.1,error_penalty=0.1,answer_judge=0.1
```

并在 `optimize` 中增加可选参数：

```bash
--train-hermes-jsonl trajectory_samples.jsonl
--sel-hermes-jsonl failed_trajectories.jsonl
```

加载流程：

```text
--train-hermes-jsonl
  -> reward_extractor.load_hermes_jsonl
  -> train_split = [task_id]
  -> _install_cached_rollout(engine, extracted_trajs)
```

### Phase 4: hermes_auto 改造

`RolloutHarness._run_hermes_subagent()` 当前应标记为 legacy/static mode。建议拆分为：

```text
hermes_static_score  -> score_skill.py
hermes_trajectory    -> 读取 Hermes Runtime 真实轨迹并抽 reward
```

短期可以先不真正执行 Hermes delegate/cron，只支持读取已生成 JSONL；中期再接入真实 Hermes run command 或 batch runner。

### Phase 5: reflection prompt 改造

更新 `templates/analyst_error.md` 和 `templates/analyst_success.md`，要求 analyst 优先读取：

- `metadata.reward_components`
- `metadata.tool_policy_diagnostics`
- `metadata.hermes.tool_stats`
- `metadata.hermes.tool_error_counts`

并把编辑建议限定为“改变工具使用行为的规则”，例如：

- 何时优先 `read_file` 而非 `terminal`。
- 何时禁止重复调用高失败率工具。
- 何时先 `skills_list` 再 `skill_view`。
- 如何处理 tool unavailable / tool error。

## 6. Candidate Selection 策略

Validation gate 应从单一平均分升级为多维门控：

```text
accept iff:
  candidate_mean_reward > baseline_mean_reward + min_delta
  and candidate_completed_rate >= baseline_completed_rate
  and candidate_tool_failure_rate <= baseline_tool_failure_rate + tolerance
  and drift_check_passes
```

这样可以避免一种常见 reward hacking：candidate 靠减少工具调用获得 efficiency 分，但任务完成率下降。

建议保存以下 metrics：

```json
{
  "baseline_reward": 0.72,
  "candidate_reward": 0.81,
  "completed_rate_delta": 0.05,
  "tool_success_rate_delta": 0.12,
  "api_calls_delta": -2.3,
  "high_failure_tools_fixed": ["terminal"]
}
```

## 7. 与 score_skill.py 的关系

不建议删除 `score_skill.py`，但要调整主导地位：

| 组件 | 新角色 |
|------|--------|
| `reward_extractor.py` | 主 reward，负责 candidate selection |
| `score_skill.py` | 静态 guardrail，检查 Markdown/YAML/结构问题 |
| LLM Judge | 可选弱信号，只占小权重或只用于 answer quality |
| Hermes trajectory | 权威 runtime evidence |

如果 `score_skill.py` 发现候选 skill 结构损坏，应直接拒绝；但结构评分高不应自动意味着 candidate 更优。

## 8. 里程碑

### M1: 可离线验证的 Reward Extractor

- 新增 `reward_extractor.py`。
- 能把 Hermes JSONL 转成 SkillOpt JSON trajectories。
- 单元测试覆盖 completed、tool_stats、tool_error_counts 缺失/异常情况。

### M2: CLI 与 optimize 接入

- 增加 `extract-rewards` 命令。
- `optimize` 支持 `--train-hermes-jsonl` / `--sel-hermes-jsonl`。
- `manual` cached rollout 使用 extractor 产出的 score。

### M3: Reflection 使用 tool policy diagnostics

- analyst templates 引入 reward components。
- failed trajectory 中高失败工具会进入反思摘要。
- 成功轨迹中的高效 toolset 组合会进入 preservation edits。

### M4: Validation Gate 多维化

- `_validate_and_gate` 不只比较平均 `score`。
- metrics 持久化 reward components delta。
- 防止 efficiency-only reward hacking。

### M5: 真实 Hermes Runtime 闭环

- 接入 Hermes batch runner 或可重复的 Hermes agent run 命令。
- 每个 candidate rollout 生成新的官方 trajectory JSONL。
- Extractor 从新轨迹抽 reward 后参与 selection。

## 9. 最小可行实现顺序

建议先做最小闭环，而不是一次性重写 engine：

1. 新增 extractor，把官方 JSONL 转成现有 `Trajectory`。
2. 让 `--train-trajs` / `--sel-trajs` 可直接消费 extractor 输出。
3. 保持 engine 不变，先验证 reward 分数能驱动 candidate selection。
4. 再把 diagnostics 注入 reflection prompt。
5. 最后扩展 validation gate 和 hermes_auto runtime。

这样既尊重当前代码结构，也能迅速验证 review 中的核心假设：Skill Optimizer 是否已经从 Prompt Optimizer 过渡到基于 Hermes 轨迹的 SkillRL。

## 10. 风险与防护

| 风险 | 防护 |
|------|------|
| reward hacking：减少工具调用但不完成任务 | completed 权重大于 efficiency，validation gate 检查 completed rate |
| 工具失败率受环境影响 | metadata 记录 environment/tool availability，按任务类型聚合比较 |
| CLI 格式轨迹缺少 tool_stats | extractor 使用默认值并标记 `reward_confidence=low` |
| LLM Judge 不稳定 | 默认只给 0.10 权重，且允许关闭 |
| score_skill 被误当主 reward | 文档和 CLI 命名改为 static guardrail |

## 11. 结论

这个项目下一步不应继续围绕 `delegate_task -> success_rate` 发明新接口，而应把 Hermes 官方 trajectory schema 已经提供的 runtime 信号抽取出来。只要 `Trajectory Reward Extractor` 接入训练/验证 split，并让 validation gate 使用这些 reward，Skill Optimizer 就会从静态 Prompt/文档优化器升级为更接近 SkillRL 的系统：它优化的是 skill 对 Hermes 工具选择、工具失败规避、任务完成效率的行为约束，而不只是 Markdown 文本本身。
