# SkillOpt v2 改造：Stage 1–4 完成报告

> 日期：2026-06-01
> 基线版本：v2.0.0（Stage 0 重构后）

---

## Stage 1 — 闭环修复（candidate-aware evaluation）

### 目标
让引擎能真正评估候选 skill 的质量，而不是使用固定评分。

### 改动

| 文件 | 改动 | 说明 |
|------|------|------|
| `scripts/score_skill.py` | **新增** | 独立的结构质量评分器。接受 `--skill`、`--verbose`、`--stability`、`--output` 参数 |
| `scripts/engine.py` | `run()` 开头加入 baseline 初始化 | 启动时用 D_sel 评估原始 skill，`current_score` 不再以 0.0 初始化。验证：`best_score > baseline_score` |
| `scripts/rollout.py` | `_run_script()` 加入 `{skill_path}` 替换 | 模板中的 `{skill_path}` 运行时替换为候选文件路径。移除 `shell=True`，改用 `shlex.split` |
| `scripts/rollout.py` | `_run_script()` 加入 `{skill_dir}` 替换 | 候选目录路径替换，方便相对路径引用 |

### score_skill.py 功能

```bash
# 结构评分
uv run python3 scripts/score_skill.py --skill SKILL.md
# 输出: overall_score, 9项per-check评分, issues列表

# 详细输出
uv run python3 scripts/score_skill.py --skill SKILL.md --verbose

# 稳定性自检（Stage 3 复用）
uv run python3 scripts/score_skill.py --skill SKILL.md --stability
# 输出: run1_score, run2_score, max_diff, stable (bool)
# Exit 0 = stable, 1 = unstable

# JSON 输出到文件
uv run python3 scripts/score_skill.py --skill SKILL.md --output result.json
```

### 验证结果
```
[SkillOpt] Baseline score (D_sel): 0.6000
current_score = 0.6000  (不是 0.0)
score_skill.py --stability: STABLE (diff=0.000000)
```

---

## Stage 2 — 稳定性（drift detection）

### 目标
防止 skill 在多次优化后无声地偏离原始结构。

### 改动

| 文件 | 改动 | 说明 |
|------|------|------|
| `scripts/protocols.py` | `OptimizationState` 新增 4 个字段 | `origin_skill_path`, `origin_skill_hash`, `drift_detected`, `drift_count` |
| `scripts/protocols.py` | `OptimizerConfig` 新增 `min_similarity` | 默认值 `0.6` |
| `scripts/engine.py` | `run()` 设置 origin 锚点 | `origin_skill_path` 和 `origin_skill_hash` 在 run() 启动时记录一次，永不更新 |
| `scripts/engine.py` | `_detect_drift()` 新增 | 候选与 origin 的 line-Jaccard 相似度低于 `min_similarity` 时拒绝 |
| `scripts/engine.py` | `_validate_and_gate()` 调用 drift check | 每个候选先过 drift 再过 score gate |
| `scripts/engine.py` | `_compute_text_similarity()` 新增 | 行级 Jaccard 相似度计算函数 |
| `scripts/optimizer_cli.py` | `--min-similarity` 参数 | 通过 CLI 可调 |

### 行为
```
candidate similarity < 0.6 → [DRIFT] rejected
origin_skill_path 不可变          → 所有候选始终与原点比较
drift_count 累计                  → run() 结束时报告
```

### 验证结果
```
Self-similarity=1.0000
Modified similarity=0.9888 (< 1.0)
Drift detection wired into validation gate
```

---

## Stage 3 — 健壮性（reward hacking 防护）

### 目标
防止评分器被单一指标操纵。

### 改动

| 文件 | 改动 | 说明 |
|------|------|------|
| `scripts/protocols.py` | `OptimizerConfig` 新增 `use_secondary_scorer` | 启用后调用二次评分器交叉验证 |
| `scripts/protocols.py` | `OptimizerConfig` 新增 `holdout_rollout` | 启用 holdout 集评估 |
| `scripts/protocols.py` | `OptimizationState` 新增 `holdout_score` | 记录 holdout 集分数（纯监控，不门控） |
| `scripts/engine.py` | `_validate_and_gate()` 二次评分 | secondary_score < primary * 0.5 时告警 |
| `scripts/engine.py` | `_validate_and_gate()` holdout 评估 | holdout 集在门控后评估，仅记录不阻断 |
| `scripts/score_skill.py` | `--stability` 稳定性自检 | 同一文件跑两次，diff < 0.01 判定为 stable |
| `scripts/optimizer_cli.py` | `--secondary-scorer` `--holdout-trajs` | CLI 开关 |

### 配置
```bash
# 启用二次评分器 + holdout 集
uv run python3 run.py optimize \
  --skill ./SKILL.md \
  --train-trajs train.json \
  --sel-trajs sel.json \
  --holdout-trajs holdout.json \
  --secondary-scorer
```

---

## Stage 4 — Hermes rollout（hermes_auto mode）

### 目标
提供无需外部 rollout 引擎的独立评估模式。

### 改动

| 文件 | 改动 | 说明 |
|------|------|------|
| `scripts/rollout.py` | `_run_hermes_subagent()` 实现 | 直接调用 `score_skill()` 进行功能评估 |
| `scripts/rollout.py` | `_run_single()` 新增分支 | `mode='hermes_auto'` 分发到子代理 |
| `scripts/rollout.py` | 导入 `import sys` | 支持子进程 fallback |

### 行为
`hermes_auto` 模式直接调用 `score_skill()` 对候选 skill 评分：
```
mode=hermes_auto → score_skill(candidate_path).overall_score
Fallback: subprocess → python3 score_skill.py --skill candidate.md
```

### 验证结果
```
hermes_auto score: 0.9444 (对当前 SKILL.md)
n_checks=9, n_issues=1, mode=hermes_auto
```

---

## 新增 CLI 参数总览

| 参数 | 默认值 | Stage | 说明 |
|------|--------|-------|------|
| `--min-similarity` | `0.6` | 2 | 候选与 origin 的最小文本相似度 |
| `--secondary-scorer` | `False` | 3 | 启用二次评分器交叉验证 |
| `--holdout-trajs` | `None` | 3 | Holdout 轨迹 JSON 文件路径 |

score_skill.py 独立 CLI：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skill` | (必需) | 目标 SKILL.md 路径 |
| `--verbose` | `False` | 逐项输出评分明细 |
| `--stability` | `False` | 跑两次并报告稳定性 |
| `--output` | `None` | JSON 结果写入文件 |

---

## 文件变更清单

```
新增:
  scripts/score_skill.py       (235 行)  结构质量评分器

修改:
  scripts/protocols.py         +22 行    6 个新 state/config 字段
  scripts/engine.py            +63 行    baseline init, drift, holdout, secondary scorer
  scripts/rollout.py           +62 行    {skill_path} 替换, hermes_auto mode
  scripts/optimizer_cli.py     +16 行    3 个新 CLI 参数
  scripts/__init__.py          +8 行     新导出

创建:
  references/stage-summary.md  (本文件)
```

---

## 与 roadmap 对照

```
Stage 0 (结构合规)    ✅ 已完成
Stage 1 (闭环修复)    ✅ score_skill.py + baseline + {skill_path}
Stage 2 (稳定性)      ✅ origin hash + drift detection
Stage 3 (健壮性)      ✅ secondary scorer + holdout + stability check
Stage 4 (Hermes rollout) ✅ hermes_auto mode + _run_hermes_subagent
```
