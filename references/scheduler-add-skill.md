# 添加新 Skill 到自治调度器

## 前提

- skill-optimizer 的 Hybrid Scheduler 已部署（有 `opt_scheduler.py` + `scheduler_config.json` + cron job）
- 目标 skill 有一个可评分的任务（用作轨迹数据的基础）
- 目标 skill 已被某 cron job 引用（否则需要人工确认优化的意义）

## 步骤

### 1. 确认 skill 的 SKILL.md 路径

```bash
ls ~/.hermes/skills/<name>/SKILL.md            # 根目录 skill
ls ~/.hermes/skills/<category>/<name>/SKILL.md  # 子目录 skill
```

### 2. 准备轨迹数据

在 `scheduler_data/<name>/` 下创建：

```json
# scheduler_data/<name>/train.json  — 训练轨迹（8-10+ 条）
# scheduler_data/<name>/sel.json    — 验证轨迹（2+ 条）
```

轨迹格式见 `scripts/protocols.py` 的 `Trajectory` 数据类。

### 3. 注册到配置

**方式 A：手动编辑** `scripts/scheduler_config.json`

```json
{
  "skills": [
    {"name": "<name>", "skill_path": "/path/to/SKILL.md", "interval_hours": 48, "epochs": 1, "steps": 2}
  ]
}
```

**方式 B：自动扫描**（推荐，从 cron job 的 skill 引用发现路径）

```bash
cd ~/.hermes/skills/software-development/skill-optimizer
uv run python3 scripts/opt_scheduler.py --scan-cron skill-name-1 skill-name-2 --update-config
```

### 4. 验证

```bash
uv run python3 scripts/opt_scheduler.py --check-only
```

应输出新 skill 的状态（"未优化"/"需要优化"/"闲置中"）。

### 5. 试运行（可选）

```bash
uv run python3 scripts/opt_scheduler.py --force
```

观察调度器是否按配置执行优化并产出 Markdown 报告。

## 常见问题

| 问题 | 排查 |
|------|------|
| 调度器报告 NOT_FOUND | skill 名称拼写不对，或 SKILL.md 不在预期路径。用 `--scan-cron` 替代手动指定 |
| 调度器跳过（无轨迹） | `scheduler_data/<name>/` 下缺少 `train.json`。可以是空数组（`[]`），但不能不存在 |
| 优化完无报告 | 如果分数未提升（所有编辑被验证门控拒绝），调度器静默退出 |
| 频繁触发 | 在 `scheduler_config.json` 中调大 `interval_hours`（默认 48h） |
