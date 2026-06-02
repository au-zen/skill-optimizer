# Testing the SkillOpt Engine

## Project Structure

```
scripts/
├── __init__.py
├── protocols.py       # Data classes
├── llm_client.py      # LLM API client
├── rollout.py         # Rollout executor
├── engine.py          # Core engine
└── optimizer_cli.py   # CLI entry
```

All modules use relative imports (`from .protocols import ...`), so tests MUST be run from the project root.

## Test Execution

### From Project Root (REQUIRED)

```bash
cd ~/.hermes/skills/software-development/skill-optimizer

# Module imports
uv run python3 -c "from scripts.protocols import Trajectory; print('OK')"

# CLI help
uv run python3 -m scripts.optimizer_cli --help

# Or via run.py
uv run python3 run.py optimize --help
```

### Running External Test Scripts

For complex multi-step tests, write a script to /tmp/ and run from project root:

```bash
write_file(path="/tmp/test_foo.py", content="...")
terminal(command="cd /path/to/project && uv run python3 /tmp/test_foo.py")
```

### Common Test Patterns

**1. Verify imports:**
```python
from scripts.protocols import Trajectory, OptimizerConfig
from scripts.engine import SkillOptEngine
from scripts.llm_client import OptimizerLLMClient
```

**2. Test dataclass instantiation (check fields in protocols.py first):**
```python
# Trajectory fields: task_id, task_input, messages, tool_calls, output, score, error, metadata
t = Trajectory(task_id="t1", task_input="do X", score=0.8)

# OptimizerConfig fields: rollout_batch_size, reflection_minibatch_size, score_threshold,
#   edit_budget_init, edit_budget_final, schedule, max_epochs, max_steps_per_epoch, ...
config = OptimizerConfig(score_threshold=0.7, max_epochs=1, max_steps_per_epoch=1)
```

**3. Test static methods without engine instantiation:**
```python
from scripts.engine import SkillOptEngine
issues = SkillOptEngine._validate_skill_structure("---\nname: test\n---\n\n# Heading\ncontent")
assert len(issues) == 0
```

## Integration Test (Phase 3 Pattern)

Full end-to-end test of the optimizer with LLM reflection, merge, ranking, and validation gate.

### Setup: Stub Skill + Trajectory Data

Create a minimal stub skill with YAML frontmatter:
```markdown
---
name: test-skill
description: A stub for integration testing
version: 1.0.0
---

# Test Skill

## Rules
1. Always provide working code
2. Be concise
```

Create trajectory JSON files (train and sel splits). Each trajectory is a JSON object with:
- `task_id`: unique string
- `task_input`: the task description
- `messages`: list of `{role, content}` dicts
- `tool_calls`: list of tool call dicts
- `output`: the agent's final output text
- `score`: float in [0, 1] (≥0.5 = success, <0.5 = failure)
- `error`: null or error string
- `metadata`: dict with extra info

Mix success (score≥0.5) and failure (score<0.5) trajectories. 6-8 train + 2 sel is a good minimum.

### Run the Optimizer

```bash
cd ~/.hermes/skills/software-development/skill-optimizer

uv run python3 run.py optimize \
  --skill /tmp/test/stub_skill.md \
  --train-trajs /tmp/test/train_trajs.json \
  --sel-trajs /tmp/test/sel_trajs.json \
  --epochs 1 --steps 2 \
  --budget-init 3 --budget-final 1 \
  --work-dir /tmp/test/work \
  --hermes-default
```

### Verify Outputs

Checklist:
- [ ] Engine starts: "Loaded N train trajectories" + "M sel trajectories"
- [ ] Each step: Rollout > Proposed edits > Accepted/Rejected
- [ ] `work/best_skill.md` exists and has valid YAML frontmatter
- [ ] `work/state.json` exists: `current_score`, `best_score`, `epoch`, `step`, `accepted_edits`
- [ ] `work/metrics.jsonl`: one JSON line per step with `current_score`, `candidate_score`, `delta`
- [ ] `work/candidates/`: one candidate file per step
- [ ] Best score > 0 (at least some edits accepted)
- [ ] No edits rejected due to structural validation (unless expected)
- [ ] `best_skill.md` is under `work/` directory, NOT alongside original skill

### Test with Real Hermes Skill

Pick a small real skill (e.g. `mattpocock/skills/engineering/prototype`, 30 lines). Create trajectories that reflect real usage patterns of that skill. Run the same optimizer command. The engine should:
- Accept edits that improve the sel split score
- Reject edits that don't improve
- Never modify the original skill file (only reads it)
- Write all outputs (best_skill.md, candidates, state) under `--work-dir`

### Common Integration Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `[VALIDATION] Missing YAML frontmatter` | Stub skill lacks `---` delimiters | Add YAML frontmatter to stub skill |
| `best_skill.md` appears next to original skill | `best_skill_path` computed before `work_dir` | Move `work_dir` creation before state init in engine.py |
| `## ## Rules` in output | Section name from LLM already contains `##` | Use `_sanitize_section()` in `_edit_add` |
| `[ERR] No sel split configured` | Missing `--sel-trajs` | Add sel trajectory file with ≥2 trajectories |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ImportError: attempted relative import with no known parent package` | Running from wrong directory or via wrong command | `cd /project/root && uv run python3 ...` |
| `TypeError: unexpected keyword argument 'X'` | Guessed field names instead of reading source | `read_file(protocols.py)` first, check dataclass fields |
| LLM `chat()` returns empty string | OpenCode Zen model only returns reasoning tokens, no content | Use `--optimizer-model` to switch to a completion-capable model or set `SKILLOPT_API_KEY` |
| `Warning: No API key provided` | Hermes default model detected but no key needed | Normal for `--hermes-default` with opencode-zen; suppression handled in `llm_client.py` |
