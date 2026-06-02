---
name: skill-optimizer
description: Use when optimizing or auditing a SKILL.md document via SkillOpt
  methodology — systematic text-space optimization with bounded edits, validation
  gating, rejected-edit buffers, and epoch-wise slow updates. Load when asked to
  improve, iterate on, or auto-optimize any Hermes skill. Zero inference-time
  overhead at deployment. Based on arXiv:2605.23904 (Microsoft Research, May 2026).
version: 2.0.0
author: SkillOpt (Microsoft Research) — adapted for Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [skillopt, optimization, skill-evolution, training, text-space-optimization]
    related_skills: [writing-plans, subagent-driven-development, hermes-agent-skill-authoring]
---

# SkillOpt: Text-Space Skill Optimizer

## Overview

SkillOpt treats SKILL.md documents as trainable external state. An independent
optimizer model proposes bounded text edits to a target skill, gated by a held-out
validation split — analogous to gradient descent, but operating in natural language.

**Important scope note:** The bundled self-optimization workflow (via
`auto_collect_trajs.py`) evaluates *document quality and structural compliance*,
not downstream agent performance. It optimizes the skill-optimizer's own SKILL.md
against Hermes authoring conventions. To evaluate functional agent behavior, provide
custom rollout evaluators — see `references/hermes-traj-collection.md`.

Key results on arXiv:2605.23904: average +23.5 points across 6 benchmarks vs
baseline, outperforming manual authoring, TextGrad, GEPA, and EvoSkill.

Full methodology, architecture diagrams, prompt contracts, and pseudocode:
`skill_view("skill-optimizer", "references/methodology.md")`

## When to Use

- User asks to optimize, improve, or audit a specific SKILL.md
- User mentions "skillopt", "skill optimization", or "text-space optimization"
- A skill has been used repeatedly and shows consistent failure patterns
- Systematic skill improvement is needed with a measurable evaluation function

Don't use for:
- One-off manual edits (use `skill_manage(action='patch')` directly)
- Skills with no clear scoring function
- Fewer than 3–4 example tasks (insufficient for generalization)
- Real-time interactive optimization (SkillOpt is an offline process)

## Quickstart

### Manual (human-in-the-loop)
Load this skill, read `## Optimization Policy`, then apply the 8-stage reflection
process. Full methodology: `skill_view("skill-optimizer", "references/methodology.md")`

### Automated
```bash
cd ~/.hermes/skills/software-development/skill-optimizer

# Prepare train + sel trajectory files (see references/hermes-traj-collection.md)
uv run python3 run.py optimize \
  --skill /path/to/TARGET/SKILL.md \
  --train-trajs train_trajs.json \
  --sel-trajs sel_trajs.json \
  --epochs 2 --steps 2 \
  --hermes-default
```

Reference files:
- Environment & API keys: `skill_view("skill-optimizer", "references/env-setup.md")`
- Trajectory collection (static): `skill_view("skill-optimizer", "references/hermes-traj-collection.md")`
- Trajectory collection (session history): `scripts/collect_from_sessions.py --skill <name>`
- Merge trajectory pools: `scripts/merge_trajs.py --skill <name>`
- Autonomous scheduling: `skill_view("skill-optimizer", "references/scheduler-add-skill.md")`
- Engine invariants: `skill_view("skill-optimizer", "references/runtime-contract.md")`
- Engine code audit (2026-06-02): `skill_view("skill-optimizer", "references/audit-2026-06-02.md")`

## Optimization Policy

This is the **only section the automated optimizer is permitted to edit**
(the `## Optimization Policy` convention — a prompt-level constraint, not a code-level block).

- **Validation split is mandatory.** No `--sel-trajs` → engine refuses. No exceptions.
- **Edits must be bounded.** Maximum `edit_budget` edits per step; full rewrites forbidden.
- **Strict gating.** Candidate must strictly improve on D_sel (`candidate_score > current_score`) to be accepted.
- **Preserve failure records.** Rejected edits enter `rejected_buffer` as negative feedback for subsequent steps.
- **Stagnation protection.** Consecutive steps without improvement are tracked via `rejected_buffer` and inform the next reflection cycle via negative feedback. Full stagnation-aware early stopping is not implemented at the engine level.
- **Candidate-aware evaluation.** The scoring command must reference the candidate path via `{skill_path}` — a fixed command scoring the same file every step is a closed-loop failure.

## Common Pitfalls

1. **train/sel using the same file.** `--train-trajs` and `--sel-trajs` must be separate JSON files. `merge_trajs.py --sel-ratio 0.2` controls the validation split ratio when merging trajectory pools.

2. **manual mode validating with stale trajectories.** Trajectories in manual mode were collected under a specific skill version. Using them to gate a different candidate version makes the validation gate meaningless. Use `hermes_auto` rollout mode (under development) or a custom external evaluator for genuine functional validation.

3. **`{skill_path}` missing from the task template.** If `_run_script` runs a fixed command with no `{skill_path}` substitution, every candidate scores identically — the optimization loop is silent-broken. Verify by checking that `metrics.jsonl` shows score variance across steps.

4. **opencode-zen token exhaustion.** `deepseek-v4-flash-free` is a reasoning model. With `max_tokens < 200`, the content field may be empty (all tokens consumed by reasoning). Set `max_tokens ≥ 200`.

5. **`best_skill_path` landing in the wrong directory.** `best_skill_path` depends on `self.work_dir` being resolved and created **before** `OptimizationState` is instantiated. The current engine code already upholds this order (work_dir created at line 142-143, best_skill_path set at line 148). Any refactoring must preserve this: `Path(config.work_dir).resolve().mkdir(parents=True, exist_ok=True)` → then pass `str(work_dir / "best_skill.md")` as the path. Reversing the order writes the file to the original skill directory instead of `work_dir/`.

6. **Self-optimization scope confusion.** `auto_collect_trajs.py` generates structural-check trajectories only. It measures document compliance, not functional agent improvement. See Overview for the precise scope boundary.

7. **Rules-layer pollution from implementation details.** Only `## Optimization Policy` is editable. Writing engine internals (API fallback chains, path handling, buffer sizes) into this section exposes system constraints to the optimizer, risking silent breakage if they get "optimized away".

8. **`id()` reuse in `merge_trajs.py` deduplication.** `merge_trajs.py` uses `id(t)` as a synthetic task ID for anonymous trajectories (those without a `task_id` field). Python's `id()` is unique only for the lifetime of the object — if the dict holding the reference is GC'd and a new trajectory object gets the same id, dedup silently collapses them. Replace with `uuid.uuid4().hex[:8]` or a monotonic counter.

9. **Redundant function-level import shadowing module-level import.** A function scoped `import re` when `re` is already imported at the top of the module compiles fine, works, but wastes the function's first import hit, confuses readers, and accrete silently as code is added. During an audit, grep for `import` and `from ... import` lines indented at function level, then check whether the symbol is already imported at module scope. Remove the function-level import; the module-level import is always available.

10. **Dead imports at function scope.** Importing a symbol inside a function and never using it (e.g. `from .protocols import OptimizationState` inside `_detect_drift` where `OptimizationState` is never referenced) is dead code that lints pass silently because the import itself is a valid statement. During audit, check every function-scoped import for at least one use of the imported symbol in the function body.

11. **`__all__` leaking private symbols.** Underscore-prefixed names (e.g. `_compute_text_similarity`, `_find_section`) in `__all__` violate the Python convention that `_` means "internal, not part of the public API". While the symbols are still importable via explicit `from .module import _name`, putting them in `__all__` advertises them as public. Audit: check `__all__` lists for any entry starting with `_`.

12. **Silent `None` from `try/except ImportError`.** Wrapping `from .module import name` in a `try/except ImportError: name = None` block silently converts a missing dependency into `None`. If the downstream code later calls `name(...)`, the error is `TypeError: 'NoneType' object is not callable` — opaque and disconnected from the root cause (missing dependency). Only use this pattern when the module truly is optional AND every call site guards with `if name is not None:`. Otherwise, let the ImportError propagate naturally.

13. **Audit order heuristic.** When auditing implementation code (scripts/), read files in dependency order: protocols (types) → engine (core logic) → CLI entry (exits) → merge/collect utilities (leaf scripts). This catches bugs in the data model before chasing symptoms in leaf code. Read `__init__.py` last — it is a pure re-export layer whose issues (wrong names, try/except wrapping) only make sense after you know what the underlying modules export.

## Verification Checklist

- [ ] `best_skill.md` exists under `work_dir/`, not the original skill directory
- [ ] `state.json` shows `best_score > baseline_score` (baseline set by `run()` pre-loop evaluation, not 0.0)
- [ ] `metrics.jsonl` contains score variance — if every step shows identical `current_score` and `candidate_score`, the `{skill_path}` substitution is broken
- [ ] `accepted_edits` in `state.json` is non-empty (at least one valid optimization occurred)
- [ ] Edit history is auditable (each accept/reject entry has epoch, step, score)
- [ ] `score_skill.py --skill <path>` run twice on the same file returns scores within 0.01 (scorer stability)
- [ ] Slow-update field content is general, not instance-specific
