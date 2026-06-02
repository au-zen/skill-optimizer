# SkillOpt Methodology

> Paper: *SkillOpt: Executive Strategy for Self-Evolving Agent Skills*
> arXiv:2605.23904, Microsoft Research, May 2026
> Authors: Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou et al.
> Code: https://aka.ms/skillopt

## Results Summary

| Benchmark | No Skill | SkillOpt | Gain |
|-----------|----------|----------|------|
| SearchQA | 77.7 | 87.3 | +9.6 |
| SpreadsheetBench | 41.8 | 80.7 | **+38.9** |
| OfficeQA | 33.1 | 72.1 | **+39.0** |
| DocVQA | 78.8 | 91.2 | +12.4 |
| LiveMathematicianBench | 37.6 | 66.9 | +29.3 |
| ALFWorld | 83.6 | 95.5 | +11.9 |
| **Average** | **58.8** | **82.3** | **+23.5** |

## Core Loop

```
FOR epoch = 1 TO max_epochs:
  FOR step = 1 TO max_steps_per_epoch:
    Lt = cosine_schedule(epoch, step, init=5, final=1)

    trajectories = rollout(target_model, current_skill, D_tr)
    failures, successes = split(trajectories, threshold)

    failure_edits = reflect(analyst_error, minibatch(failures, size=4))
    success_edits = reflect(analyst_success, minibatch(successes, size=4))

    merged = merge_final(
        merge_failure(failure_edits),
        merge_success(success_edits)
    )
    top_edits = rank_and_select(merged, budget=Lt, rejected_buffer)
    candidate = apply_edits(current_skill, top_edits)

    candidate_score = evaluate(target_model, candidate, D_sel)
    if candidate_score > current_score:
        current_skill = candidate
        current_score = candidate_score
        if candidate_score > best_score:
            best_score = candidate_score
            export(best_skill.md)
    else:
        rejected_buffer.append(top_edits, score_drop)

  slow_field = slow_update(accepted_edits, rejected_buffer)
  save_protected_field(current_skill, slow_field)
```

## Data Splits

- **D_tr (train)** — generates experience trajectories, used to propose edits
- **D_sel (selection)** — held-out validation set, gates whether edits are accepted
- **D_test (test)** — only for final reporting; invisible during optimization

## Edit Budget (Text Learning Rate)

The edit budget L_t is SkillOpt's analogy to a learning rate.

**Cosine schedule (default):**
```
L_t = L_final + (L_init - L_final) × (1 + cos(π × t/T)) / 2
```
- L_init = 5 (early steps, larger changes allowed)
- L_final = 1 (late steps, fine-grained adjustments only)

Unbounded rewrites erase effective rules, introduce incompatible instructions, and
overfit local failures. The budget enforces discipline.

## The 8 Prompt Contracts

These templates live in `templates/`. The engine loads them via `load_prompt()`.

| Contract | Role |
|----------|------|
| `analyst_error.md` | Identifies procedural errors in failure trajectories, proposes add/delete/replace edits |
| `analyst_success.md` | Identifies rules being followed in successes, proposes reinforcement edits |
| `merge_failure.md` | Deduplicates and generalizes failure-prevention edits |
| `merge_success.md` | Deduplicates success-preservation edits |
| `merge_final.md` | Combines both pools; failure corrections take priority |
| `ranking.md` | Selects top-Lt edits by expected impact, diversity, and compatibility |
| `slow_update.md` | Extracts long-horizon lessons at epoch boundary into protected field |
| `meta_skill.md` | Maintains optimizer-level memory across epochs |

## Rejected-Edit Buffer

Rejected updates remain useful. The buffer records:
- Observed failure patterns
- Attempted edits and their score deltas
- Failure location and context

Subsequent reflection calls receive this buffer as negative feedback, preventing
repetition of failed patterns. **Size limit:** last 20 rejected entries.

## Slow / Meta Update

Runs at each epoch boundary — analogous to a momentum term in gradient descent.

- **Protected field:** stores long-horizon lessons that step-level edits cannot overwrite
- **Meta update:** optimizer maintains its own "meta-skill" recording editing strategy experience

Prevents consecutive skill revisions from drifting in inconsistent directions.

## Transferability

SkillOpt results transfer across contexts:
- **Cross-model:** skill optimized on GPT-5.5 benefits smaller model variants
- **Cross-harness:** Codex-trained SpreadsheetBench skill → Claude Code: **+59.7 points**
- **Cross-benchmark:** OlympiadBench-optimized skill yields positive transfer to Omni-MATH

One optimization pass produces a plain-text artifact reusable across related models,
execution environments, or tasks — without changing model weights.

## Hyperparameter Reference

| Parameter | Default | Notes |
|-----------|---------|-------|
| rollout_batch_size | 8 | Trajectories per update. More = stable, slower |
| reflection_minibatch_size | 4 | Minibatch size per path |
| edit_budget_init | 5 | Initial edit budget |
| edit_budget_final | 1 | Final edit budget (cosine decay) |
| schedule | cosine | cosine / constant / linear |
| max_epochs | 4 | |
| max_steps_per_epoch | 4 | |
| accept_strict | True | False = non-strict (≥) acceptance |
| slow_update_interval | per_epoch | |
| rejected_buffer_size | 20 | |

## Relation to Hermes Skill System

| Hermes already has | SkillOpt adds |
|--------------------|---------------|
| Agent loads SKILL.md to guide behavior | Systematic optimization of SKILL.md content |
| Manual skill authoring | Trajectory-based automated text optimization |
| Static skill documents | Trainable, iterable external state |
| CLI or prompt edits | Validation-gated bounded edit loop |
| No version control | best_skill.md + full edit history |
