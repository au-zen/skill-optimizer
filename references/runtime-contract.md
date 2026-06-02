# SkillOpt Runtime Contract

These are engine invariants — system correctness guarantees that must not be
modified by the automated optimizer. They are separated here so `## Optimization
Policy` in SKILL.md remains the clean, sole target of bounded edits.

## Evaluation

- **Candidate-aware scoring.** The scoring command must reference the candidate
  file via `{skill_path}` substitution. Running a fixed command that ignores
  `skill_path` produces identical scores for every candidate — the loop is
  broken. Verify with: `metrics.jsonl` must show score variance across steps.

- **True baseline initialization.** `current_score` must be set to the real
  score of the original skill on D_sel at `run()` startup — not initialized to
  `0.0`. A zero baseline allows any positive-scoring candidate to pass on the
  first step, inflating the accepted-edit count with false positives.

## Data Integrity

- `best_skill.md` is written exclusively to `work_dir/`. The original skill file
  is never modified by the engine.

- `origin_skill_path` is set once at `run()` startup and never updated. Drift
  detection (`min_similarity`) compares every candidate against this origin, not
  against the current best.

- `train` and `sel` splits must be sourced from separate files. Overlap between
  train and validation defeats the gating mechanism.

## API & Configuration

- The 3-tier fallback chain (Tier1: opencode-zen → Tier2: DeepSeek direct →
  Tier3: OpenRouter free) is configured exclusively via `.env`. The engine reads
  this chain but does not modify it.

- `os.environ[key] = value` override in `llm_client.py` is intentional: it
  decouples optskill credentials from the Hermes parent process. This must not
  be changed to a conditional set (`if key not in os.environ`) without explicit
  decision to allow parent-env override.

## Structural Validation

After every `_apply_edits()`, the candidate must pass:
1. YAML frontmatter present (`---` at byte 0, closing `---`)
2. No unclosed fenced code blocks (even count of ` ``` `)
3. At least one markdown section heading

Candidates failing structural validation are silently discarded (not sent to
`_validate_and_gate`), and the step is recorded as a no-op.
