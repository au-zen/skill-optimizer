# Slow Update Prompt

Review all edits accepted and rejected during this epoch. Extract long-horizon lessons.

1. **Identify recurring failure modes** — what patterns appeared across multiple steps? These are systemic, not one-off.
2. **Extract stable, long-horizon lessons** — lessons that should persist across multiple epochs. Not task-specific, but about skill structure.
3. **Identify what didn't work** — patterns of rejected edits. What types of edits keep failing validation?
4. **Identify what consistently worked** — types of edits that were accepted and improved scores.

Write a concise summary (2-5 paragraphs) to the **protected slow-update field**. This field:
- Cannot be modified by step-level edits
- Is updated only at epoch boundaries
- Serves as a "momentum" term, anchoring the skill's evolution

Output format — return a JSON object:
```json
{
  "slow_field": "## Slow-Update Field (Protected)\n\n### Long-horizon lessons from Epoch {epoch}...",
  "meta_notes": {
    "recurring_patterns": ["pattern A", "pattern B"],
    "effective_edit_types": ["add rules", "clarify ambiguous phrasing"],
    "ineffective_edit_types": ["wording-only changes"]
  }
}
```
