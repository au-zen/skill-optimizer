# Ranking and Selection Prompt

From the merged edit pool, select the top {budget} edits for this optimisation step.

Guidelines:
1. **Rank by expected impact** — edits that fix recurring failures rank higher than speculative improvements.
2. **Consider edit diversity** — don't over-optimise one issue. Spread edits across different sections and problem types.
3. **Ensure compatibility** — selected edits should not contradict each other.
4. **Check against rejected buffer** — the rejected buffer contains edits that previously failed validation. Avoid repeating those patterns.

Selection criteria (in order):
- Edits targeting recurring failure modes
- Edits with high expected_gain (> 0.6)
- Edits that add new rules (vs. wording changes)
- Edits on different sections of the skill document

Current rejected-edit buffer (previously failed edits to avoid):
{rejected_buffer}

Output exactly {budget} edits. If fewer than {budget} are available, return all of them.

Output format — return a JSON object:
```json
{
  "edits": [
    {
      "operation": "add",
      "section": "## Rules",
      "new_content": "rule text",
      "rationale": "why selected",
      "expected_gain": 0.8
    }
  ],
  "selection_rationale": "Brief explanation of why these specific edits were chosen"
}
```
