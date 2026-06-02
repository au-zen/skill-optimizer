# Failure Merge Prompt

Consolidate the proposed failure-prevention edits from multiple analyst runs:

1. **Remove duplicates** — if two edits propose the same fix, keep only one.
2. **Remove contradictions** — if one edit says "always X" and another says "never X", resolve the conflict.
3. **Generalise example-specific fixes** — if an edit fixes a specific case, can it be broadened to a reusable rule?
4. **Prioritise** — rank by expected impact. High-impact, general edits should outrank narrow, low-impact ones.
5. **Tag conflicts with success edits** — if a failure-prevention edit would break a known success pattern, flag it.

Output format — return a JSON object:
```json
{
  "edits": [
    {
      "operation": "add",
      "section": "## Rules",
      "old_content": "",
      "new_content": "- Generalised rule from 3 specific fixes",
      "rationale": "Consolidated: 3 related failure patterns point to the same root cause",
      "expected_gain": 0.8
    }
  ],
  "suppressed": ["edit_4 (duplicate of edit_1)"]
}
```
