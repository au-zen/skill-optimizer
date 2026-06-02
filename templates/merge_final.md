# Final Merge Prompt

Combine failure-prevention and success-preservation edits into a single merged list:

1. **Failure corrections take priority over success preservation** — if a success edit would prevent a necessary failure fix, keep the failure fix.
2. **Resolve conflicts** — if a failure edit and a success edit contradict each other, keep the one with higher expected gain. If equal, keep the failure fix.
3. **Output a deduplicated list** — ensure no duplicates survived from both sides.
4. **Preserve rationale** — keep the rationale from whichever edit is retained.

Output format — return a JSON object:
```json
{
  "edits": [
    {
      "operation": "add",
      "section": "## Rules",
      "new_content": "rule text",
      "rationale": "kept from failure edits (priority)",
      "expected_gain": 0.7
    }
  ],
  "resolved_conflicts": ["conflict between edit_2 and edit_7 → kept edit_2 (higher gain)"]
}
```
