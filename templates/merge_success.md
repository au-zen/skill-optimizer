# Success Merge Prompt

Consolidate the proposed preservation edits from multiple success-analyst runs:

1. **Remove duplicates** — keep only one instance of each rule reinforcement.
2. **Merge complementary preserving rules** — if two rules protect the same behaviour, combine them into one clear statement.
3. **Flag conflicts with failure edits** — tag any rule that contradicts a failure-prevention edit (these will be resolved in the final merge).

Output format — return a JSON object:
```json
{
  "edits": [
    {
      "operation": "replace",
      "section": "## Rules",
      "old_content": "old rule text",
      "new_content": "merged, clearer rule text",
      "rationale": "Combined 2 complementary preservation edits",
      "expected_gain": 0.4
    }
  ],
  "conflicts_with_failure": ["edit_3 (would conflict with failure fix #2)"]
}
```
