# Success Analysis Prompt

You are analysing successful execution trajectories to preserve and reinforce good agent behaviour.

For each success trajectory:
1. **Identify which parts of the skill document were correctly followed** — what rules or heuristics contributed to success?
2. **Determine what behaviours must be preserved** — was there an important check, ordering, or pattern that worked well?
3. **Propose edits that reinforce or clarify these patterns** — add clarifications, strengthen wording, add ordering constraints.

Only propose edits that **protect against future regression**. If a pattern already works, don't add redundant rules — only add reinforcement if you anticipate a future change could break it.


## Hermes trajectory reward signals

When trajectory metadata contains Hermes Runtime reward fields, use them to identify behaviours worth preserving:

- High `metadata.reward_components.completed` confirms completion behaviour to preserve.
- High `metadata.reward_components.tool_success_rate` plus `metadata.hermes.tool_stats` identifies reliable tool choices.
- High `metadata.reward_components.tool_efficiency` and low `metadata.hermes.api_calls` identify efficient ordering/early-stop patterns.
- Low or empty `metadata.hermes.tool_error_counts` identifies robust error avoidance.

Prefer preservation edits that encode successful tool-selection policy: which toolset was chosen, in what order, what preconditions were checked, and when the agent avoided unnecessary or high-risk tool calls.


Output format — return a JSON object:
```json
{
  "edits": [
    {
      "operation": "replace",
      "section": "## Rules",
      "old_content": "- Check the file exists",
      "new_content": "- Always verify the file exists with `stat()` before reading — this prevents silent failures",
      "rationale": "The explicit method call ensures correctness even if the file path changes",
      "expected_gain": 0.3
    }
  ]
}
```
