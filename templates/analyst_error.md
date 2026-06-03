# Failure Analysis Prompt

You are analysing failed execution trajectories to improve the agent skill document.

For each failure trajectory:
1. **Identify the procedural error** — not the output error itself, but what rule or heuristic would have prevented it.
2. **Determine the root cause** — was it missing instructions, ambiguous phrasing, contradictory rules, or a missing edge case?
3. **Propose a specific bounded edit** — an add/delete/replace operation targeting a specific section of the skill document.
4. **Rate expected impact** — high/medium/low.

Focus on **recurring, generalisable failures**, not one-off mistakes. A single failure might be noise; the same failure across multiple trajectories is a skill gap.


## Hermes trajectory reward signals

When trajectory metadata contains Hermes Runtime reward fields, treat them as primary evidence:

- `metadata.reward_components.completed` shows whether the task completed.
- `metadata.reward_components.tool_success_rate` and `metadata.hermes.tool_stats` show whether the tool-selection policy worked.
- `metadata.reward_components.tool_efficiency` and `metadata.hermes.api_calls` show whether the run wasted calls.
- `metadata.reward_components.error_penalty` and `metadata.hermes.tool_error_counts` show tool/runtime errors.
- `metadata.tool_policy_diagnostics.high_failure_tools` and `overused_tools` should drive concrete tool-policy edits.

Prefer edits that change tool-use behaviour, not just prose quality. For example, add rules for when to prefer `read_file` over `terminal`, when to stop retrying a failing tool, when to inspect available skills before calling `skill_view`, or how to handle tool unavailable/runtime errors.


Output format — return a JSON object:
```json
{
  "edits": [
    {
      "operation": "add",
      "section": "## Rules",
      "old_content": "",
      "new_content": "- When doing X, always check Y first",
      "rationale": "Two of three failures were caused by skipping Y",
      "expected_gain": 0.7
    }
  ]
}
```
