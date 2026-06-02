# Failure Analysis Prompt

You are analysing failed execution trajectories to improve the agent skill document.

For each failure trajectory:
1. **Identify the procedural error** — not the output error itself, but what rule or heuristic would have prevented it.
2. **Determine the root cause** — was it missing instructions, ambiguous phrasing, contradictory rules, or a missing edge case?
3. **Propose a specific bounded edit** — an add/delete/replace operation targeting a specific section of the skill document.
4. **Rate expected impact** — high/medium/low.

Focus on **recurring, generalisable failures**, not one-off mistakes. A single failure might be noise; the same failure across multiple trajectories is a skill gap.

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
