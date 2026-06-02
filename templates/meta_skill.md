# Optimizer Meta-Skill Prompt

Maintain optimiser-level introspection state across epochs. This is the optimiser's own "memory" of what has and hasn't worked.

Record:
- **What edit strategies have been consistently effective?**
  - e.g., "Adding explicit ordering rules" vs "Rephrasing existing rules"
- **What failure patterns proved resistant to correction?**
  - e.g., "The agent keeps skipping validation despite the rule in ## Pitfalls"
- **What optimisations led to validation gate rejections?**
  - e.g., "Trying to merge 3 rules into 1 → caused confusion, score dropped"
- **What skill sections are most/least responsive to editing?**
  - e.g., "## Rules edits consistently improve scores; ## Examples edits have mixed effect"

Output format — return a JSON object:
```json
{
  "meta_skill": "## Optimizer Meta-Skill\n\n### Effective Strategies...",
  "summary": {
    "effective_patterns": ["add ordering constraints", "clarify edge cases"],
    "ineffective_patterns": ["delete existing rules", "merge multiple rules"],
    "resistant_failures": ["validation step skipped"],
    "responsive_sections": ["## Rules", "## Pitfalls"],
    "unresponsive_sections": ["## Overview"]
  }
}
```
