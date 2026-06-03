"""
SkillOpt Engine — Text-space optimizer for agent skill documents.

Core optimization loop (see paper arXiv:2605.23904):
  1. Rollout       — Execute target model with current skill on D_tr.
  2. Reflect       — Analyse failures & successes, propose bounded edits.
  3. Apply         — Patch SKILL.md with top-L_t edits → candidate.
  4. Validate      — Gate candidate on D_sel; accept iff strict improvement.
  5. Slow update   — At epoch boundary, distil long-horizon lessons.

Zero inference-time overhead: the optimised skill is a plain-text artifact.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .protocols import (
    Edit,
    EditBatch,
    OptimizationState,
    OptimizerConfig,
    Trajectory,
)
from .llm_client import OptimizerLLMClient, load_prompt
from .rollout import RolloutHarness


# ── Section boundary regex for skill documents ────────────────────────

_SECTION_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _compute_text_similarity(text_a: str, text_b: str) -> float:
    """Jaccard-like similarity on line-sorted tokens."""
    tokens_a = set(text_a.lower().splitlines())
    tokens_b = set(text_b.lower().splitlines())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / max(len(union), 1)


def _find_section(content: str, section: str) -> tuple[int, int] | None:
    """Find the byte-range (start, end) of a markdown section heading.

    Returns (line_start_offset, line_end_offset) of the section body
    (including the heading, excluding the next section).

    ``section`` may be "Rules", "## Rules", or "# Rules" — the code
    normalises it by stripping any leading ``#`` characters and whitespace.
    """
    # Normalise: strip leading # and whitespace
    section_clean = section.lstrip("#").strip().lower()

    lines = content.splitlines(keepends=True)
    heading_line = None

    for idx, line in enumerate(lines):
        m = _SECTION_RE.match(line)
        if m and (m.group(2).strip().lower() == section_clean):
            heading_line = idx
            break

    if heading_line is None:
        return None

    start = heading_line
    end = len(lines)

    for idx in range(heading_line + 1, len(lines)):
        if _SECTION_RE.match(lines[idx]):
            end = idx
            break

    # Find byte offsets
    byte_start = sum(len(lines[i]) for i in range(start))
    byte_end = sum(len(lines[i]) for i in range(end))
    return (byte_start, byte_end)


# ── Parser for structured edit output ─────────────────────────────────

def _parse_edit_list(raw: Any) -> list[Edit]:
    """Parse LLM output into a list of Edit objects.

    Accepts:
      - A list of dicts (direct JSON parsing)
      - A dict with an "edits" key
      - A string containing JSON
    """
    if isinstance(raw, str):
        from .llm_client import _extract_json
        raw = _extract_json(raw) or []

    if isinstance(raw, dict):
        raw = raw.get("edits", raw.get("edits_list", [raw]))

    edits = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            edits.append(Edit(
                operation=str(item.get("operation", "add")),
                section=str(item.get("section", "")),
                old_content=str(item.get("old_content", "")),
                new_content=str(item.get("new_content", item.get("content", ""))),
                rationale=str(item.get("rationale", "")),
                expected_gain=float(item.get("expected_gain", 0.0)),
            ))
        except (ValueError, TypeError):
            continue
    return edits


# ── SkillOpt Engine ──────────────────────────────────────────────────


class SkillOptEngine:
    """Main optimisation loop controller."""

    def __init__(
        self,
        *,
        target_skill_path: str,
        config: OptimizerConfig | None = None,
        llm_client: OptimizerLLMClient | None = None,
        rollout_harness: RolloutHarness | None = None,
        train_split: list[str] | None = None,
        sel_split: list[str] | None = None,
        test_split: list[str] | None = None,
    ):
        self.config = config or OptimizerConfig()

        self.work_dir = Path(self.config.work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.state = OptimizationState(
            current_skill_path=str(Path(target_skill_path).resolve(strict=False)),
            best_skill_path=str(self.work_dir / "best_skill.md"),
        )
        holdout = test_split or []
        self.splits = {
            "train": train_split or [],
            "sel": sel_split or [],
            "test": test_split or [],
            "holdout": holdout,
        }
        # Stage 3: secondary scorer placeholder
        self._secondary_scorer = None

        self.llm = llm_client or OptimizerLLMClient(
            model=self.config.optimizer_model,
            api_base=self.config.optimizer_api_base,
        )
        self.harness = rollout_harness or RolloutHarness(mode="manual")

        # Internal metrics
        self._metrics: list[dict] = []
        self._last_sel_trajs: list[Trajectory] = []

    # ═════════════════════════════════════════════════════════════════
    #  Public API
    # ═════════════════════════════════════════════════════════════════

    def run(self) -> OptimizationState:
        """Run the full optimisation loop (all epochs, all steps)."""
        resolved = str(Path(self.state.current_skill_path).resolve())
        self.state.current_skill_path = resolved
        self.state.origin_skill_path = resolved
        self.state.origin_skill_hash = self.state.compute_skill_hash(resolved)

        if not self.splits["sel"]:
            raise ValueError(
                "A validation split is required for optimization. Provide --sel or "
                "--sel-trajs so candidates can be gated on D_sel."
            )

        shutil.copy2(resolved, self.state.best_skill_path)
        self.state.best_skill_path = str(Path(self.state.best_skill_path).resolve())

        # Stage 1: real baseline initialization
        baseline_trajs = self._rollout(resolved, self.splits["sel"])
        self._last_sel_trajs = baseline_trajs
        if baseline_trajs:
            self.state.current_score = sum(t.score for t in baseline_trajs) / len(baseline_trajs)
            self.state.best_score = self.state.current_score
            print(f"[SkillOpt] Baseline score (D_sel): {self.state.current_score:.4f}")
        else:
            print("[SkillOpt] [WARN] No trajectories from baseline rollout; score stays at 0.0")

        print(f"[SkillOpt] Starting optimisation of {resolved}")
        print(f"[SkillOpt] Train: {len(self.splits['train'])} tasks  "
              f"Sel: {len(self.splits['sel'])} tasks  "
              f"Test: {len(self.splits['test'])} tasks  "
              f"Holdout: {len(self.splits.get('holdout', []))} tasks")
        print(f"[SkillOpt] Origin: {self.state.origin_skill_path}")

        for _ in range(self.config.max_epochs):
            self.run_epoch()

        print(f"\n[SkillOpt] Done. Best score: {self.state.best_score:.4f}")
        print(f"[SkillOpt] Best skill: {self.state.best_skill_path}")
        print(f"[SkillOpt] Origin anchor: {self.state.origin_skill_path}")
        print(f"[SkillOpt] Drift events: {self.state.drift_count}")
        self._save_state()
        return self.state

    def run_epoch(self):
        """Run one complete epoch of the optimisation loop."""
        self.state.step = 0
        print(f"\n{'='*60}")
        print(f"[SkillOpt] Epoch {self.state.epoch + 1}/{self.config.max_epochs}")
        print(f"{'='*60}")

        for step in range(self.config.max_steps_per_epoch):
            self.state.step = step
            budget = self.compute_edit_budget(
                self.state.epoch, step, self.config
            )
            print(f"\n  ── Step {step + 1}/{self.config.max_steps_per_epoch}  "
                  f"(budget={budget}) ──")

            # === Forward Pass ===
            trajectories = self._rollout(
                self.state.current_skill_path, self.splits["train"]
            )
            print(f"      Rollout: {len(trajectories)} trajectories  "
                  f"({sum(1 for t in trajectories if t.is_success)} success)")

            if not trajectories:
                print("      No trajectories, skipping step.")
                continue

            # === Backward Pass ===
            edits = self._reflect_and_propose(trajectories, budget)
            if not edits:
                print("      No edits proposed.")
                continue
            print(f"      Proposed {len(edits)} edits (budget={budget})")

            # === Apply edits → candidate ===
            candidate_path = self._apply_edits(
                self.state.current_skill_path, edits
            )
            if candidate_path is None:
                print("      Failed to apply edits.")
                continue

            # === Validation Gate ===
            accepted = self._validate_and_gate(candidate_path)
            if accepted:
                print(f"      ✓ Accepted  (score: {self.state.current_score:.4f})")
                if self.state.current_score > self.state.best_score:
                    self.state.best_score = self.state.current_score
                    shutil.copy2(candidate_path, self.state.best_skill_path)
                    self.state.best_skill_path = str(
                        Path(self.state.best_skill_path).resolve()
                    )
                    print(f"      ★ New best!  ({self.state.best_score:.4f})")
                # Track accepted edits
                for e in edits:
                    self.state.accepted_edits.append({
                        **e.to_dict(),
                        "epoch": self.state.epoch,
                        "step": step,
                        "score": self.state.current_score,
                    })
            else:
                print(f"      ✗ Rejected")
                # Store in rejected buffer
                self.state.rejected_buffer.append({
                    "edits": [e.to_dict() for e in edits],
                    "epoch": self.state.epoch,
                    "step": step,
                })
                # Bound buffer size
                if len(self.state.rejected_buffer) > self.config.rejected_buffer_size:
                    self.state.rejected_buffer = self.state.rejected_buffer[
                        -self.config.rejected_buffer_size :
                    ]

            self._save_state()
            self._save_metrics()

        # === Slow/Meta Update at epoch boundary ===
        self._slow_update()
        self.state.skill_hash = self.state.compute_skill_hash(
            self.state.current_skill_path
        )
        self.state.epoch += 1
        self._save_state()

    # ═════════════════════════════════════════════════════════════════
    #  Step components
    # ═════════════════════════════════════════════════════════════════

    def _rollout(
        self, skill_path: str, tasks: list[str]
    ) -> list[Trajectory]:
        """Execute target model with skill on D_tr."""
        if not tasks:
            return []
        return self.harness.execute(
            skill_path, tasks, batch_size=self.config.rollout_batch_size
        )

    def _reflect_and_propose(
        self, trajectories: list[Trajectory], budget: int
    ) -> list[Edit]:
        """LLM-driven reflection: failures → success → merge → rank → clip."""
        # Apply config threshold to trajectories
        for t in trajectories:
            t.is_success_threshold = self.config.score_threshold

        failures = [t for t in trajectories if not t.is_success]
        successes = [t for t in trajectories if t.is_success]

        mb_size = self.config.reflection_minibatch_size

        # ── 1. Failure analysis ──
        failure_edits: list[Edit] = []
        for i in range(0, max(len(failures), 1), mb_size):
            batch = failures[i : i + mb_size]
            edits = self._reflect_minibatch(batch, "failure")
            failure_edits.extend(edits)

        # ── 2. Success analysis ──
        success_edits: list[Edit] = []
        for i in range(0, max(len(successes), 1), mb_size):
            batch = successes[i : i + mb_size]
            edits = self._reflect_minibatch(batch, "success")
            success_edits.extend(edits)

        # ── 3. Merge failure edits ──
        merged_failure = self._merge_edits(failure_edits, "failure")

        # ── 4. Merge success edits ──
        merged_success = self._merge_edits(success_edits, "success")

        # ── 5. Final merge ──
        all_candidates = self._merge_final(merged_failure, merged_success)

        # ── 6. Rank & clip to budget ──
        top_edits = self._rank_and_select(all_candidates, budget)

        return top_edits

    def _reflect_minibatch(
        self, trajectories: list[Trajectory], kind: str
    ) -> list[Edit]:
        """Run analyst_error or analyst_success on one minibatch."""
        if not trajectories:
            return []

        prompt_name = "analyst_error" if kind == "failure" else "analyst_success"

        # Build trajectory summary for the prompt
        traj_summary = ""
        for t in trajectories:
            traj_summary += (
                f"### Task: {t.task_id}\n"
                f"Input: {t.task_input[:500]}\n"
                f"Output: {t.output[:500]}\n"
                f"Score: {t.score:.2f}\n"
                f"Error: {t.error or 'None'}\n\n"
            )

        # Include rejected buffer as negative feedback
        rejected_context = ""
        if self.state.rejected_buffer:
            recent = self.state.rejected_buffer[-5:]
            rejected_context = "Previously rejected edits (avoid these patterns):\n"
            for entry in recent:
                for e in entry.get("edits", []):
                    rejected_context += f"- [{e.get('operation','')}] {e.get('section','')}: {e.get('rationale','')[:200]}\n"
            rejected_context += "\n"

        prompt = load_prompt(prompt_name)
        full_prompt = (
            f"## Current Skill (excerpts)\n"
            f"Skill path: {self.state.current_skill_path}\n\n"
            f"## Execution Trajectories\n"
            f"{traj_summary}\n"
            f"{rejected_context}\n"
            f"---\n"
            f"{prompt}\n\n"
            f"Return your response as a JSON object with an 'edits' array. "
            f"Each edit must have: operation (add/delete/replace), section "
            f"(the markdown heading), old_content (for delete/replace), "
            f"new_content (for add/replace), rationale, expected_gain."
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a SkillOpt optimizer analysing agent execution "
                        "trajectories and proposing bounded text edits to improve "
                        "the agent's skill document. Return structured JSON."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ]
            result = self.llm.chat_structured(messages)
            edits = _parse_edit_list(result)
            return edits
        except Exception as e:
            print(f"      [WARN] Reflection failed: {e}")
            return []

    def _merge_edits(
        self, edits: list[Edit], kind: str
    ) -> list[Edit]:
        """Merge/consolidate edits using the merge prompt."""
        if len(edits) <= 1:
            return edits

        prompt_name = "merge_failure" if kind == "failure" else "merge_success"

        edits_summary = json.dumps(
            [e.to_dict() for e in edits], indent=2, ensure_ascii=False
        )

        prompt = load_prompt(prompt_name)
        full_prompt = (
            f"## Proposed {kind} edits to merge and consolidate:\n\n"
            f"{edits_summary}\n\n"
            f"{prompt}\n\n"
            f"Return your response as a JSON object with an 'edits' array. "
            f"Each edit must have: operation, section, old_content, "
            f"new_content, rationale, expected_gain."
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a SkillOpt editor-merging assistant. Consolidate "
                        "edit proposals, remove duplicates, generalise narrow fixes. "
                        "Return structured JSON."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ]
            result = self.llm.chat_structured(messages)
            merged = _parse_edit_list(result)
            return merged if merged else edits
        except Exception as e:
            print(f"      [WARN] Merge ({kind}) failed: {e}")
            return edits

    def _merge_final(
        self, failure_edits: list[Edit], success_edits: list[Edit]
    ) -> list[Edit]:
        """Final merge: failure edits take priority over success edits."""
        all_edits = failure_edits + success_edits
        if not all_edits:
            return []

        if not failure_edits or not success_edits:
            return all_edits

        prompt = load_prompt("merge_final")
        summary = json.dumps(
            {
                "failure_edits": [e.to_dict() for e in failure_edits],
                "success_edits": [e.to_dict() for e in success_edits],
            },
            indent=2,
            ensure_ascii=False,
        )

        full_prompt = (
            f"## Failure-prevention edits (priority):\n"
            f"{json.dumps([e.to_dict() for e in failure_edits], indent=2)}\n\n"
            f"## Success-preservation edits:\n"
            f"{json.dumps([e.to_dict() for e in success_edits], indent=2)}\n\n"
            f"{prompt}\n\n"
            f"Return a JSON object with an 'edits' array (merged, deduplicated)."
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a SkillOpt merge coordinator. Combine failure and "
                        "success edits, resolve conflicts, return deduplicated list. "
                        "Return structured JSON."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ]
            result = self.llm.chat_structured(messages)
            merged = _parse_edit_list(result)
            return merged if merged else all_edits
        except Exception as e:
            print(f"      [WARN] Final merge failed: {e}")
            return all_edits

    def _rank_and_select(
        self, edits: list[Edit], budget: int
    ) -> list[Edit]:
        """Rank edits by expected impact, clip to budget.

        If budget >= len(edits), just sort by expected_gain and return all.
        Otherwise, use LLM ranking for intelligent selection.
        """
        if not edits:
            return []

        # Sort by expected_gain descending (fallback)
        edits_sorted = sorted(edits, key=lambda e: e.expected_gain, reverse=True)

        if len(edits) <= budget:
            return edits_sorted

        # Need LLM-assisted ranking when budget is tight
        prompt = load_prompt(
            "ranking",
            budget=budget,
            rejected_buffer=json.dumps(
                self.state.rejected_buffer[-10:], indent=2, ensure_ascii=False
            ),
        )
        edits_summary = json.dumps(
            [e.to_dict() for e in edits], indent=2, ensure_ascii=False
        )

        full_prompt = (
            f"## Available edits (ranked by expected_gain as initial sort):\n"
            f"{edits_summary}\n\n"
            f"{prompt}\n\n"
            f"Return a JSON object with an 'edits' array containing exactly {budget} "
            f"edit objects. Each must have: operation, section, old_content, "
            f"new_content, rationale, expected_gain."
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a SkillOpt ranking assistant. Select the most "
                        "promising edits within the given budget. Return structured JSON."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ]
            result = self.llm.chat_structured(messages)
            ranked = _parse_edit_list(result)
            if ranked:
                return ranked[:budget]  # Safety clip
        except Exception as e:
            print(f"      [WARN] Ranking failed: {e}")

        # Fallback: return top by expected_gain
        return edits_sorted[:budget]

    # ═════════════════════════════════════════════════════════════════
    #  Edit application
    # ═════════════════════════════════════════════════════════════════

    def _apply_edits(
        self, skill_path: str, edits: list[Edit]
    ) -> str | None:
        """Apply edits to skill document → produce candidate file.

        Returns path to candidate file, or None on failure.
        """
        source = Path(skill_path).resolve()
        if not source.exists():
            print(f"      [ERR] Skill not found: {source}")
            return None

        content = source.read_text(encoding="utf-8")

        for edit in edits:
            op = edit.operation.lower().strip()
            section = edit.section.strip()

            if op == "add":
                content = self._edit_add(content, section, edit.new_content)
            elif op == "delete":
                content = self._edit_delete(
                    content, section, edit.old_content
                )
            elif op == "replace":
                content = self._edit_replace(
                    content, section, edit.old_content, edit.new_content
                )
            else:
                print(f"      [WARN] Unknown operation: {op}")

        # ── Structural validation ──
        issues = self._validate_skill_structure(content)
        if issues:
            for issue in issues:
                print(f"      [VALIDATION] {issue}")
            print("      [ERR] Candidate fails structural validation — edits reverted.")
            return None

        # Write candidate
        candidate_dir = self.work_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = (
            candidate_dir
            / f"candidate_e{self.state.epoch}s{self.state.step}.md"
        )
        candidate_path.write_text(content, encoding="utf-8")
        return str(candidate_path)

    @staticmethod
    def _validate_skill_structure(content: str) -> list[str]:
        """Check markdown structural integrity after edits.

        Returns a list of issue descriptions (empty = valid).
        """
        issues: list[str] = []

        # 1. YAML frontmatter must be present
        stripped = content.lstrip()
        if not stripped.startswith("---"):
            issues.append("Missing YAML frontmatter (must start with ---)")
        else:
            # Find closing ---
            end_idx = content.find("---", 3)
            if end_idx == -1:
                issues.append("Unclosed YAML frontmatter (no closing --- found)")
            else:
                # Ensure at least one key-value pair in frontmatter
                fm = content[3:end_idx].strip()
                if not fm:
                    issues.append("Empty YAML frontmatter")

        # 2. No unclosed fenced code blocks (odd count of ```)
        fence_count = content.count("```")
        if fence_count % 2 != 0:
            issues.append(f"Unclosed fenced code block ({fence_count} backtick fences)")

        # 3. At least one section heading
        headings = re.findall(r"^#{1,3}\s+\S", content, re.MULTILINE)
        if not headings:
            issues.append("No markdown section headings found")

        return issues

    @staticmethod
    def _sanitize_section(name: str) -> str:
        """Strip markdown heading markers from a section name."""
        return name.lstrip("#").strip()

    @staticmethod
    def _edit_add(content: str, section: str, new_text: str) -> str:
        """Append new content to a section."""
        clean_section = SkillOptEngine._sanitize_section(section)
        span = _find_section(content, clean_section)
        if span is None:
            # Section not found: append at end
            content += f"\n\n## {clean_section}\n{new_text}\n"
            return content

        end = span[1]
        # Insert before the next section if this is at the end
        if end >= len(content):
            content += f"\n{new_text}\n"
        else:
            # Insert just before the next section heading
            insert_pos = content.rfind("\n", 0, end)
            if insert_pos < 0:
                insert_pos = end
            content = (
                content[:insert_pos]
                + f"\n{new_text}\n"
                + content[insert_pos:]
            )
        return content

    @staticmethod
    def _edit_delete(content: str, section: str, old_text: str) -> str:
        """Delete specific text from a section."""
        if not old_text:
            return content
        # Try precise match first
        replaced = content.replace(old_text, "")
        if replaced != content:
            return replaced
        # Fuzzy: remove lines containing the text (within the section)
        span = _find_section(content, section)
        if span is None:
            return content
        before = content[: span[0]]
        section_body = content[span[0] : span[1]]
        after = content[span[1] :]

        # Remove lines that contain the old_text (fuzzy match)
        lines = section_body.splitlines(keepends=True)
        filtered = [l for l in lines if old_text.strip().lower() not in l.lower()]
        return before + "".join(filtered) + after

    @staticmethod
    def _edit_replace(
        content: str, section: str, old_text: str, new_text: str
    ) -> str:
        """Replace old_text with new_text within the given section."""
        if not old_text:
            return SkillOptEngine._edit_add(content, section, new_text)
        # Precise replacement within the whole doc
        replaced = content.replace(old_text, new_text, 1)
        if replaced != content:
            return replaced
        # Fuzzy: find in section and replace
        span = _find_section(content, section)
        if span is None:
            return content
        before = content[: span[0]]
        section_body = content[span[0] : span[1]]
        after = content[span[1] :]

        new_section_body = section_body.replace(old_text, new_text, 1)
        if new_section_body == section_body:
            # Fuzzy line-level replacement
            lines = section_body.splitlines(keepends=True)
            for i, line in enumerate(lines):
                if old_text.strip().lower() in line.lower():
                    indent = line[: len(line) - len(line.lstrip())]
                    lines[i] = f"{indent}{new_text}\n"
                    break
            new_section_body = "".join(lines)
        return before + new_section_body + after

    # ═════════════════════════════════════════════════════════════════
    #  Validation gate
    # ═════════════════════════════════════════════════════════════════

    def _validate_and_gate(self, candidate_path: str) -> bool:
        """Run candidate on D_sel. Accept iff strict improvement."""
        if not self.splits["sel"]:
            print(
                "      [ERR] No sel split configured. A validation split (--sel or --sel-trajs) "
                "is REQUIRED for the validation gate. Without it, the gate would unconditionally "
                "accept every edit, defeating the core protection mechanism."
            )
            return False

        # Stage 2: drift detection against origin_skill
        if self.state.origin_skill_path and self.config.min_similarity > 0:
            drift_ok = self._detect_drift(candidate_path)
            if not drift_ok:
                print(f"      [DRIFT] Candidate diverges too far from origin — rejected.")
                return False

        # Score the candidate on sel split
        candidate_trajs = self._rollout(candidate_path, self.splits["sel"])
        if not candidate_trajs:
            return False

        candidate_summary = self._summarize_reward_metrics(candidate_trajs)
        current_trajs = self._last_sel_trajs
        if not current_trajs:
            current_trajs = self._rollout(self.state.current_skill_path, self.splits["sel"])
            self._last_sel_trajs = current_trajs
        current_summary = self._summarize_reward_metrics(current_trajs) if current_trajs else {
            "mean_score": self.state.current_score,
            "completed_rate": 0.0,
            "tool_failure_rate": 1.0,
            "tool_success_rate": 0.0,
            "mean_api_calls": 0.0,
        }
        candidate_score = candidate_summary["mean_score"]
        current_score = self.state.current_score

        secondary_score: float | None = None

        # Stage 3: secondary scorer cross-validation
        if self.config.use_secondary_scorer and self._secondary_scorer is not None:
            try:
                secondary_score = self._secondary_scorer(candidate_path)
                print(f"      Secondary scorer: {secondary_score:.4f}")
                if secondary_score < candidate_score * 0.5:
                    print(f"      [WARN] Secondary score ({secondary_score:.4f}) diverges "
                          f"from primary ({candidate_score:.4f}) — possible reward hacking")
            except Exception as e:
                print(f"      [WARN] Secondary scorer failed: {e}")

        # Stage 3: holdout evaluation (monitoring only, not gating)
        if self.config.holdout_rollout and self.splits.get("holdout"):
            holdout_trajs = self._rollout(candidate_path, self.splits["holdout"])
            if holdout_trajs:
                holdout_score = sum(t.score for t in holdout_trajs) / len(holdout_trajs)
                self.state.holdout_score = holdout_score
                print(f"      Holdout: {holdout_score:.4f}")

        metric = {
            "epoch": self.state.epoch,
            "step": self.state.step,
            "current_score": current_score,
            "candidate_score": candidate_score,
            "delta": candidate_score - current_score,
            "n_sel": len(candidate_trajs),
            "current_reward_metrics": current_summary,
            "candidate_reward_metrics": candidate_summary,
            "completed_rate_delta": candidate_summary["completed_rate"] - current_summary["completed_rate"],
            "tool_failure_rate_delta": candidate_summary["tool_failure_rate"] - current_summary["tool_failure_rate"],
            "tool_success_rate_delta": candidate_summary["tool_success_rate"] - current_summary["tool_success_rate"],
            "api_calls_delta": candidate_summary["mean_api_calls"] - current_summary["mean_api_calls"],
        }
        if secondary_score is not None:
            metric["secondary_score"] = secondary_score
        self._metrics.append(metric)

        score_delta = candidate_score - current_score
        if self.config.accept_strict:
            score_gate = score_delta > self.config.min_reward_delta
        else:
            score_gate = score_delta >= self.config.min_reward_delta

        completed_gate = (
            candidate_summary["completed_rate"] + self.config.completed_rate_tolerance
            >= current_summary["completed_rate"]
        )
        failure_gate = (
            candidate_summary["tool_failure_rate"]
            <= current_summary["tool_failure_rate"] + self.config.tool_failure_rate_tolerance
        )
        accepted = score_gate and completed_gate and failure_gate

        if not score_gate:
            print("      [GATE] Reward did not improve enough.")
        if not completed_gate:
            print("      [GATE] Completed rate regressed.")
        if not failure_gate:
            print("      [GATE] Tool failure rate regressed.")

        if accepted:
            self.state.current_score = candidate_score
            self.state.current_skill_path = candidate_path
            self._last_sel_trajs = candidate_trajs

        return accepted

    @staticmethod
    def _summarize_reward_metrics(trajectories: list[Trajectory]) -> dict:
        """Aggregate trajectory reward metadata for multidimensional gating."""
        if not trajectories:
            return {
                "mean_score": 0.0,
                "completed_rate": 0.0,
                "tool_failure_rate": 0.0,
                "tool_success_rate": 0.0,
                "mean_api_calls": 0.0,
            }

        mean_score = sum(t.score for t in trajectories) / len(trajectories)
        completed_values: list[float] = []
        total_tool_calls = 0
        total_tool_success = 0
        total_tool_failure = 0
        api_calls: list[float] = []

        for traj in trajectories:
            metadata = traj.metadata if isinstance(traj.metadata, dict) else {}
            components = metadata.get("reward_components") if isinstance(metadata.get("reward_components"), dict) else {}
            hermes = metadata.get("hermes") if isinstance(metadata.get("hermes"), dict) else {}

            if "completed" in components:
                completed_values.append(float(components.get("completed", 0.0)))
            elif "completed" in hermes:
                completed_values.append(1.0 if hermes.get("completed") else 0.0)
            else:
                completed_values.append(1.0 if traj.is_success else 0.0)

            if hermes.get("api_calls") is not None:
                try:
                    api_calls.append(float(hermes.get("api_calls")))
                except (TypeError, ValueError):
                    pass

            tool_stats = hermes.get("tool_stats") if isinstance(hermes.get("tool_stats"), dict) else {}
            for stats in tool_stats.values():
                if not isinstance(stats, dict):
                    continue
                count = int(stats.get("count") or 0)
                success = int(stats.get("success") or 0)
                failure = int(stats.get("failure") or 0)
                if count <= 0:
                    count = success + failure
                total_tool_calls += max(count, 0)
                total_tool_success += max(success, 0)
                total_tool_failure += max(failure, 0)

        tool_failure_rate = (total_tool_failure / total_tool_calls) if total_tool_calls else 0.0
        tool_success_rate = (total_tool_success / total_tool_calls) if total_tool_calls else 1.0
        mean_api_calls = (sum(api_calls) / len(api_calls)) if api_calls else 0.0

        return {
            "mean_score": round(mean_score, 4),
            "completed_rate": round(sum(completed_values) / len(completed_values), 4),
            "tool_failure_rate": round(tool_failure_rate, 4),
            "tool_success_rate": round(tool_success_rate, 4),
            "mean_api_calls": round(mean_api_calls, 4),
        }

    # ═════════════════════════════════════════════════════════════════
    #  Drift detection
    # ═════════════════════════════════════════════════════════════════

    def _detect_drift(self, candidate_path: str) -> bool:
        """Check candidate similarity against origin_skill. Returns True if safe."""

        origin_path = self.state.origin_skill_path
        if not origin_path or not Path(origin_path).exists():
            return True

        candidate_text = Path(candidate_path).read_text(encoding="utf-8")
        origin_text = Path(origin_path).read_text(encoding="utf-8")

        similarity = _compute_text_similarity(candidate_text, origin_text)
        min_sim = self.config.min_similarity

        if similarity < min_sim:
            self.state.drift_detected = True
            self.state.drift_count += 1
            print(f"      [DRIFT] similarity={similarity:.3f} < min={min_sim} "
                  f"(count={self.state.drift_count})")
            return False

        return True

    # ═════════════════════════════════════════════════════════════════
    #  Slow / meta update
    # ═════════════════════════════════════════════════════════════════

    def _slow_update(self):
        """At epoch boundary: extract long-horizon lessons into protected field."""
        if not self.state.accepted_edits and not self.state.rejected_buffer:
            return

        prompt = load_prompt("slow_update")
        summary = json.dumps(
            {
                "epoch": self.state.epoch,
                "accepted_edits": self.state.accepted_edits[-20:],
                "rejected_buffer": self.state.rejected_buffer[-20:],
                "current_score": self.state.current_score,
                "best_score": self.state.best_score,
            },
            indent=2,
            ensure_ascii=False,
        )

        full_prompt = (
            f"## Epoch {self.state.epoch} Summary\n\n{summary}\n\n{prompt}\n\n"
            f"Return a JSON object with a 'slow_field' string containing "
            f"the distilled lessons (2-5 paragraphs)."
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a SkillOpt slow-update analyst. Extract "
                        "long-horizon lessons from an epoch of edits. "
                        "Return structured JSON."
                    ),
                },
                {"role": "user", "content": full_prompt},
            ]
            result = self.llm.chat_structured(messages)
            if isinstance(result, dict):
                field = result.get("slow_field", result.get("content", ""))
                if field:
                    self.state.slow_field_content = field
        except Exception as e:
            print(f"      [WARN] Slow update failed: {e}")

    # ═════════════════════════════════════════════════════════════════
    #  Schedules
    # ═════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_edit_budget(
        epoch: int, step: int, config: OptimizerConfig
    ) -> int:
        """Compute the edit budget (text learning rate) for this step."""
        total_steps = config.max_epochs * config.max_steps_per_epoch
        current_step = epoch * config.max_steps_per_epoch + step
        progress = current_step / max(total_steps - 1, 1)
        init = config.edit_budget_init
        final = config.edit_budget_final

        if config.schedule == "constant":
            budget = init
        elif config.schedule == "linear":
            budget = final + (init - final) * (1 - progress)
        elif config.schedule == "cosine":
            budget = final + (init - final) * (1 + math.cos(progress * math.pi)) / 2
        else:
            budget = init

        return max(int(round(budget)), 1)

    # ═════════════════════════════════════════════════════════════════
    #  Persistence
    # ═════════════════════════════════════════════════════════════════

    def _save_state(self):
        """Persist optimisation state to work_dir/state.json."""
        path = self.work_dir / "state.json"
        path.write_text(
            json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_metrics(self):
        """Append all pending metrics to work_dir/metrics.jsonl."""
        path = self.work_dir / "metrics.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            for m in self._metrics:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        self._metrics.clear()  # Flush after write

    @staticmethod
    def load_state(work_dir: str) -> OptimizationState | None:
        """Load a saved optimisation state from a work directory."""
        path = Path(work_dir) / "state.json"
        if path.exists():
            return OptimizationState.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        return None
