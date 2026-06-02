"""
Rollout Harness — Executes target agent with a given skill and collects
scored trajectories.

Supported modes
---------------
manual      : Load pre-collected trajectories from a JSON file.
              Useful when running SkillOpt on existing data.
script      : Run a shell command or Python script per task.
              The harness captures stdout / exit code and scores it.
hermes_auto : (placeholder) For future Hermes-agent integration via
              delegate_task or cronjob subagents.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from .protocols import Trajectory


# ── Scorers ──────────────────────────────────────────────────────────


class Scorer:
    """Abstract base for scoring function."""

    def score(self, task_input: str, output: str, error: str | None = None) -> float:
        raise NotImplementedError


class KeywordScorer(Scorer):
    """Simple keyword-based scorer: presence of key terms → higher score."""

    def __init__(self, keywords: list[str] | None = None):
        self.keywords = keywords or []

    def score(self, task_input: str, output: str, error: str | None = None) -> float:
        if error:
            return 0.0
        if not output.strip():
            return 0.0
        kws_found = sum(1 for kw in self.keywords if kw.lower() in output.lower())
        total = max(len(self.keywords), 1)
        base = min(0.3 + 0.7 * (kws_found / total), 1.0)
        return base


class ExitCodeScorer(Scorer):
    """Exit-code-based scorer: 0 → 1.0, non-zero → 0.0."""

    def score(self, task_input: str, output: str, error: str | None = None) -> float:
        return 0.0 if error else 1.0


class JsonScorer(Scorer):
    """Extracts a 'score' field from JSON output."""

    def score(self, task_input: str, output: str, error: str | None = None) -> float:
        if error:
            return 0.0
        try:
            data = json.loads(output)
            return float(data.get("score", data.get("overall_score", 0.0)))
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0


class CustomScorer(Scorer):
    """Wrap a callable as a scorer."""

    def __init__(self, fn: Callable[[str, str, str | None], float]):
        self.fn = fn

    def score(self, task_input: str, output: str, error: str | None = None) -> float:
        return self.fn(task_input, output, error)


# ── Harness ──────────────────────────────────────────────────────────


class RolloutHarness:
    """Executes tasks and returns scored Trajectory objects."""

    def __init__(
        self,
        mode: str = "manual",
        scorer: Scorer | None = None,
        runner_fn: Callable | None = None,
    ):
        self.mode = mode
        self.scorer = scorer or KeywordScorer()
        self.runner_fn = runner_fn

    # ── Execution ──────────────────────────────────────────────────

    def execute(
        self, skill_path: str, tasks: list[str], batch_size: int = 8
    ) -> list[Trajectory]:
        """Run tasks with the given skill, return scored trajectories.

        Parameters
        ----------
        skill_path : Path to the SKILL.md to use.
        tasks      : List of task descriptors (strings or JSON task IDs).
        batch_size : How many tasks to run concurrently (sequential for now).

        Returns a list of Trajectory objects.
        """
        trajectories: list[Trajectory] = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            for task in batch:
                traj = self._run_single(skill_path, task)
                trajectories.append(traj)
                time.sleep(0.3)

        return trajectories

    def _run_single(self, skill_path: str, task: str) -> Trajectory:
        """Execute one task.  Mode-dependent dispatch."""
        if self.mode == "manual":
            return self._run_manual(task)
        elif self.mode == "script":
            return self._run_script(skill_path, task)
        elif self.mode == "hermes_auto":
            return self._run_hermes_subagent(skill_path, task)
        else:
            raise ValueError(f"Unknown rollout mode: {self.mode}")

    def _run_manual(self, descriptor: str) -> Trajectory:
        """Manual mode: the descriptor is a path to a single trajectory JSON."""
        data = json.loads(Path(descriptor).read_text(encoding="utf-8"))
        traj = Trajectory.from_dict(data)
        return traj

    def _run_script(self, skill_path: str, task: str) -> Trajectory:
        """Script mode: run a command per task, substitute {skill_path}."""
        # Stage 1: substitute {skill_path} into the task template
        task_resolved = task.replace("{skill_path}", skill_path)
        task_resolved = task_resolved.replace("{skill_dir}", str(Path(skill_path).parent))

        try:
            # Use shlex.split to avoid shell=True (security, correctness)
            cmd_parts = shlex.split(task_resolved)
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout.strip()
            error = result.stderr.strip() if result.returncode != 0 else None
            score = self.scorer.score(task_resolved, output, error)

            return Trajectory(
                task_id=task_resolved[:60],
                task_input=task_resolved,
                output=output,
                score=score,
                error=error,
                metadata={
                    "return_code": result.returncode,
                    "skill_path": skill_path,
                },
            )
        except subprocess.TimeoutExpired:
            return Trajectory(
                task_id=task_resolved[:60],
                task_input=task_resolved,
                output="",
                score=0.0,
                error="timeout",
                metadata={"skill_path": skill_path},
            )
        except Exception as e:
            return Trajectory(
                task_id=task_resolved[:60],
                task_input=task_resolved,
                output="",
                score=0.0,
                error=str(e),
                metadata={"skill_path": skill_path},
            )

    def _run_hermes_subagent(self, skill_path: str, task: str) -> Trajectory:
        """Hermes subagent mode — evaluate skill by running score_skill.py.

        This is a lightweight functional evaluation that scores the candidate
        skill document's structural quality. For true agent-based evaluation,
        replace with a Hermes delegate_task call.
        """
        try:
            from .score_skill import score_skill as _score_fn
            result = _score_fn(skill_path, skip_name_check=True)
            score = result["overall_score"]
            output = json.dumps(result, indent=2)
            return Trajectory(
                task_id=f"score_{Path(skill_path).stem}",
                task_input=task,
                output=output,
                score=score,
                error=None,
                metadata={
                    "skill_path": skill_path,
                    "mode": "hermes_auto",
                    "n_checks": result.get("n_checks", 0),
                    "n_issues": result.get("n_issues", 0),
                },
            )
        except ImportError:
            # Fallback: run score_skill.py as subprocess
            try:
                cmd = [
                    sys.executable or "python3",
                    str(Path(__file__).parent / "score_skill.py"),
                    "--skill", skill_path,
                    "--skip-name-check",
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                data = json.loads(r.stdout) if r.stdout else {"overall_score": 0.0}
                score = data.get("overall_score", 0.0)
                return Trajectory(
                    task_id=f"score_{Path(skill_path).stem}",
                    task_input=task,
                    output=r.stdout,
                    score=score,
                    error=r.stderr if r.returncode != 0 else None,
                    metadata={"skill_path": skill_path, "mode": "hermes_auto"},
                )
            except Exception as e:
                return Trajectory(
                    task_id=f"score_{Path(skill_path).stem}",
                    task_input=task,
                    output="",
                    score=0.0,
                    error=str(e),
                    metadata={"skill_path": skill_path, "mode": "hermes_auto_fallback"},
                )

    # ── I/O helpers ────────────────────────────────────────────────

    @staticmethod
    def load_trajectories(path: str) -> list[Trajectory]:
        """Load a batch of trajectories from a JSON array file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return [Trajectory.from_dict(d) for d in raw]

    @staticmethod
    def save_trajectories(trajectories: list[Trajectory], path: str):
        """Save trajectories to a JSON file for later manual-mode use."""
        data = [t.to_dict() for t in trajectories]
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    h = RolloutHarness(mode="script", scorer=ExitCodeScorer())
    print("RolloutHarness (script mode) ready.")
