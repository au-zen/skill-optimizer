"""
SkillOpt Data Protocols — All flow types in the optimization loop.

These dataclasses define the contract between components:
  Trajectory        — Single execution trace from target agent
  Edit              — Single bounded text operation on SKILL.md
  EditBatch         — Budget-limited set of edits for one step
  OptimizationState — Persistent state across steps/epochs
  OptimizerConfig   — All tunable hyperparameters
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── 1. Trajectory ──────────────────────────────────────────────────────


@dataclass
class Trajectory:
    """A single execution trace from the target agent on one task.

    Fields
    ------
    task_id    : Unique identifier for the task.
    task_input : The prompt / instructions given to the target agent.
    messages   : Full conversation (list of dicts with role/content).
    tool_calls : All tool invocations during the run.
    output     : Final output produced by the agent.
    score      : Scalar reward in [0, 1].
    error      : Optional error description (None if successful).
    metadata   : Extensible bag of extra info (duration, token counts …).
    """

    task_id: str
    task_input: str
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    output: str = ""
    score: float = 0.0
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    is_success_threshold: float = 0.5

    @property
    def is_success(self) -> bool:
        """Heuristic: score >= threshold is considered a success."""
        return self.score >= self.is_success_threshold

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_input": self.task_input,
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "output": self.output,
            "score": self.score,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trajectory:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})




@dataclass
class RewardResult:
    """Auditable reward extracted from an execution trajectory.

    ``Trajectory.score`` remains the scalar used by the engine, while these
    fields keep the reward components and tool-policy diagnostics available
    for reflection prompts and validation metrics.
    """

    score: float
    components: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    confidence: str = "high"

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "components": self.components,
            "diagnostics": self.diagnostics,
            "confidence": self.confidence,
        }


# ── 2. Edit operations ────────────────────────────────────────────────


@dataclass
class Edit:
    """A single bounded text edit on the skill document.

    Fields
    ------
    operation     : "add" | "delete" | "replace"
    section       : Target section heading in the skill doc (e.g. "## Rules").
    old_content   : For 'delete'/'replace' — the exact text to remove/replace.
    new_content   : For 'add'/'replace' — the new text to insert.
    rationale     : Why this edit is expected to help.
    expected_gain : Estimated impact (0–1 scale).
    """

    operation: str  # "add" | "delete" | "replace"
    section: str
    old_content: str = ""
    new_content: str = ""
    rationale: str = ""
    expected_gain: float = 0.0

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "section": self.section,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "rationale": self.rationale,
            "expected_gain": self.expected_gain,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Edit:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── 3. Batch ──────────────────────────────────────────────────────────


@dataclass
class EditBatch:
    """A budget-limited set of edits proposed for one optimization step."""

    edits: list[Edit] = field(default_factory=list)
    budget_used: int = 0
    budget_total: int = 0
    epoch: int = 0
    step: int = 0

    @property
    def is_full(self) -> bool:
        return self.budget_used >= self.budget_total

    def to_dict(self) -> dict:
        return {
            "edits": [e.to_dict() for e in self.edits],
            "budget_used": self.budget_used,
            "budget_total": self.budget_total,
            "epoch": self.epoch,
            "step": self.step,
        }


# ── 4. Persistent state ───────────────────────────────────────────────


@dataclass
class OptimizationState:
    """Persistent state that survives across steps and is saved to disk."""

    current_skill_path: str = ""
    current_score: float = 0.0
    best_skill_path: str = ""
    best_score: float = 0.0

    epoch: int = 0
    step: int = 0

    # Edit history
    accepted_edits: list[dict] = field(default_factory=list)
    rejected_buffer: list[dict] = field(default_factory=list)

    # Slow / meta fields
    slow_field_content: str = ""
    meta_skill_content: str = ""

    # Skill hashes for change tracking
    skill_hash: str = ""

    # Stage 2: origin anchor (set once at run() startup, never updated)
    origin_skill_path: str = ""
    origin_skill_hash: str = ""

    # Stage 2: drift detection
    drift_detected: bool = False
    drift_count: int = 0

    # Stage 3: holdout tracking
    holdout_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "current_skill_path": self.current_skill_path,
            "current_score": self.current_score,
            "best_skill_path": self.best_skill_path,
            "best_score": self.best_score,
            "epoch": self.epoch,
            "step": self.step,
            "accepted_edits": self.accepted_edits,
            "rejected_buffer": self.rejected_buffer,
            "slow_field_content": self.slow_field_content,
            "meta_skill_content": self.meta_skill_content,
            "skill_hash": self.skill_hash,
            "origin_skill_path": self.origin_skill_path,
            "origin_skill_hash": self.origin_skill_hash,
            "drift_detected": self.drift_detected,
            "drift_count": self.drift_count,
            "holdout_score": self.holdout_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OptimizationState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def compute_skill_hash(self, skill_path: str) -> str:
        """SHA-256 of skill file content for change tracking."""
        content = Path(skill_path).read_text(encoding="utf-8")
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── 5. Config ─────────────────────────────────────────────────────────


@dataclass
class OptimizerConfig:
    """All tunable hyperparameters for SkillOpt engine."""

    # ── Rollout ──
    rollout_batch_size: int = 8
    reflection_minibatch_size: int = 4
    score_threshold: float = 0.5

    # ── Edit budget ──
    edit_budget_init: int = 5
    edit_budget_final: int = 1
    schedule: str = "cosine"  # "cosine" | "constant" | "linear"

    # ── Loop control ──
    max_epochs: int = 4
    max_steps_per_epoch: int = 4

    # ── Validation ──
    accept_strict: bool = True  # Only accept if score strictly improves
    min_reward_delta: float = 0.0
    completed_rate_tolerance: float = 0.0
    tool_failure_rate_tolerance: float = 0.05

    # ── Rejected buffer ──
    rejected_buffer_size: int = 20

    # ── Drift detection (Stage 2) ──
    min_similarity: float = 0.6  # minimum similarity to origin to accept edits

    # ── Robustness (Stage 3) ──
    use_secondary_scorer: bool = False
    holdout_rollout: bool = False

    # ── Optimizer LLM ──
    optimizer_model: str = "openai/gpt-4o-2024-11-20"  # or "anthropic/claude-sonnet-4"
    optimizer_api_base: str = "https://openrouter.ai/api/v1"

    # ── Paths ──
    work_dir: str = "./skillopt_work"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> OptimizerConfig:
        valid = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})
