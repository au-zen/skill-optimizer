"""SkillOpt engine — optimization loop for Hermes agent skill documents."""

from .protocols import Trajectory, Edit, EditBatch, OptimizationState, OptimizerConfig
from .engine import SkillOptEngine, _compute_text_similarity, _find_section
from .rollout import RolloutHarness
from .llm_client import OptimizerLLMClient
from .score_skill import score_skill, check_stability

__all__ = [
    "Trajectory", "Edit", "EditBatch", "OptimizationState", "OptimizerConfig",
    "SkillOptEngine", "RolloutHarness", "OptimizerLLMClient",
    "score_skill", "check_stability",
]
