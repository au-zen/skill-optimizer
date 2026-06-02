"""SkillOpt engine — optimisation loop for Hermes agent skill documents."""

from .protocols import Trajectory, Edit, EditBatch, OptimizationState, OptimizerConfig
from .engine import SkillOptEngine, _compute_text_similarity, _find_section
from .rollout import RolloutHarness
from .llm_client import OptimizerLLMClient

try:
    from .score_skill import score_skill, check_stability
except ImportError:
    score_skill = None
    check_stability = None

__all__ = [
    "Trajectory", "Edit", "EditBatch", "OptimizationState", "OptimizerConfig",
    "SkillOptEngine", "RolloutHarness", "OptimizerLLMClient",
    "score_skill", "check_stability",
    "_compute_text_similarity", "_find_section",
]
