"""Hermes trajectory reward extraction utilities.

This module converts Hermes Runtime trajectory entries into SkillOpt
``Trajectory`` objects whose ``score`` is derived from runtime signals such as
``completed``, ``api_calls``, ``tool_stats`` and ``tool_error_counts``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .protocols import RewardResult, Trajectory


DEFAULT_WEIGHTS = {
    "completed": 0.50,
    "tool_success_rate": 0.20,
    "tool_efficiency": 0.10,
    "error_penalty": 0.10,
    "answer_judge": 0.10,
}


@dataclass
class RewardConfig:
    """Configuration for Hermes trajectory reward shaping."""

    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    max_api_calls: int = 8
    max_tool_calls: int = 12
    max_tool_calls_per_tool: int = 8
    high_failure_rate: float = 0.30
    default_answer_judge: float = 0.50



# ── Public API ────────────────────────────────────────────────────────


def extract_reward(entry: dict[str, Any], config: RewardConfig | None = None) -> RewardResult:
    """Extract a scalar reward from one Hermes trajectory entry.

    The function is deliberately tolerant of partially populated trajectory
    records. Missing runtime fields produce neutral component values and lower
    the reward confidence instead of failing conversion.
    """

    config = config or RewardConfig()
    hermes = _extract_hermes_fields(entry)
    tool_stats = _normalise_tool_stats(hermes.get("tool_stats", {}))
    tool_error_counts = _normalise_error_counts(hermes.get("tool_error_counts", {}))

    completed_present = "completed" in hermes
    completed = _as_float_bool(hermes.get("completed", False))

    total_tool_calls = sum(s["count"] for s in tool_stats.values())
    total_success = sum(s["success"] for s in tool_stats.values())
    total_failure = sum(s["failure"] for s in tool_stats.values())
    total_named_errors = sum(tool_error_counts.values())
    total_errors = max(total_failure, total_named_errors)

    confidence_notes: list[str] = []
    if not completed_present:
        confidence_notes.append("missing_completed")
    if not tool_stats:
        confidence_notes.append("missing_tool_stats")

    if total_tool_calls > 0:
        tool_success_rate = _clamp(total_success / total_tool_calls)
        error_penalty = _clamp(1.0 - (total_errors / total_tool_calls))
    else:
        # Neutral-to-positive defaults: no tools can be correct for some tasks,
        # but lack of tool_stats means lower confidence.
        tool_success_rate = 1.0 if completed else 0.5
        error_penalty = 1.0 if completed else 0.5

    api_calls = _safe_int(hermes.get("api_calls"), default=None)
    tool_efficiency = _tool_efficiency(
        api_calls=api_calls,
        total_tool_calls=total_tool_calls if tool_stats else None,
        config=config,
    )
    if api_calls is None:
        confidence_notes.append("missing_api_calls")

    answer_judge = _extract_answer_judge(entry, config.default_answer_judge)
    if answer_judge == config.default_answer_judge:
        confidence_notes.append("default_answer_judge")

    components = {
        "completed": completed,
        "tool_success_rate": tool_success_rate,
        "tool_efficiency": tool_efficiency,
        "error_penalty": error_penalty,
        "answer_judge": answer_judge,
    }
    score = _weighted_score(components, config.weights)
    diagnostics = _diagnose_tool_policy(
        tool_stats=tool_stats,
        tool_error_counts=tool_error_counts,
        api_calls=api_calls,
        config=config,
    )
    if confidence_notes:
        diagnostics["confidence_notes"] = confidence_notes

    confidence = "high"
    if len(confidence_notes) >= 3:
        confidence = "low"
    elif confidence_notes:
        confidence = "medium"

    return RewardResult(
        score=round(score, 4),
        components={k: round(v, 4) for k, v in components.items()},
        diagnostics=diagnostics,
        confidence=confidence,
    )


def hermes_entry_to_trajectory(
    entry: dict[str, Any], config: RewardConfig | None = None
) -> Trajectory:
    """Convert one Hermes trajectory entry into a SkillOpt ``Trajectory``."""

    config = config or RewardConfig()
    reward = extract_reward(entry, config)
    hermes = _extract_hermes_fields(entry)

    metadata = dict(entry.get("metadata") or {})
    metadata.update(
        {
            "reward_source": "hermes_trajectory",
            "reward_components": reward.components,
            "reward_confidence": reward.confidence,
            "tool_policy_diagnostics": reward.diagnostics,
            "hermes": {
                "completed": hermes.get("completed"),
                "api_calls": hermes.get("api_calls"),
                "toolsets_used": hermes.get("toolsets_used", []),
                "tool_stats": hermes.get("tool_stats", {}),
                "tool_error_counts": hermes.get("tool_error_counts", {}),
            },
        }
    )

    return Trajectory(
        task_id=str(
            entry.get("task_id")
            or entry.get("id")
            or entry.get("trajectory_id")
            or _stable_task_id(entry)
        ),
        task_input=str(
            entry.get("task_input")
            or entry.get("input")
            or entry.get("prompt")
            or entry.get("task")
            or ""
        ),
        messages=list(entry.get("messages") or []),
        tool_calls=list(entry.get("tool_calls") or entry.get("tool_calls_made") or []),
        output=str(entry.get("output") or entry.get("final_output") or entry.get("answer") or ""),
        score=reward.score,
        error=entry.get("error") or entry.get("failure_reason"),
        metadata=metadata,
    )


def load_hermes_jsonl(
    paths: str | Path | Iterable[str | Path], config: RewardConfig | None = None
) -> list[Trajectory]:
    """Load Hermes JSONL/JSON files and return reward-scored trajectories."""

    config = config or RewardConfig()
    if isinstance(paths, (str, Path)):
        paths = [paths]

    trajectories: list[Trajectory] = []
    for path_like in paths:
        path = Path(path_like)
        for entry in _load_entries(path):
            if isinstance(entry, dict):
                trajectories.append(hermes_entry_to_trajectory(entry, config))
    return trajectories


def convert_hermes_jsonl(
    input_paths: Iterable[str | Path], output_path: str | Path, config: RewardConfig | None = None
) -> list[Trajectory]:
    """Convert Hermes trajectory files into a SkillOpt JSON trajectory array."""

    trajectories = load_hermes_jsonl(input_paths, config=config)
    payload = [t.to_dict() for t in trajectories]
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return trajectories


def parse_weights(raw: str | None) -> dict[str, float]:
    """Parse ``name=value`` comma-separated reward weights."""

    weights = DEFAULT_WEIGHTS.copy()
    if not raw:
        return weights
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight item {item!r}; expected name=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in DEFAULT_WEIGHTS:
            raise ValueError(f"Unknown reward component weight: {key}")
        weights[key] = float(value)
    return weights


# ── Internal helpers ──────────────────────────────────────────────────


def _extract_hermes_fields(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    nested = metadata.get("hermes") if isinstance(metadata.get("hermes"), dict) else {}
    hermes = dict(nested)
    for key in ("completed", "api_calls", "toolsets_used", "tool_stats", "tool_error_counts"):
        if key in entry:
            hermes[key] = entry[key]
        elif key in metadata and key not in hermes:
            hermes[key] = metadata[key]
    return hermes


def _normalise_tool_stats(raw: Any) -> dict[str, dict[str, int]]:
    if not isinstance(raw, dict):
        return {}
    normalised: dict[str, dict[str, int]] = {}
    for tool, stats in raw.items():
        if not isinstance(stats, dict):
            continue
        count = _safe_int(stats.get("count"), 0)
        success = _safe_int(stats.get("success"), 0)
        failure = _safe_int(stats.get("failure"), 0)
        if count <= 0:
            count = success + failure
        if count <= 0:
            continue
        if success == 0 and failure == 0:
            success = max(count - failure, 0)
        normalised[str(tool)] = {
            "count": max(count, 0),
            "success": max(success, 0),
            "failure": max(failure, 0),
        }
    return normalised


def _normalise_error_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): max(_safe_int(v, 0), 0) for k, v in raw.items()}


def _tool_efficiency(api_calls: int | None, total_tool_calls: int | None, config: RewardConfig) -> float:
    parts: list[float] = []
    if api_calls is not None:
        parts.append(_budget_score(api_calls, config.max_api_calls))
    if total_tool_calls is not None:
        parts.append(_budget_score(total_tool_calls, config.max_tool_calls))
    if not parts:
        return 0.5
    return _clamp(sum(parts) / len(parts))


def _budget_score(value: int, budget: int) -> float:
    if budget <= 0:
        return 1.0
    if value <= budget:
        return 1.0
    return _clamp(1.0 - ((value - budget) / max(budget, 1)))


def _extract_answer_judge(entry: dict[str, Any], default: float) -> float:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    for source in (entry, metadata):
        for key in ("answer_judge", "answer_score", "judge_score"):
            if key in source:
                return _clamp(_safe_float(source[key], default))
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    if "score" in entry and metadata.get("reward_source") != "hermes_trajectory":
        return _clamp(_safe_float(entry["score"], default))
    return _clamp(default)


def _weighted_score(components: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(max(v, 0.0) for v in weights.values())
    if total_weight <= 0:
        return 0.0
    score = sum(components.get(k, 0.0) * max(w, 0.0) for k, w in weights.items()) / total_weight
    return _clamp(score)


def _diagnose_tool_policy(
    *,
    tool_stats: dict[str, dict[str, int]],
    tool_error_counts: dict[str, int],
    api_calls: int | None,
    config: RewardConfig,
) -> dict[str, Any]:
    high_failure_tools: list[str] = []
    overused_tools: list[str] = []
    for tool, stats in tool_stats.items():
        count = stats["count"]
        failure_rate = stats["failure"] / count if count else 0.0
        if failure_rate >= config.high_failure_rate and stats["failure"] > 0:
            high_failure_tools.append(tool)
        if count > config.max_tool_calls_per_tool:
            overused_tools.append(tool)

    suggestions: list[str] = []
    if high_failure_tools:
        suggestions.append(
            "Constrain or add preconditions for high-failure tools: "
            + ", ".join(high_failure_tools)
        )
    if overused_tools:
        suggestions.append(
            "Reduce repeated calls or add early-stop checks for overused tools: "
            + ", ".join(overused_tools)
        )
    if api_calls is not None and api_calls > config.max_api_calls:
        suggestions.append("Prefer cheaper inspection steps before additional model/API calls.")

    return {
        "high_failure_tools": high_failure_tools,
        "overused_tools": overused_tools,
        "tool_error_counts": tool_error_counts,
        "api_call_budget_exceeded": api_calls is not None and api_calls > config.max_api_calls,
        "suggestion": " ".join(suggestions) if suggestions else "No obvious tool-policy issue detected.",
    }


def _load_entries(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("trajectories", "samples", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return []


def _stable_task_id(entry: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
    return "hermes_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _as_float_bool(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"true", "yes", "1", "completed", "success"} else 0.0
    return 0.0


def _safe_int(value: Any, default: int | None = 0) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ── Standalone CLI ────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Hermes trajectories into SkillOpt reward trajectories.")
    parser.add_argument("--input", nargs="+", required=True, help="Hermes JSONL/JSON trajectory files")
    parser.add_argument("--output", required=True, help="Output SkillOpt JSON trajectory array")
    parser.add_argument("--weights", help="Comma-separated reward weights, e.g. completed=0.5,tool_success_rate=0.2")
    parser.add_argument("--max-api-calls", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    args = parser.parse_args(argv)

    config = RewardConfig(
        weights=parse_weights(args.weights),
        max_api_calls=args.max_api_calls,
        max_tool_calls=args.max_tool_calls,
    )
    trajectories = convert_hermes_jsonl(args.input, args.output, config=config)
    print(f"Converted {len(trajectories)} trajectories to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
