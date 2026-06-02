#!/usr/bin/env python3
"""
merge_trajs.py — Merge static + session trajectories into a unified training pool.

Implements the Experience Replay Buffer pattern from SkillOpt:
  static_trajs  (auto_collect_trajs.py)  — document compliance, structure checks
  session_trajs (collect_from_sessions.py) — real agent behavior, outcome signals

The merger deduplicates, re-weights by source, enforces pool size limits,
and produces train.json + sel.json ready for engine.py.

Usage:
    uv run python scripts/merge_trajs.py --skill skill-optimizer
    uv run python scripts/merge_trajs.py --skill skill-optimizer --dry-run
    uv run python scripts/merge_trajs.py --skill skill-optimizer --pool-size 30
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SCHEDULER_DATA = BASE / "scheduler_data"

# Source weight multipliers applied to raw scores before merging.
# session_trajs carry behavior + outcome signals → higher weight.
# static_trajs are pure document checks → lower weight but still useful for cold start.
SOURCE_WEIGHTS: dict[str, float] = {
    "hermes-session-db": 1.0,   # real behavior
    "auto-collect-v1":   0.6,   # document compliance
    "manual":            0.8,   # hand-labeled
}

DEFAULT_POOL_SIZE = 30     # max trajectories in merged pool
DEFAULT_SEL_RATIO = 0.20   # fraction reserved for sel (validation)
DEFAULT_SEL_MIN   = 2      # minimum sel size regardless of ratio


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _source_tag(traj: dict) -> str:
    """Extract the source tag from trajectory metadata."""
    return traj.get("metadata", {}).get("source", "auto-collect-v1")


def _weighted_score(traj: dict) -> float:
    """Apply source weight to raw score."""
    raw = traj.get("score", 0.0)
    w = SOURCE_WEIGHTS.get(_source_tag(traj), 0.5)
    return min(1.0, raw * w)


def _dedup(trajs: list[dict]) -> list[dict]:
    """Deduplicate by task_id, keeping the highest weighted score."""
    best: dict[str, dict] = {}
    for t in trajs:
        tid = t.get("task_id", "")
        if not tid:
            # No task_id — keep as-is with a synthetic ID
            t["task_id"] = f"anon_{id(t)}"
            best[t["task_id"]] = t
            continue
        existing = best.get(tid)
        if existing is None or _weighted_score(t) > _weighted_score(existing):
            best[tid] = t
    return list(best.values())


def _sel_split(
    trajs: list[dict],
    sel_ratio: float = DEFAULT_SEL_RATIO,
    sel_min: int = DEFAULT_SEL_MIN,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split into train + sel.

    Selection strategy:
    - sel should be diverse: include both high-score and low-score examples
    - sel must contain at least one session-sourced trajectory if available
    - train gets the rest
    """
    rng = random.Random(seed)

    n_sel = max(sel_min, int(len(trajs) * sel_ratio))
    n_sel = min(n_sel, len(trajs) - 1)  # always keep at least 1 in train

    sorted_trajs = sorted(trajs, key=_weighted_score)

    # Stratified pick: take from bottom and top of score distribution
    n_low  = n_sel // 2
    n_high = n_sel - n_low

    sel_candidates  = sorted_trajs[:n_low] + sorted_trajs[-n_high:]
    train_remainder = [t for t in sorted_trajs if t not in sel_candidates]

    # Ensure at least one session traj in sel if any exist in the pool
    session_trajs_in_sel = [t for t in sel_candidates if _source_tag(t) == "hermes-session-db"]
    if not session_trajs_in_sel:
        session_pool = [t for t in trajs if _source_tag(t) == "hermes-session-db"]
        if session_pool:
            inject = rng.choice(session_pool)
            if inject not in sel_candidates:
                sel_candidates.append(inject)
                if inject in train_remainder:
                    train_remainder.remove(inject)

    return train_remainder, sel_candidates


def merge(
    skill_name: str,
    pool_size: int = DEFAULT_POOL_SIZE,
    sel_ratio: float = DEFAULT_SEL_RATIO,
    dry_run: bool = False,
    data_dir: Path | None = None,
) -> dict:
    """Merge trajectory sources for one skill into train.json + sel.json."""
    data_dir = data_dir or (SCHEDULER_DATA / skill_name)

    # Load all available sources
    static_trajs  = _load_json(data_dir / "train.json") + _load_json(data_dir / "sel.json")
    session_trajs = _load_json(data_dir / "session_trajs.json")

    n_static  = len(static_trajs)
    n_session = len(session_trajs)

    print(f"[merge_trajs] skill={skill_name}")
    print(f"  static trajs:  {n_static}  (from auto_collect_trajs.py)")
    print(f"  session trajs: {n_session} (from collect_from_sessions.py)")

    if not static_trajs and not session_trajs:
        return {
            "success": False,
            "error": (
                f"No trajectory files found in {data_dir}. "
                "Run auto_collect_trajs.py and/or collect_from_sessions.py first."
            ),
        }

    # Merge and deduplicate
    all_trajs = _dedup(static_trajs + session_trajs)
    print(f"  after dedup:   {len(all_trajs)} unique trajectories")

    # Sort by weighted score descending, cap at pool_size
    all_trajs.sort(key=_weighted_score, reverse=True)
    if len(all_trajs) > pool_size:
        dropped = len(all_trajs) - pool_size
        all_trajs = all_trajs[:pool_size]
        print(f"  capped to {pool_size} (dropped {dropped} lowest-scoring)")

    if len(all_trajs) < 3:
        return {
            "success": False,
            "error": f"Only {len(all_trajs)} trajectories after merge. Need at least 3.",
        }

    # Split
    train, sel = _sel_split(all_trajs, sel_ratio=sel_ratio)
    print(f"  train: {len(train)}, sel: {len(sel)}")

    # Score summary
    avg_train = sum(_weighted_score(t) for t in train) / len(train) if train else 0
    avg_sel   = sum(_weighted_score(t) for t in sel) / len(sel) if sel else 0

    source_dist: dict[str, int] = {}
    for t in all_trajs:
        s = _source_tag(t)
        source_dist[s] = source_dist.get(s, 0) + 1

    summary = {
        "success": True,
        "total": len(all_trajs),
        "train": len(train),
        "sel": len(sel),
        "avg_train_score": round(avg_train, 4),
        "avg_sel_score": round(avg_sel, 4),
        "source_distribution": source_dist,
        "pool_size_cap": pool_size,
        "n_static_input": n_static,
        "n_session_input": n_session,
    }

    if dry_run:
        summary["dry_run"] = True
        return summary

    # Write output (overwrite the train/sel split used by engine.py)
    (data_dir / "train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (data_dir / "sel.json").write_text(
        json.dumps(sel, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Written → {data_dir}/train.json, sel.json")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Merge static + session trajectories into a unified SkillOpt training pool"
    )
    parser.add_argument("--skill", required=True, help="Skill name")
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE,
                        help=f"Max trajectories in merged pool (default {DEFAULT_POOL_SIZE})")
    parser.add_argument("--sel-ratio", type=float, default=DEFAULT_SEL_RATIO,
                        help=f"Fraction for sel split (default {DEFAULT_SEL_RATIO})")
    parser.add_argument("--dry-run", action="store_true", help="Show summary without writing")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override scheduler_data/<skill> path")
    args = parser.parse_args()

    result = merge(
        skill_name=args.skill,
        pool_size=args.pool_size,
        sel_ratio=args.sel_ratio,
        dry_run=args.dry_run,
        data_dir=args.data_dir,
    )

    if not result.get("success"):
        print(f"ERROR: {result.get('error')}")
        sys.exit(1)

    print(f"\n{'DRY RUN — ' if result.get('dry_run') else ''}Merge complete:")
    for k, v in result.items():
        if k not in ("success", "dry_run"):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
