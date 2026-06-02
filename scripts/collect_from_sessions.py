#!/usr/bin/env python3
"""
collect_from_sessions.py — Extract SkillOpt trajectories from ~/.hermes/state.db

Mines real Hermes Agent session history to build training data for skill-optimizer.
Unlike auto_collect_trajs.py (static file checks), this captures true agent behavior:
tool call sequences, skill loading patterns, and outcome signals from actual usage.

Usage:
    uv run python scripts/collect_from_sessions.py --skill skill-optimizer
    uv run python scripts/collect_from_sessions.py --skill skill-optimizer --days 30
    uv run python scripts/collect_from_sessions.py --skill skill-optimizer --dry-run
    uv run python scripts/collect_from_sessions.py --list-sessions skill-optimizer
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path.home() / ".hermes" / "state.db"
BASE = Path(__file__).resolve().parent.parent
SCHEDULER_DATA = BASE / "scheduler_data"


# ── Scoring weights ────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    "behavior": 0.30,   # tool call chain quality
    "outcome":  0.50,   # result signals (patch success, no revert, re-use)
    "context":  0.20,   # session metadata quality
}


# ── Session behavior signals ───────────────────────────────────────────

@dataclass
class SessionSignals:
    """Extracted behavioral signals from one session."""
    session_id: str
    source: str
    model: str
    started_at: float
    ended_at: float | None
    end_reason: str | None
    title: str | None

    # Behavior layer
    skill_views: list[str] = field(default_factory=list)       # skill names viewed
    skill_patches: list[dict] = field(default_factory=list)    # skill_manage patch calls
    skill_views_succeeded: int = 0
    skill_patches_succeeded: int = 0
    task_completed: bool = False

    # Messages
    messages: list[dict] = field(default_factory=list)
    tool_calls_raw: list[dict] = field(default_factory=list)

    # Outcome layer
    session_ended_cleanly: bool = False
    patch_subsequently_reverted: bool = False
    skill_reused_in_later_session: bool = False


def _open_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Hermes state.db not found at {db_path}. "
            "Is Hermes Agent installed and has been used at least once?"
        )
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _find_skill_sessions(
    con: sqlite3.Connection,
    skill_name: str,
    since_ts: float = 0.0,
) -> list[str]:
    """Return session IDs that contain skill_view calls for skill_name."""
    try:
        # Use FTS5 trigram index for substring search
        rows = con.execute("""
            SELECT DISTINCT m.session_id
            FROM messages m
            WHERE m.tool_name = 'skill_view'
              AND m.content LIKE ?
              AND m.timestamp >= ?
        """, (f'%"{skill_name}"%', since_ts)).fetchall()
        return [r["session_id"] for r in rows]
    except sqlite3.OperationalError:
        # Fallback without tool_name column (older schema)
        rows = con.execute("""
            SELECT DISTINCT session_id
            FROM messages
            WHERE content LIKE ?
              AND timestamp >= ?
        """, (f'%skill_view%{skill_name}%', since_ts)).fetchall()
        return [r["session_id"] for r in rows]


def _extract_signals(con: sqlite3.Connection, session_id: str, skill_name: str) -> SessionSignals:
    """Parse one session's messages into behavioral signals."""
    # Session metadata
    sess = con.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if not sess:
        return None

    signals = SessionSignals(
        session_id=session_id,
        source=sess["source"] or "unknown",
        model=sess["model"] or "unknown",
        started_at=sess["started_at"],
        ended_at=sess["ended_at"],
        end_reason=sess["end_reason"],
        title=sess["title"],
        session_ended_cleanly=sess["end_reason"] in ("user_exit", "completed", "stop"),
    )

    # Messages
    msgs = con.execute("""
        SELECT role, content, tool_calls, tool_name, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY timestamp
    """, (session_id,)).fetchall()

    for msg in msgs:
        role = msg["role"]
        content = msg["content"] or ""
        tool_name = msg["tool_name"] or ""

        signals.messages.append({
            "role": role,
            "content": content[:500],  # truncate for storage
        })

        # Parse tool calls
        raw_tc = msg["tool_calls"]
        if raw_tc:
            try:
                tc_list = json.loads(raw_tc)
                for tc in (tc_list if isinstance(tc_list, list) else [tc_list]):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args_raw = fn.get("arguments", "{}")
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except json.JSONDecodeError:
                        fn_args = {}

                    signals.tool_calls_raw.append({
                        "name": fn_name,
                        "args": fn_args,
                        "timestamp": msg["timestamp"],
                    })

                    if fn_name == "skill_view":
                        viewed = fn_args.get("name", "")
                        if viewed:
                            signals.skill_views.append(viewed)
                        if skill_name in str(fn_args):
                            signals.skill_views_succeeded += 1

                    elif fn_name == "skill_manage":
                        action = fn_args.get("action", "")
                        if action == "patch" and skill_name in str(fn_args):
                            signals.skill_patches.append(fn_args)
                            signals.skill_patches_succeeded += 1

            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # Simple heuristic: task completion signal from content
        if role == "assistant" and any(
            kw in content.lower()
            for kw in ["optimization complete", "best_skill", "accepted edits",
                       "skill updated", "优化完成", "patch applied"]
        ):
            signals.task_completed = True

    return signals


def _score_session(signals: SessionSignals, skill_name: str) -> tuple[float, dict]:
    """
    Three-layer scoring.

    Layer 1 (behavior, 30%): Did the agent use skill tools correctly?
    Layer 2 (outcome, 50%):  Did the task reach a meaningful result?
    Layer 3 (context, 20%):  Session quality signals.

    Returns (score, details_dict).
    """
    details = {}

    # ── Layer 1: Behavior ──────────────────────────────────────────────
    b_score = 0.0

    # skill_view was called for this skill
    skill_viewed = skill_name in signals.skill_views
    b_score += 0.4 if skill_viewed else 0.0
    details["skill_viewed"] = skill_viewed

    # skill_view was productive (appeared in calls, not just mentioned in content)
    b_score += 0.3 if signals.skill_views_succeeded > 0 else 0.0
    details["skill_views_succeeded"] = signals.skill_views_succeeded

    # patch was attempted
    patch_attempted = signals.skill_patches_succeeded > 0
    b_score += 0.3 if patch_attempted else 0.0
    details["patch_attempted"] = patch_attempted

    b_score = min(1.0, b_score)
    details["behavior_score"] = round(b_score, 3)

    # ── Layer 2: Outcome ───────────────────────────────────────────────
    o_score = 0.0

    # Task completion signal
    o_score += 0.4 if signals.task_completed else 0.0
    details["task_completed"] = signals.task_completed

    # Session ended cleanly (not timeout, not error)
    o_score += 0.3 if signals.session_ended_cleanly else 0.0
    details["session_ended_cleanly"] = signals.session_ended_cleanly

    # Patch was subsequently reverted (negative signal)
    if signals.patch_subsequently_reverted:
        o_score -= 0.3
    details["patch_reverted"] = signals.patch_subsequently_reverted

    # Skill reused in a later session (positive signal)
    o_score += 0.3 if signals.skill_reused_in_later_session else 0.0
    details["skill_reused"] = signals.skill_reused_in_later_session

    o_score = max(0.0, min(1.0, o_score))
    details["outcome_score"] = round(o_score, 3)

    # ── Layer 3: Context ───────────────────────────────────────────────
    c_score = 0.0

    # CLI sessions are higher quality than gateway (more deliberate)
    c_score += 0.5 if signals.source == "cli" else 0.3
    details["source"] = signals.source

    # Has a title (suggests deliberate, named task)
    c_score += 0.3 if signals.title else 0.0
    details["has_title"] = bool(signals.title)

    # Has reasonable message count (not a one-liner, not a giant dump)
    msg_count = len(signals.messages)
    if 5 <= msg_count <= 60:
        c_score += 0.2
    elif msg_count < 5:
        c_score += 0.1
    details["message_count"] = msg_count

    c_score = min(1.0, c_score)
    details["context_score"] = round(c_score, 3)

    # ── Weighted total ─────────────────────────────────────────────────
    total = (
        SCORE_WEIGHTS["behavior"] * b_score
        + SCORE_WEIGHTS["outcome"] * o_score
        + SCORE_WEIGHTS["context"] * c_score
    )
    total = round(max(0.0, min(1.0, total)), 4)
    details["total_score"] = total

    return total, details


def _signals_to_trajectory(signals: SessionSignals, skill_name: str, score: float, details: dict) -> dict:
    """Pack SessionSignals into Trajectory-compatible dict."""
    # Reconstruct a meaningful task_input from session title or first user message
    first_user = next(
        (m["content"] for m in signals.messages if m["role"] == "user"),
        f"Hermes session using skill: {skill_name}"
    )
    task_input = signals.title or first_user[:200]

    # Reconstruct output from last assistant message
    last_assistant = next(
        (m["content"] for m in reversed(signals.messages) if m["role"] == "assistant"),
        ""
    )

    return {
        "task_id": f"session_{signals.session_id[:16]}",
        "task_input": task_input,
        "messages": signals.messages,
        "tool_calls": [
            {
                "tool": tc["name"],
                "params": tc.get("args", {}),
                "result_present": True,
                "timestamp": tc.get("timestamp"),
            }
            for tc in signals.tool_calls_raw
        ],
        "output": last_assistant[:1000],
        "score": score,
        "error": None if score > 0.3 else "low_quality_session",
        "metadata": {
            "source": "hermes-session-db",
            "session_id": signals.session_id,
            "session_source": signals.source,
            "model": signals.model,
            "started_at": signals.started_at,
            "skill_views": signals.skill_views,
            "patches_applied": signals.skill_patches_succeeded,
            "score_details": details,
        },
    }


def _check_reuse(con: sqlite3.Connection, session_id: str, skill_name: str, after_ts: float) -> bool:
    """Check if skill_name was loaded in any session after after_ts (reuse signal)."""
    rows = con.execute("""
        SELECT COUNT(*) as cnt
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
        WHERE m.tool_name = 'skill_view'
          AND m.content LIKE ?
          AND s.started_at > ?
          AND m.session_id != ?
    """, (f'%"{skill_name}"%', after_ts, session_id)).fetchone()
    return (rows["cnt"] or 0) > 0


def collect(
    skill_name: str,
    days: int = 90,
    min_score: float = 0.1,
    dry_run: bool = False,
    db_path: Path = DB_PATH,
) -> dict:
    """Main collection entry point."""
    con = _open_db(db_path)
    since_ts = time.time() - days * 86400

    print(f"[collect_from_sessions] skill={skill_name}, days={days}, db={db_path}")

    session_ids = _find_skill_sessions(con, skill_name, since_ts)
    print(f"  Found {len(session_ids)} sessions containing skill_view({skill_name!r})")

    if not session_ids:
        con.close()
        return {
            "success": True,
            "trajectories": 0,
            "message": "No sessions found. Run Hermes Agent with this skill to build history.",
        }

    trajs = []
    skipped = 0

    for sid in session_ids:
        signals = _extract_signals(con, sid, skill_name)
        if not signals:
            skipped += 1
            continue

        # Enrich: check if skill was reused after this session
        if signals.ended_at:
            signals.skill_reused_in_later_session = _check_reuse(
                con, sid, skill_name, signals.ended_at
            )

        score, details = _score_session(signals, skill_name)
        if score < min_score:
            skipped += 1
            continue

        traj = _signals_to_trajectory(signals, skill_name, score, details)
        trajs.append(traj)

    con.close()

    if not trajs:
        return {
            "success": True,
            "trajectories": 0,
            "skipped": skipped,
            "message": f"All {skipped} sessions scored below min_score={min_score}.",
        }

    avg_score = sum(t["score"] for t in trajs) / len(trajs)
    print(f"  Collected {len(trajs)} trajectories (skipped {skipped}), avg_score={avg_score:.3f}")

    if dry_run:
        return {"success": True, "trajectories": len(trajs), "skipped": skipped,
                "avg_score": avg_score, "dry_run": True, "data": trajs}

    # Save
    data_dir = SCHEDULER_DATA / skill_name
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "session_trajs.json"
    out_path.write_text(json.dumps(trajs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved → {out_path}")

    return {
        "success": True,
        "trajectories": len(trajs),
        "skipped": skipped,
        "avg_score": avg_score,
        "output": str(out_path),
    }


def list_sessions(skill_name: str, days: int = 90, db_path: Path = DB_PATH) -> None:
    """Print a summary of sessions relevant to skill_name."""
    con = _open_db(db_path)
    since_ts = time.time() - days * 86400
    session_ids = _find_skill_sessions(con, skill_name, since_ts)

    if not session_ids:
        print(f"No sessions found for skill={skill_name!r} in the last {days} days.")
        con.close()
        return

    print(f"{'Session ID':<26} {'Source':<10} {'Title':<30} {'Score':<7} {'Msgs':<5} {'Patches'}")
    print("-" * 90)
    for sid in session_ids:
        signals = _extract_signals(con, sid, skill_name)
        if not signals:
            continue
        score, _ = _score_session(signals, skill_name)
        title = (signals.title or "")[:28]
        print(f"{sid[:24]:<26} {signals.source:<10} {title:<30} {score:<7.3f} "
              f"{len(signals.messages):<5} {signals.skill_patches_succeeded}")
    con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract SkillOpt trajectories from Hermes session history"
    )
    parser.add_argument("--skill", required=True, help="Skill name to mine sessions for")
    parser.add_argument("--days", type=int, default=90, help="Look back N days (default 90)")
    parser.add_argument("--min-score", type=float, default=0.1,
                        help="Discard sessions scoring below this (default 0.1)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--list-sessions", action="store_true",
                        help="List matching sessions with scores")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to state.db")
    args = parser.parse_args()

    try:
        if args.list_sessions:
            list_sessions(args.skill, days=args.days, db_path=args.db)
            return

        result = collect(
            skill_name=args.skill,
            days=args.days,
            min_score=args.min_score,
            dry_run=args.dry_run,
            db_path=args.db,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\n{'DRY RUN — ' if result.get('dry_run') else ''}Result:")
    for k, v in result.items():
        if k != "data":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
