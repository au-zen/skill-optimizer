#!/usr/bin/env python3
"""
SkillOpt Autonomous Scheduler — Hermes-native hybrid mode (脚本执行 + LLM 报告).

Architecture (Option C):
  - Scheduler script runs periodically via no_agent cron.
  - Checks state → decides if optimization needed → runs engine → updates state.
  - Empty stdout = no change (silent, no notification).
  - Structured JSON on stdout = optimization happened (cron delivers to user).

Usage
-----
  uv run python3 scripts/opt_scheduler.py [--check-only] [--force]

  --check-only  Only report what needs optimization, don't run.
  --force       Ignore interval checks, optimize all skills now.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────

SKILL_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = SKILL_DIR / "scripts" / "scheduler_config.json"
DEFAULT_STATE_PATH = SKILL_DIR / "scheduler_state.json"
ENGINE_CLI = ["uv", "run", "python3", "run.py"]


# ── State management ──────────────────────────────────────────────────


def load_state(path: Path) -> dict[str, Any]:
    """Load scheduler state from JSON file."""
    if not path.exists():
        return {"skills": {}, "global": {"last_run": None, "total_optimizations": 0}}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"skills": {}, "global": {"last_run": None, "total_optimizations": 0}}


def save_state(state: dict[str, Any], path: Path):
    """Persist scheduler state to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state["global"]["last_run"] = _now_iso()
    with open(path, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ── Config loading ────────────────────────────────────────────────────


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load scheduler config, merging with defaults."""
    path = path or DEFAULT_CONFIG_PATH

    defaults = {
        "interval_hours": 24,
        "default_epochs": 1,
        "default_steps": 2,
        "default_budget_init": 3,
        "default_budget_final": 1,
        "skills": [],
    }

    if not path.exists():
        return defaults

    try:
        with open(path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return defaults

    # Merge with defaults
    for k in defaults:
        cfg.setdefault(k, defaults[k])
    return cfg


# ── Helpers ───────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _skill_hash(skill_path: str) -> str:
    """Quick content hash for detecting skill changes."""
    p = Path(skill_path).expanduser().resolve()
    if not p.exists():
        return ""
    content = p.read_text(encoding="utf-8")
    return sha256(content.encode()).hexdigest()[:12]


def _hours_since(iso_str: str | None) -> float | None:
    """Return hours elapsed since an ISO timestamp, or None if never."""
    if not iso_str:
        return None
    try:
        then = datetime.fromisoformat(iso_str)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - then).total_seconds() / 3600
    except (ValueError, TypeError):
        return None


def _resolve(p: str) -> str:
    """Expand ~ and resolve to absolute path."""
    return str(Path(p).expanduser().resolve())


# ── Core logic ────────────────────────────────────────────────────────


def check_skill_needs_opt(
    skill_cfg: dict[str, Any],
    skill_state: dict[str, Any] | None,
    *,
    force: bool = False,
    default_interval: int = 24,
) -> tuple[bool, str]:
    """Determine if a skill needs optimization.

    Returns (needs_opt: bool, reason: str).
    """
    if force:
        return True, "forced"

    if skill_state is None:
        return True, "never optimized"

    last_opt = skill_state.get("last_optimized")
    if last_opt is None:
        return True, "never optimized"

    interval = skill_cfg.get("interval_hours", default_interval)
    hours = _hours_since(last_opt)
    if hours is None:
        return True, "invalid last_optimized timestamp"

    if hours >= interval:
        return True, f"{hours:.0f}h since last opt (interval={interval}h)"

    return False, f"only {hours:.0f}h elapsed (need {interval}h)"


def resolve_traj_path(path_str: str | None) -> str | None:
    """Resolve a trajectory file path, returning None if it doesn't exist."""
    if not path_str:
        return None
    p = Path(path_str).expanduser().resolve()
    return str(p) if p.exists() else None


def run_optimization(
    skill_cfg: dict[str, Any],
    workdir_base: Path,
) -> dict[str, Any]:
    """Run skill-optimizer CLI for one skill. Returns result summary."""
    name = skill_cfg["name"]
    skill_path = _resolve(skill_cfg["skill_path"])
    train_trajs = resolve_traj_path(skill_cfg.get("train_trajs"))
    sel_trajs = resolve_traj_path(skill_cfg.get("sel_trajs"))
    # Use a unique work dir per run (timestamped)
    work_dir = workdir_base / name / time.strftime("%Y%m%d_%H%M%S")
    work_dir.mkdir(parents=True, exist_ok=True)

    epochs = skill_cfg.get("epochs", 1)
    steps = skill_cfg.get("steps", 2)
    budget_init = skill_cfg.get("budget_init", 3)
    budget_final = skill_cfg.get("budget_final", 1)

    result: dict[str, Any] = {
        "skill": name,
        "skill_path": skill_path,
        "status": "skipped",
        "score": None,
        "best_score": None,
        "edits_accepted": 0,
        "edits_rejected": 0,
        "error": None,
    }

    # Check prerequisites
    if not Path(skill_path).exists():
        result["error"] = f"Skill not found: {skill_path}"
        result["status"] = "error"
        return result

    if not train_trajs:
        result["error"] = "No train trajectories configured or found"
        result["status"] = "skipped"
        return result

    # Build CLI command
    cmd = list(ENGINE_CLI) + [
        "optimize",
        "--skill", skill_path,
        "--train-trajs", train_trajs,
        "--work-dir", str(work_dir),
        "--epochs", str(epochs),
        "--steps", str(steps),
        "--budget-init", str(budget_init),
        "--budget-final", str(budget_final),
        "--hermes-default",
    ]
    if sel_trajs:
        cmd += ["--sel-trajs", sel_trajs]

    try:
        print(f"  [scheduler] Running optimization for '{name}'...", file=sys.stderr)
        started = time.time()

        proc = subprocess.run(
            cmd,
            cwd=str(SKILL_DIR),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per skill
        )

        elapsed = time.time() - started
        result["elapsed_seconds"] = round(elapsed, 1)

        if proc.returncode != 0:
            result["status"] = "error"
            result["error"] = proc.stderr.strip()[:500] or f"Exit code {proc.returncode}"
            result["cli_stdout"] = proc.stdout.strip()[-2000:]
            result["cli_stderr"] = proc.stderr.strip()[-2000:]
            return result

        # Parse results from CLI stdout
        cli_out = proc.stdout.strip()
        result["cli_stdout"] = cli_out[-2000:]

        # Extract key metrics from CLI output
        for line in cli_out.splitlines():
            if "Best score:" in line:
                try:
                    result["best_score"] = float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "Edits accepted:" in line:
                try:
                    result["edits_accepted"] = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "Edits rejected:" in line:
                try:
                    result["edits_rejected"] = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass

        # Read final state for score
        state_file = work_dir / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state_data = json.load(f)
                result["score"] = state_data.get("current_score")
                if result["best_score"] is None:
                    result["best_score"] = state_data.get("best_score")
            except (json.JSONDecodeError, OSError):
                pass

        # Read best_skill.md
        best_skill = work_dir / "best_skill.md"
        if best_skill.exists():
            result["best_skill_path"] = str(best_skill)

        result["status"] = "completed"
        if result["edits_accepted"] == 0:
            result["status"] = "no_improvement"

    except subprocess.TimeoutExpired:
        result["status"] = "error"
        result["error"] = "Optimization timed out after 600s"
    except FileNotFoundError as e:
        result["status"] = "error"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "error"
        result["error"] = f"Unexpected error: {e}"

    return result


def update_state_after_opt(
    state: dict[str, Any],
    skill_cfg: dict[str, Any],
    opt_result: dict[str, Any],
):
    """Update scheduler state after an optimization run."""
    name = skill_cfg["name"]
    skill_path = _resolve(skill_cfg["skill_path"])

    skill_state = state["skills"].setdefault(name, {})
    skill_state["skill_path"] = skill_path
    skill_state["last_optimized"] = _now_iso()

    if opt_result.get("score") is not None:
        skill_state["last_score"] = opt_result["score"]
        current_best = skill_state.get("best_score")
        if current_best is None or opt_result["score"] > current_best:
            skill_state["best_score"] = opt_result["score"]

    if opt_result.get("best_score") is not None:
        current_best = skill_state.get("best_score")
        if current_best is None or opt_result["best_score"] > current_best:
            skill_state["best_score"] = opt_result["best_score"]

    skill_state["optimization_count"] = skill_state.get("optimization_count", 0) + 1
    skill_state["skill_hash"] = _skill_hash(skill_path)
    skill_state["last_status"] = opt_result.get("status", "unknown")
    skill_state["edits_accepted"] = opt_result.get("edits_accepted", 0)
    skill_state["edits_rejected"] = opt_result.get("edits_rejected", 0)

    state["global"]["total_optimizations"] = (
        state["global"].get("total_optimizations", 0) + 1
    )


def format_report(
    results: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    """Format optimization results as a human-readable report."""
    now = _now_iso()
    lines: list[str] = []
    lines.append(f"# SkillOpt 自治调度报告 — {now}")
    lines.append("")

    completed = [r for r in results if r["status"] == "completed"]
    no_impr = [r for r in results if r["status"] == "no_improvement"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    if completed:
        lines.append(f"## ✅ 优化完成 ({len(completed)} skills)")
        lines.append("")
        for r in completed:
            lines.append(f"**{r['skill']}**: score {r['score']:.4f}, "
                         f"{r['edits_accepted']} accepted / {r['edits_rejected']} rejected")
            if r.get("best_skill_path"):
                lines.append(f"  — best_skill: `{r['best_skill_path']}`")
                lines.append(f"  — elapsed: {r.get('elapsed_seconds', '?')}s")
            lines.append("")

    if no_impr:
        lines.append(f"### ⏭ 无改进 ({len(no_impr)} skills)")
        for r in no_impr:
            lines.append(f"- {r['skill']}: no edits accepted")
        lines.append("")

    if errors:
        lines.append(f"### ❌ 错误 ({len(errors)} skills)")
        for r in errors:
            lines.append(f"- {r['skill']}: {r.get('error', 'unknown')}")
        lines.append("")

    if skipped:
        lines.append(f"### ⏸ 跳过 ({len(skipped)} skills)")
        for r in skipped:
            err = r.get("error") or "no trajectories available"
            lines.append(f"- {r['skill']}: {err}")
        lines.append("")

    # State summary
    lines.append("---")
    lines.append(f"下次检查: {config.get('interval_hours', 24)}h 后")

    # Add JSON version for machine consumption
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({
        "timestamp": now,
        "completed": len(completed),
        "errors": len(errors),
        "skipped": len(skipped),
        "no_improvement": len(no_impr),
        "details": [{
            "skill": r["skill"],
            "status": r["status"],
            "score": r.get("score"),
            "edits_accepted": r.get("edits_accepted", 0),
            "elapsed": r.get("elapsed_seconds"),
        } for r in results],
    }, indent=2, ensure_ascii=False))
    lines.append("```")

    return "\n".join(lines)


# ── Main scheduler loop ───────────────────────────────────────────────


def run_scheduler(
    *,
    check_only: bool = False,
    force: bool = False,
    config_path: Path | None = None,
    state_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Execute one scheduler cycle. Returns list of per-skill results."""
    config = load_config(config_path)
    state = load_state(state_path or DEFAULT_STATE_PATH)
    skills = config.get("skills", [])

    if not skills:
        print("  [scheduler] No target skills configured.", file=sys.stderr)
        return []

    default_interval = config.get("interval_hours", 24)
    workdir_base = SKILL_DIR / "scheduler_work"
    results: list[dict[str, Any]] = []

    for skill_cfg in skills:
        name = skill_cfg.get("name", "unnamed")
        skill_path = _resolve(skill_cfg.get("skill_path", ""))
        print(f"  [scheduler] [{name}] checking...", file=sys.stderr)

        # Get previous state for this skill
        prev_state = state["skills"].get(name)

        # Check if optimization is needed
        needs_opt, reason = check_skill_needs_opt(
            skill_cfg, prev_state,
            force=force,
            default_interval=default_interval,
        )

        if not needs_opt:
            print(f"  [scheduler] [{name}] skipping: {reason}", file=sys.stderr)
            results.append({
                "skill": name,
                "skill_path": skill_path,
                "status": "skipped",
                "reason": reason,
            })
            continue

        if check_only:
            print(f"  [scheduler] [{name}] WOULD optimize: {reason}", file=sys.stderr)
            results.append({
                "skill": name,
                "skill_path": skill_path,
                "status": "would_optimize",
                "reason": reason,
            })
            continue

        # Run optimization
        print(f"  [scheduler] [{name}] optimizing: {reason}", file=sys.stderr)
        opt_result = run_optimization(skill_cfg, workdir_base)

        # Update state
        update_state_after_opt(state, skill_cfg, opt_result)

        results.append(opt_result)

        # Brief status per skill
        status = opt_result.get("status", "error")
        score = opt_result.get("score", "?")
        accepted = opt_result.get("edits_accepted", 0)
        print(f"  [scheduler] [{name}] → {status} (score={score}, edits={accepted})", file=sys.stderr)

    # Save state
    save_state(state, state_path or DEFAULT_STATE_PATH)

    return results


def scan_skills_from_cron(
    skill_names: list[str],
    data_base: Path | None = None,
) -> list[dict[str, Any]]:
    """Discover skills by name, find their SKILL.md, check for trajectory data.

    Returns a list of config entries suitable for scheduler_config.json.
    Each entry includes a ``has_trajs`` flag indicating whether trajectory
    data was found at ``scheduler_data/<name>/``.
    """
    if not skill_names:
        print("  [scan] No skill names provided.", file=sys.stderr)
        return []

    skills_dir = Path.home() / ".hermes" / "skills"
    data_base = data_base or SKILL_DIR / "scheduler_data"
    results: list[dict[str, Any]] = []

    for name in sorted(set(skill_names)):
        # Skip self (no point optimizing the scheduler itself)
        if name == "skill-optimizer":
            continue

        # Find SKILL.md anywhere under ~/.hermes/skills/
        # Handles: <name>/SKILL.md, <cat>/<name>/SKILL.md, <cat>/<sub>/<name>/SKILL.md
        skill_path = None
        for candidate in skills_dir.rglob(f"**/{name}/SKILL.md"):
            skill_path = candidate.resolve()
            break

        if not skill_path:
            print(f"  [scan] ✗ {name}: no SKILL.md found under {skills_dir}", file=sys.stderr)
            results.append({
                "name": name,
                "skill_path": None,
                "status": "not_found",
                "error": "SKILL.md not found",
            })
            continue

        # Check for trajectory files
        traj_dir = data_base / name
        train_trajs = str(traj_dir / "train.json") if (traj_dir / "train.json").exists() else None
        sel_trajs = str(traj_dir / "sel.json") if (traj_dir / "sel.json").exists() else None
        has_trajs = train_trajs is not None

        entry = {
            "name": name,
            "skill_path": str(skill_path),
            "train_trajs": f"scheduler_data/{name}/train.json",
            "sel_trajs": f"scheduler_data/{name}/sel.json" if sel_trajs else None,
            "interval_hours": 48,
            "epochs": 1,
            "steps": 2,
            "budget_init": 3,
            "budget_final": 1,
            "has_trajs": has_trajs,
            "skill_size_chars": skill_path.stat().st_size,
            "status": "found" if has_trajs else "no_trajs",
        }

        if has_trajs:
            print(f"  [scan] ✓ {name} ({skill_path.name} dir, {skill_path.stat().st_size} chars) — trajectory OK", file=sys.stderr)
        else:
            print(f"  [scan] ~ {name} ({skill_path.name} dir, {skill_path.stat().st_size} chars) — NO trajectory", file=sys.stderr)

        results.append(entry)

    return results


# ── CLI entry ─────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SkillOpt Autonomous Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only report what needs optimization, don't run."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore interval checks, optimize all skills now."
    )
    parser.add_argument(
        "--scan-cron", nargs="*",
        help="Scan skills referenced by cron jobs. Pass skill names as args, or leave empty for no-op."
    )
    parser.add_argument(
        "--update-config", action="store_true",
        help="When used with --scan-cron, write results into the scheduler config file."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to scheduler config JSON (default: scripts/scheduler_config.json)"
    )
    parser.add_argument(
        "--state", type=str, default=None,
        help="Path to scheduler state JSON (default: scheduler_state.json)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only output structured JSON, no stderr diagnostics"
    )

    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve() if args.config else DEFAULT_CONFIG_PATH
    state_path = Path(args.state).expanduser().resolve() if args.state else DEFAULT_STATE_PATH

    # ── Scan-cron mode ────────────────────────────────────────────
    if args.scan_cron is not None:
        skill_names = args.scan_cron
        if not skill_names:
            if not args.quiet:
                print("No skill names provided. Usage:", file=sys.stderr)
                print("  uv run python3 scripts/opt_scheduler.py --scan-cron skill1 skill2 ...", file=sys.stderr)
            return

        entries = scan_skills_from_cron(skill_names)

        if not entries:
            if not args.quiet:
                print("No skills found.", file=sys.stderr)
            return

        # Build the config snippet
        config_snippet = {
            "interval_hours": 24,
            "default_epochs": 1,
            "default_steps": 2,
            "default_budget_init": 3,
            "default_budget_final": 1,
            "skills": [],
        }

        has_trajs = [e for e in entries if e.get("has_trajs")]
        no_trajs = [e for e in entries if not e.get("has_trajs")]
        not_found = [e for e in entries if e.get("status") == "not_found"]

        for e in entries:
            if e.get("skill_path"):
                config_snippet["skills"].append({
                    "name": e["name"],
                    "skill_path": e["skill_path"],
                    "train_trajs": e["train_trajs"],
                    "sel_trajs": e["sel_trajs"],
                    "interval_hours": e.get("interval_hours", 48),
                    "epochs": e.get("epochs", 1),
                    "steps": e.get("steps", 2),
                    "budget_init": e.get("budget_init", 3),
                    "budget_final": e.get("budget_final", 1),
                })

        # Write config if --update-config
        if args.update_config:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config_snippet, f, indent=2, ensure_ascii=False)
            print(f"\nConfig written to {config_path}", file=sys.stderr)

        # Output report
        lines = ["## SkillOpt 扫描结果 (cron 引用 skill)", ""]

        if has_trajs:
            lines.append(f"### ✅ 有轨迹文件 ({len(has_trajs)} skills)")
            lines.append("")
            for e in has_trajs:
                lines.append(f"- **{e['name']}**: {e['skill_path']} ({e.get('skill_size_chars', '?')} chars)")
            lines.append("")

        if no_trajs:
            lines.append(f"### ⏳ 需准备轨迹 ({len(no_trajs)} skills)")
            lines.append("")
            for e in no_trajs:
                lines.append(f"- **{e['name']}**: {e['skill_path']} ({e.get('skill_size_chars', '?')} chars)")
            lines.append("")
            lines.append("创建方法: `mkdir -p scheduler_data/<name>/` 并放入 train.json + sel.json")
            lines.append("")

        if not_found:
            lines.append(f"### ✗ 未找到 ({len(not_found)} skills)")
            for e in not_found:
                lines.append(f"- {e['name']}: {e.get('error', 'unknown')}")
            lines.append("")

        lines.append("---")
        lines.append(f"总共 {len(entries)} 个 skill，{len(has_trajs)} 个可立即优化")

        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({
            "found": len([e for e in entries if e["status"] not in ("not_found",)]),
            "has_trajs": len(has_trajs),
            "no_trajs": len(no_trajs),
            "not_found": len(not_found),
            "config_skills": len(config_snippet["skills"]),
        }, indent=2, ensure_ascii=False))
        lines.append("```")

        print("\n".join(lines))
        return

    results = run_scheduler(
        check_only=args.check_only,
        force=args.force,
        config_path=config_path,
        state_path=state_path,
    )

    # Determine what to output to stdout (this is what cron delivers)
    if args.check_only:
        # Output a summary even in check-only mode
        would_opt = [r for r in results if r["status"] == "would_optimize"]
        skipped = [r for r in results if r["status"] == "skipped"]
        if would_opt:
            report_lines = ["## SkillOpt 调度检查", ""]
            for r in would_opt:
                report_lines.append(f"- **{r['skill']}**: {r.get('reason', 'needs optimization')}")
            if skipped:
                report_lines.append("")
                for r in skipped:
                    report_lines.append(f"- {r['skill']}: {r.get('reason', 'skipped')}")
            print("\n".join(report_lines))
        else:
            # Silent — no skills need optimization
            pass

    elif any(r["status"] == "completed" for r in results):
        # At least one skill improved — output full report
        config = load_config(config_path)
        report = format_report(results, config)
        print(report)
    elif any(r["status"] == "error" for r in results):
        # Errors occurred — output error summary
        for r in results:
            if r["status"] == "error":
                print(f"❌ {r['skill']}: {r.get('error', 'unknown')}")
    else:
        # No changes — silent
        pass


if __name__ == "__main__":
    main()
