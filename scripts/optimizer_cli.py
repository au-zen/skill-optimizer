#!/usr/bin/env python3
"""
SkillOpt CLI — Command-line entry point for the optimisation engine.

Usage
-----
  uv run python3 scripts/optimizer_cli.py <command> [options]

Commands
--------
  optimize   Run full optimisation loop on a skill.
  rollout    Single rollout batch (collect trajectories).
  reflect    Single reflection step (propose edits from trajectories).
  validate   Run validation gate on a candidate skill.
  view       View optimisation state / best skill.
  metrics    Show optimisation metrics.

Examples
--------
  # Full optimisation
  uv run python3 scripts/optimizer_cli.py optimize \\
      --skill ./my-skill/SKILL.md \\
      --train task_1.json task_2.json \\
      --sel task_3.json \\
      --epochs 4 --steps 4 --budget-init 5 --budget-final 1

  # Use manual trajectory mode
  uv run python3 scripts/optimizer_cli.py optimize \\
      --skill ./my-skill/SKILL.md \\
      --train-trajs collected_train.json \\
      --sel-trajs collected_sel.json \\
      --epochs 2 --steps 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .protocols import OptimizerConfig, OptimizationState, Trajectory
from .engine import SkillOptEngine
from .llm_client import OptimizerLLMClient
from .rollout import RolloutHarness, JsonScorer
from .reward_extractor import RewardConfig, convert_hermes_jsonl, load_hermes_jsonl, parse_weights


# ── Helpers ──────────────────────────────────────────────────────────


def _load_trajectories(path: str) -> list[Trajectory]:
    """Load pre-collected trajectories from a JSON file."""
    return RolloutHarness.load_trajectories(path)


def _load_hermes_trajectories(paths: list[str] | None, weights: str | None = None) -> list[Trajectory]:
    """Load Hermes Runtime JSONL/JSON and extract SkillOpt rewards."""
    if not paths:
        return []
    config = RewardConfig(weights=parse_weights(weights))
    return load_hermes_jsonl(paths, config=config)


def _trajectory_paths(paths: list[str]) -> list[str]:
    """Resolve trajectory file paths."""
    return [str(Path(p).resolve()) for p in paths]


def _install_cached_rollout(engine: SkillOptEngine, trajectories: list[Trajectory]) -> None:
    """Route task IDs to already-loaded trajectories for manual-mode commands."""
    cached_task_map = {t.task_id: t for t in trajectories}
    original_rollout = engine._rollout

    def _cached_rollout(self_eng, skill_path, tasks):
        if tasks:
            cached_trajs = [cached_task_map[t] for t in tasks if t in cached_task_map]
            if len(cached_trajs) == len(tasks):
                return cached_trajs
        return original_rollout(skill_path, tasks)

    import types

    engine._rollout = types.MethodType(_cached_rollout, engine)


class _NoopLLMClient:
    """Placeholder for commands that only exercise rollout/validation logic."""

    def chat_structured(self, *args, **kwargs):
        raise RuntimeError("This command does not perform optimizer LLM calls")


# ── Commands ─────────────────────────────────────────────────────────


def cmd_optimize(args: argparse.Namespace):
    """Run the full SkillOpt optimisation loop."""
    config = OptimizerConfig(
        rollout_batch_size=args.batch_size,
        reflection_minibatch_size=args.minibatch,
        edit_budget_init=args.budget_init,
        edit_budget_final=args.budget_final,
        schedule=args.schedule,
        max_epochs=args.epochs,
        max_steps_per_epoch=args.steps,
        accept_strict=not args.non_strict,
        optimizer_model=args.optimizer_model or "deepseek-chat",
        optimizer_api_base=args.optimizer_api_base or "https://api.deepseek.com/v1",
        work_dir=args.work_dir,
        min_similarity=args.min_similarity,
        use_secondary_scorer=args.secondary_scorer,
        min_reward_delta=args.min_reward_delta,
        completed_rate_tolerance=args.completed_rate_tolerance,
        tool_failure_rate_tolerance=args.tool_failure_rate_tolerance,
    )

    # Determine split sources
    train_split = args.train or []
    sel_split = args.sel or []

    if args.train_trajs:
        train_trajs = _load_trajectories(args.train_trajs)
        train_split = [t.task_id for t in train_trajs]
        print(f"  Loaded {len(train_trajs)} train trajectories from {args.train_trajs}")
    else:
        train_trajs = None

    if args.sel_trajs:
        sel_trajs = _load_trajectories(args.sel_trajs)
        sel_split = [t.task_id for t in sel_trajs]
        print(f"  Loaded {len(sel_trajs)} sel trajectories from {args.sel_trajs}")
    else:
        sel_trajs = None

    if args.train_hermes_jsonl:
        train_trajs = _load_hermes_trajectories(args.train_hermes_jsonl, args.reward_weights)
        train_split = [t.task_id for t in train_trajs]
        print(f"  Extracted {len(train_trajs)} train rewards from Hermes trajectories")

    if args.sel_hermes_jsonl:
        sel_trajs = _load_hermes_trajectories(args.sel_hermes_jsonl, args.reward_weights)
        sel_split = [t.task_id for t in sel_trajs]
        print(f"  Extracted {len(sel_trajs)} sel rewards from Hermes trajectories")

    # Stage 3: holdout split
    holdout_split = []
    holdout_trajs = []
    if args.holdout_trajs:
        holdout_trajs = _load_trajectories(args.holdout_trajs)
        holdout_split = [t.task_id for t in holdout_trajs]
        config.holdout_rollout = True
        print(f"  Loaded {len(holdout_trajs)} holdout trajectories from {args.holdout_trajs}")

    if not sel_split:
        raise ValueError(
            "A validation split is required for optimization. Provide --sel or --sel-trajs."
        )

    if args.hermes_default:
        llm = OptimizerLLMClient.from_hermes_config(calls_per_minute=args.rpm)
        config.optimizer_model = llm.model
        config.optimizer_api_base = llm.api_base
    else:
        llm = OptimizerLLMClient(
            model=args.optimizer_model,
            api_base=args.optimizer_api_base,
            calls_per_minute=args.rpm,
        )

    # Determine rollout mode
    rollout_mode = getattr(args, "rollout_mode", "manual")
    if rollout_mode in {"hermes_auto", "hermes_static_score"}:
        harness = RolloutHarness(mode="hermes_static_score")
        engine = SkillOptEngine(
            target_skill_path=args.skill,
            config=config,
            llm_client=llm,
            rollout_harness=harness,
            train_split=train_split,
            sel_split=sel_split,
            test_split=holdout_split,
        )
    elif train_trajs is not None:
        # Manual mode: use pre-collected trajectories
        harness = RolloutHarness(mode="manual")
        # We'll inject trajectories directly
        engine = SkillOptEngine(
            target_skill_path=args.skill,
            config=config,
            llm_client=llm,
            rollout_harness=harness,
            train_split=train_split,
            sel_split=sel_split,
            test_split=holdout_split,
        )
        # Inject pre-loaded trajectories into the harness for rollout
        cached_all = train_trajs + (sel_trajs or []) + (holdout_trajs if holdout_split else [])
        engine.harness._cached_train = cached_all
        _install_cached_rollout(engine, cached_all)
    else:
        # Script mode: run commands per task
        harness = RolloutHarness(mode="script", scorer=JsonScorer())
        engine = SkillOptEngine(
            target_skill_path=args.skill,
            config=config,
            llm_client=llm,
            rollout_harness=harness,
            train_split=train_split,
            sel_split=sel_split,
            test_split=holdout_split,
        )

    print(f"\n{'='*60}")
    print(f"  SkillOpt Optimisation")
    print(f"  Skill     : {args.skill}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Steps/Ep  : {args.steps}")
    print(f"  Schedule  : {args.schedule} ({args.budget_init} → {args.budget_final})")
    print(f"  Optimizer : {config.optimizer_model}")
    print(f"  Work dir  : {args.work_dir}")
    print(f"  Train     : {len(train_split)} tasks")
    print(f"  Sel       : {len(sel_split)} tasks")
    print(f"{'='*60}\n")

    state = engine.run()

    print(f"\n✅ Optimisation complete.")
    print(f"   Best score: {state.best_score:.4f}")
    print(f"   Best skill: {state.best_skill_path}")
    print(f"   Edits accepted: {len(state.accepted_edits)}")
    print(f"   Edits rejected: {len(state.rejected_buffer)}")


def cmd_rollout(args: argparse.Namespace):
    """Single rollout batch — collect trajectories."""
    harness = RolloutHarness(mode=args.mode, scorer=JsonScorer())
    tasks = _trajectory_paths(args.tasks) if args.mode == "manual" else args.tasks
    trajectories = harness.execute(args.skill, tasks, batch_size=args.batch_size)

    out_path = args.output or "trajectories.json"
    RolloutHarness.save_trajectories(trajectories, out_path)
    print(f"Saved {len(trajectories)} trajectories to {out_path}")


def cmd_reflect(args: argparse.Namespace):
    """Single reflection step — propose edits from trajectories."""
    from .engine import SkillOptEngine

    trajs = _load_trajectories(args.trajectories)
    config = OptimizerConfig()
    llm = OptimizerLLMClient(model=args.optimizer_model)

    engine = SkillOptEngine(
        target_skill_path=args.skill,
        config=config,
        llm_client=llm,
        train_split=[t.task_id for t in trajs],
    )
    engine.state.current_skill_path = str(Path(args.skill).resolve())

    edits = engine._reflect_and_propose(trajs, args.budget or config.edit_budget_init)
    print(json.dumps([e.to_dict() for e in edits], indent=2, ensure_ascii=False))


def cmd_validate(args: argparse.Namespace):
    """Run validation gate on a candidate skill."""
    from .engine import SkillOptEngine

    config = OptimizerConfig()
    harness = RolloutHarness(mode=args.mode, scorer=JsonScorer())
    sel_trajs = _load_trajectories(args.sel_trajs) if args.sel_trajs else []

    engine = SkillOptEngine(
        target_skill_path=args.skill,
        config=config,
        llm_client=_NoopLLMClient(),
        rollout_harness=harness,
        sel_split=[t.task_id for t in sel_trajs] if sel_trajs else [],
    )
    if sel_trajs:
        _install_cached_rollout(engine, sel_trajs)

    engine.state.current_skill_path = str(Path(args.skill).resolve())
    engine.state.origin_skill_path = engine.state.current_skill_path
    baseline_trajs = engine._rollout(engine.state.current_skill_path, engine.splits["sel"])
    if baseline_trajs:
        engine.state.current_score = sum(t.score for t in baseline_trajs) / len(baseline_trajs)
        engine.state.best_score = engine.state.current_score

    candidate_path = str(Path(args.candidate).resolve())
    accepted = engine._validate_and_gate(candidate_path)
    print(f"Candidate: {candidate_path}")
    print(f"Accepted:  {accepted}")
    print(f"Score:     {engine.state.current_score:.4f}")


def cmd_extract_rewards(args: argparse.Namespace):
    """Convert Hermes Runtime trajectories into SkillOpt reward trajectories."""
    config = RewardConfig(
        weights=parse_weights(args.weights),
        max_api_calls=args.max_api_calls,
        max_tool_calls=args.max_tool_calls,
    )
    trajectories = convert_hermes_jsonl(args.input, args.output, config=config)
    print(f"Converted {len(trajectories)} trajectories to {args.output}")


def cmd_view(args: argparse.Namespace):
    """View optimisation state."""
    state = SkillOptEngine.load_state(args.work_dir)
    if state is None:
        print(f"No state found in {args.work_dir}")
        return
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))

    # Show best skill if it exists
    if state.best_skill_path and Path(state.best_skill_path).exists():
        content = Path(state.best_skill_path).read_text()
        lines = content.splitlines()
        print(f"\n{'='*60}")
        print(f"  best_skill.md ({len(lines)} lines, {len(content)} chars)")
        print(f"{'='*60}")
        for line in lines[:50]:
            print(line)
        if len(lines) > 50:
            print(f"  ... ({len(lines) - 50} more lines)")


def cmd_metrics(args: argparse.Namespace):
    """Show optimisation metrics."""
    path = Path(args.work_dir) / "metrics.jsonl"
    if not path.exists():
        print(f"No metrics found in {args.work_dir}")
        return
    metrics = []
    with open(path) as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    if not metrics:
        print("No metrics entries.")
        return
    print(f"{'Step':<10} {'Current':<10} {'Candidate':<10} {'Delta':<10}  {'N':<5}")
    print("-" * 50)
    for m in metrics:
        print(f"E{m['epoch']}S{m['step']:<5} {m['current_score']:<10.4f} "
              f"{m['candidate_score']:<10.4f} {m['delta']:<+10.4f}  {m['n_sel']:<5}")


# ── CLI entry point ──────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SkillOpt Engine CLI — Optimise Hermes agent skills.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── optimize ──
    opt = sub.add_parser("optimize", help="Run full optimisation loop")
    opt.add_argument("--skill", required=True, help="Path to target SKILL.md")
    opt.add_argument("--train", nargs="*", default=[], help="Training task IDs")
    opt.add_argument("--sel", nargs="*", default=[], help="Selection task IDs")
    opt.add_argument("--train-trajs", help="Pre-collected train trajectories JSON")
    opt.add_argument("--sel-trajs", help="Pre-collected sel trajectories JSON")
    opt.add_argument("--train-hermes-jsonl", nargs="*", help="Hermes Runtime train trajectory JSONL/JSON files")
    opt.add_argument("--sel-hermes-jsonl", nargs="*", help="Hermes Runtime selection trajectory JSONL/JSON files")
    opt.add_argument("--reward-weights", help="Comma-separated Hermes reward weights")
    opt.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    opt.add_argument("--steps", type=int, default=4, help="Steps per epoch")
    opt.add_argument("--batch-size", type=int, default=8, help="Rollout batch size")
    opt.add_argument("--minibatch", type=int, default=4, help="Reflection minibatch size")
    opt.add_argument("--budget-init", type=int, default=5, help="Initial edit budget")
    opt.add_argument("--budget-final", type=int, default=1, help="Final edit budget")
    opt.add_argument("--schedule", choices=["cosine", "constant", "linear"],
                     default="cosine", help="Edit budget schedule")
    opt.add_argument("--non-strict", action="store_true",
                     help="Non-strict validation (>= instead of >)")
    opt.add_argument("--min-reward-delta", type=float, default=0.0,
                     help="Minimum reward delta required by validation gate")
    opt.add_argument("--completed-rate-tolerance", type=float, default=0.0,
                     help="Allowed completed-rate regression in validation gate")
    opt.add_argument("--tool-failure-rate-tolerance", type=float, default=0.05,
                     help="Allowed tool failure-rate regression in validation gate")
    opt.add_argument("--hermes-default", action="store_true",
                     help="Use Hermes' current default model as optimizer (reads ~/.hermes/config.yaml)")
    opt.add_argument("--optimizer-model",
                     default="deepseek-chat",
                     help="Optimizer LLM model name")
    opt.add_argument("--optimizer-api-base",
                     default="https://api.deepseek.com/v1",
                     help="Optimizer LLM API base URL")
    opt.add_argument("--rpm", type=int, default=30,
                     help="Optimizer LLM calls per minute")
    opt.add_argument("--work-dir", default="./skillopt_work",
                     help="Work directory for state/metrics")
    # Stage 2: drift detection
    opt.add_argument("--min-similarity", type=float, default=0.6,
                     help="Minimum text similarity to origin for accepting edits")
    # Stage 3: robustness
    opt.add_argument("--secondary-scorer", action="store_true",
                     help="Enable secondary scorer cross-validation")
    opt.add_argument("--holdout-trajs",
                     help="Holdout trajectories JSON (monitoring only)")
    opt.add_argument("--rollout-mode",
                     default="manual",
                     choices=["manual", "script", "hermes_auto", "hermes_static_score"],
                     help="Rollout mode for trajectory collection (hermes_auto is a legacy alias for hermes_static_score)")

    # ── rollout ──
    rl = sub.add_parser("rollout", help="Single rollout batch")
    rl.add_argument("--skill", required=True, help="Path to SKILL.md")
    rl.add_argument("--tasks", nargs="+", required=True, help="Task IDs")
    rl.add_argument("--mode", default="manual",
                    choices=["manual", "script"], help="Rollout mode")
    rl.add_argument("--batch-size", type=int, default=8)
    rl.add_argument("--output", help="Output JSON path")

    # ── reflect ──
    rf = sub.add_parser("reflect", help="Single reflection step")
    rf.add_argument("--skill", required=True, help="Path to SKILL.md")
    rf.add_argument("--trajectories", required=True, help="Trajectories JSON")
    rf.add_argument("--budget", type=int, help="Edit budget")
    rf.add_argument("--optimizer-model",
                    default="deepseek-chat")

    # ── validate ──
    vl = sub.add_parser("validate", help="Run validation gate")
    vl.add_argument("--skill", required=True, help="Original SKILL.md")
    vl.add_argument("--candidate", required=True, help="Candidate SKILL.md")
    vl.add_argument("--sel-trajs", help="Sel trajectories JSON")
    vl.add_argument("--mode", default="manual",
                    choices=["manual", "script"])

    # ── extract-rewards ──
    er = sub.add_parser("extract-rewards", help="Convert Hermes trajectories to SkillOpt reward trajectories")
    er.add_argument("--input", nargs="+", required=True, help="Hermes JSONL/JSON trajectory files")
    er.add_argument("--output", required=True, help="Output SkillOpt trajectory JSON")
    er.add_argument("--weights", help="Comma-separated reward weights")
    er.add_argument("--max-api-calls", type=int, default=8)
    er.add_argument("--max-tool-calls", type=int, default=12)

    # ── view ──
    vw = sub.add_parser("view", help="View optimisation state")
    vw.add_argument("--work-dir", default="./skillopt_work")

    # ── metrics ──
    mt = sub.add_parser("metrics", help="Show optimisation metrics")
    mt.add_argument("--work-dir", default="./skillopt_work")

    args = parser.parse_args()

    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "rollout":
        cmd_rollout(args)
    elif args.command == "reflect":
        cmd_reflect(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "extract-rewards":
        cmd_extract_rewards(args)
    elif args.command == "view":
        cmd_view(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
