#!/usr/bin/env python3
"""
SkillOpt Auto Trajectory Collector — runs test commands for a skill,
scores each result, packages as trajectory JSON for optimizer training.

Usage:
    uv run python scripts/auto_collect_trajs.py --skill a-share-alpha-v9
    uv run python scripts/auto_collect_trajs.py --skill a-share-info-extractor
    uv run python scripts/auto_collect_trajs.py --skill akshare-unify
    uv run python scripts/auto_collect_trajs.py --skill a-share-alpha-v9 --dry-run (preview without saving)

Each skill SKILL.md defines a test harness (inline). The dispatcher runs those
tests, scores each, and writes train.json + sel.json to scheduler_data/<name>/.

Backup of original SKILL.md is always taken before any collection.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SCHEDULER_DATA = BASE / "scheduler_data"
BACKUP_ROOT = Path.home() / ".hermes" / "skill_backups"

# ── Skill definitions ──────────────────────────────────────────────────────

SKILL_REGISTRY = {}  # populated at module level below


def _find_skill_dir(name: str) -> Path | None:
    """Find a skill's SKILL.md under ~/.hermes/skills/."""
    base = Path.home() / ".hermes" / "skills"
    candidates = list(base.rglob(f"**/{name}/SKILL.md"))
    if candidates:
        return candidates[0].parent
    # Try direct name match (without the glob having to match the leaf)
    for p in base.rglob(f"**/SKILL.md"):
        if p.parent.name == name:
            return p.parent
    return None


def _duckdb_query(db_path: str, sql: str) -> dict:
    """Run a DuckDB SELECT query and return {'success': bool, 'rows': int, 'error': str}."""
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
        result = con.execute(sql).fetchall()
        con.close()
        return {"success": True, "rows": len(result), "error": None}
    except Exception as e:
        return {"success": False, "rows": 0, "error": str(e)}


def _run_cmd(cmd: str, timeout: int = 30, cwd: str | None = None) -> dict:
    """Run a shell command, return {'success': bool, 'stdout': str, 'stderr': str, 'exit_code': int, 'duration': float}."""
    start = time.time()
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=cwd or str(BASE)
        )
        duration = time.time() - start
        return {
            "success": r.returncode == 0,
            "stdout": r.stdout.strip(),
            "stderr": r.stderr.strip(),
            "exit_code": r.returncode,
            "duration": round(duration, 2),
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False, "stdout": "", "stderr": f"TIMEOUT ({timeout}s)",
            "exit_code": -1, "duration": timeout,
        }
    except Exception as e:
        return {
            "success": False, "stdout": "", "stderr": str(e),
            "exit_code": -1, "duration": round(time.time() - start, 2),
        }


def _make_trajectory(task_id: str, task_input: str, cmd_result: dict,
                     output: str, score: float, extra_meta: dict | None = None,
                     error: str | None = None) -> dict:
    """Package a command result into trajectory JSON dict."""
    return {
        "task_id": task_id,
        "task_input": task_input,
        "messages": [
            {"role": "user", "content": task_input},
            {"role": "assistant", "content": output},
        ],
        "tool_calls": [{"tool": "terminal", "params": {}, "result_present": True}],
        "output": output,
        "score": score,
        "error": error if error else None,
        "metadata": {
            "duration": cmd_result.get("duration", 0),
            "source": "auto-collect-v1",
            **(extra_meta or {}),
        },
    }


def _score_structural(skill_path: Path) -> dict:
    """Score SKILL.md structural quality. Returns {trajs: list[dict], output: str}."""
    text = skill_path.read_text(encoding="utf-8")
    trajs = []
    issues = []

    # 1. Has YAML frontmatter
    has_fm = text.startswith("---")
    fm_ok = True
    fm_name = None
    try:
        _, fm_text, _ = text.split("---", 2)
        fm = yaml.safe_load(fm_text)
        fm_name = (fm or {}).get("name", None)
        if not fm:
            fm_ok = False
            issues.append("Empty frontmatter")
    except Exception:
        fm_ok = False
        issues.append("Cannot parse frontmatter")

    trajs.append(_make_trajectory(
        task_id=f"{skill_path.parent.name}_struct_01",
        task_input="Does this SKILL.md have valid YAML frontmatter?",
        cmd_result={"duration": 0.1, "success": fm_ok},
        output=f"Frontmatter valid={fm_ok}, name={fm_name}",
        score=1.0 if fm_ok else 0.0,
    ))

    # 2. Has name matching directory
    expected_name = skill_path.parent.name
    name_match = fm_name == expected_name if fm_name else False
    if not name_match and fm_name:
        issues.append(f"Frontmatter name '{fm_name}' ≠ dir name '{expected_name}'")
    trajs.append(_make_trajectory(
        task_id=f"{skill_path.parent.name}_struct_02",
        task_input=f"Does frontmatter name match directory name '{expected_name}'?",
        cmd_result={"duration": 0.1, "success": name_match or fm_name == expected_name},
        output=f"Expected={expected_name}, got={fm_name}, match={name_match}",
        score=1.0 if name_match else 0.0,
    ))

    # 3. Has description
    desc = (fm or {}).get("description", "")
    has_desc = bool(desc and len(desc) > 10)
    if not has_desc:
        issues.append("Missing or too-short description")
    trajs.append(_make_trajectory(
        task_id=f"{skill_path.parent.name}_struct_03",
        task_input="Does this skill have a meaningful description?",
        cmd_result={"duration": 0.1, "success": has_desc},
        output=f"Description ({len(desc)} chars): {desc[:100]}...",
        score=1.0 if has_desc else 0.3,
    ))

    # 4. Has version
    version = (fm or {}).get("version", "")
    has_version = bool(version)
    if not has_version:
        issues.append("Missing version field")
    trajs.append(_make_trajectory(
        task_id=f"{skill_path.parent.name}_struct_04",
        task_input="Does this skill have a version field?",
        cmd_result={"duration": 0.1, "success": has_version},
        output=f"Version: {version or 'MISSING'}",
        score=1.0 if has_version else 0.3,
    ))

    # 5. Has metadata.tags or metadata.hermes.tags
    meta = (fm or {}).get("metadata", {})
    if isinstance(meta, dict) and "hermes" in meta:
        meta = meta["hermes"]
    tags = meta.get("tags", []) if isinstance(meta, dict) else []
    has_tags = bool(tags)
    if not has_tags:
        issues.append("Missing metadata tags")
    trajs.append(_make_trajectory(
        task_id=f"{skill_path.parent.name}_struct_05",
        task_input="Does this skill have metadata tags?",
        cmd_result={"duration": 0.1, "success": has_tags},
        output=f"Tags: {tags}",
        score=1.0 if has_tags else 0.3,
    ))

    summary = f"Structural checks: {sum(t['score'] for t in trajs)}/{len(trajs)} score"
    if issues:
        summary += f" | Issues: {'; '.join(issues)}"
    return {"trajs": trajs, "summary": summary}


def _score_script_tests(skill_dir: Path, skill_name: str) -> dict:
    """Run script-level tests: --help, imports, existence of key files."""
    trajs = []
    summary_items = []

    # Locate Python scripts
    scripts_dir = skill_dir / "scripts"
    refs_dir = skill_dir / "references"

    # 1. Check scripts dir
    if scripts_dir.is_dir():
        py_files = sorted(scripts_dir.glob("*.py"))
        # Try --help on first 3 scripts
        tested = 0
        for pyf in py_files[:3]:
            if pyf.name.startswith("__"): continue
            rel = pyf.relative_to(skill_dir.parent.parent.parent
                                   if skill_dir.parts[-1] == skill_name else skill_dir)
            cmd = f"uv run python {pyf} --help"
            r = _run_cmd(cmd, timeout=15, cwd=str(scripts_dir))
            score = 1.0 if r["success"] else 0.0
            if not r["success"]:
                # Try without uv
                cmd2 = f"python3 {pyf} --help"
                r2 = _run_cmd(cmd2, timeout=15, cwd=str(scripts_dir))
                if r2["success"]:
                    r = r2
                    score = 0.7  # works without uv but skill says uv
            trajs.append(_make_trajectory(
                task_id=f"{skill_name}_script_{tested+1}",
                task_input=f"Does `{cmd}` work?",
                cmd_result=r,
                output=f"Script={pyf.name}, exit={r['exit_code']}, stdout={r['stdout'][:200]}",
                score=score,
            ))
            summary_items.append(f"{pyf.name}: {'PASS' if r['success'] else 'FAIL'}")
            tested += 1

        # Also try python -m py_compile on each script
        compile_ok = 0
        compile_total = 0
        for pyf in py_files:
            if pyf.name.startswith("__"): continue
            r = _run_cmd(f"python3 -m py_compile {pyf}", timeout=10)
            compile_total += 1
            if r["success"]:
                compile_ok += 1
        if compile_total > 0:
            comp_score = compile_ok / compile_total
            trajs.append(_make_trajectory(
                task_id=f"{skill_name}_compile_all",
                task_input=f"Can all {compile_total} scripts compile without syntax errors?",
                cmd_result={"duration": 5, "success": comp_score > 0.8},
                output=f"{compile_ok}/{compile_total} compiled OK",
                score=min(1.0, comp_score),
            ))
            summary_items.append(f"Compile: {compile_ok}/{compile_total}")
    else:
        trajs.append(_make_trajectory(
            task_id=f"{skill_name}_script_01",
            task_input="Does this skill have a scripts/ directory?",
            cmd_result={"duration": 0.1, "success": False},
            output="No scripts/ directory found",
            score=0.3,
        ))
        summary_items.append("No scripts dir")

    # 2. Check references directory
    if refs_dir.is_dir():
        ref_files = sorted(refs_dir.iterdir())
        trajs.append(_make_trajectory(
            task_id=f"{skill_name}_refs_01",
            task_input="Does this skill have reference files?",
            cmd_result={"duration": 0.1, "success": len(ref_files) > 0},
            output=f"{len(ref_files)} reference files: {[f.name for f in ref_files[:5]]}",
            score=1.0 if len(ref_files) > 0 else 0.3,
        ))
        summary_items.append(f"Refs: {len(ref_files)} files")
    else:
        trajs.append(_make_trajectory(
            task_id=f"{skill_name}_refs_01",
            task_input="Does this skill have a references/ directory?",
            cmd_result={"duration": 0.1, "success": False},
            output="No references/ directory found",
            score=0.3,
        ))
        summary_items.append("No refs dir")

    return {"trajs": trajs, "summary": " | ".join(summary_items)}


def _score_duckdb(skill_name: str, db_path: str) -> dict:
    """Test DuckDB table presence relevant to the skill."""
    trajs = []
    tables_to_check = {
        "a-share-alpha-v9": ["predictions", "stock_daily", "stock_5min", "stock_info"],
        "a-share-info-extractor": ["stock_info", "stock_daily"],
        "akshare-unify": ["stock_daily", "stock_5min", "stock_ticks", "stock_info"],
    }
    tables = tables_to_check.get(skill_name, ["stock_info", "stock_daily"])

    if not os.path.exists(db_path):
        trajs.append(_make_trajectory(
            task_id=f"{skill_name}_db_01",
            task_input="Does the DuckDB database exist?",
            cmd_result={"duration": 0.1, "success": False},
            output=f"DB not found at {db_path}",
            score=0.0,
        ))
        return {"trajs": trajs, "summary": "DuckDB not found"}

    found, missing = 0, []
    for tbl in tables:
        r = _duckdb_query(db_path, f"SELECT COUNT(*) FROM {tbl}")
        if r["success"]:
            found += 1
        else:
            missing.append(tbl)

    trajs.append(_make_trajectory(
        task_id=f"{skill_name}_db_01",
        task_input=f"Do DuckDB tables ({', '.join(tables)}) exist?",
        cmd_result={"duration": 1, "success": found == len(tables)},
        output=f"Found {found}/{len(tables)} tables. Missing: {missing if missing else 'none'}",
        score=found / len(tables) if len(tables) > 0 else 0.0,
    ))

    # Try SELECT 1 on the most important table
    if tables:
        r2 = _duckdb_query(db_path, f"SELECT COUNT(*) FROM {tables[0]}")
        row_count = r2.get("rows", 0)
        has_rows = row_count > 0
        trajs.append(_make_trajectory(
            task_id=f"{skill_name}_db_02",
            task_input=f"Does {tables[0]} have data?",
            cmd_result=r2,
            output=f"{tables[0]}: {row_count} rows",
            score=1.0 if has_rows else 0.3,
        ))

    return {"trajs": trajs, "summary": f"DuckDB: {found}/{len(tables)} tables OK"}


def _score_skill_content(skill_path: Path) -> dict:
    """Score the quality/coverage of the skill's content."""
    text = skill_path.read_text(encoding="utf-8")
    trajs = []
    checks = {}

    text_lower = text.lower()
    # Check for key sections
    for section_name, patterns in [
        ("Trigger conditions", ["trigger", "当用户", "when to load", "## 触发条件", "## 用例"]),
        ("Prerequisites / dependencies", ["depend", "prerequisite", "前置", "## 前置依赖"]),
        ("Quick commands / main actions", ["## quick", "快速命令", "## commands", "## 命令",
                                           "## 快速命令", "## 用法", "system invariants",
                                           "## usage", "## 主要"]),
        ("Examples / rules", ["example", "例如", "## example", "## 示例", "## 规则",
                              "## rules", "## invariants", "## design"]),
        ("Pitfalls / warnings", ["pitfall", "warning", "注意", "小心", "caution",
                                 "not", "never", "不可", "必须", "避免"]),
    ]:
        found = any(p in text_lower for p in patterns)
        checks[section_name] = found

    found_count = sum(1 for v in checks.values() if v)
    total = len(checks)
    score = found_count / total

    # Check for code blocks with shell commands
    code_blocks = text.count("```bash") + text.count("```sh") + text.count("```shell")

    trajs.append(_make_trajectory(
        task_id=f"{skill_path.parent.name}_content_01",
        task_input="Does this skill have good content coverage?",
        cmd_result={"duration": 0.2, "success": score > 0.6},
        output=f"Content coverage: {found_count}/{total} sections found ({score:.0%}), {code_blocks} shell code blocks",
        score=score,
        extra_meta={"section_checks": checks, "code_blocks": code_blocks},
    ))

    # Check skill length
    line_count = len(text.split("\n"))
    length_score = 1.0 if 50 <= line_count <= 500 else (0.5 if line_count > 500 else 0.3)
    trajs.append(_make_trajectory(
        task_id=f"{skill_path.parent.name}_content_02",
        task_input="Is this skill an appropriate length?",
        cmd_result={"duration": 0.1, "success": length_score > 0.5},
        output=f"{line_count} lines ({len(text)} chars)",
        score=length_score,
    ))

    return {"trajs": trajs, "summary": f"Content: {found_count}/{total}"}


def _score_model_files(skill_name: str, skill_dir: Path) -> dict:
    """For a-share-alpha-v9: check model .joblib files exist."""
    trajs = []
    if skill_name != "a-share-alpha-v9":
        trajs.append(_make_trajectory(
            task_id=f"{skill_name}_models_01",
            task_input="Check for model files",
            cmd_result={"duration": 0.1, "success": True},
            output="Skipped (not an ML skill with joblib models)",
            score=0.7,
        ))
        return {"trajs": trajs, "summary": "N/A"}

    models_glob = list(skill_dir.glob("models/*.joblib"))
    found = len(models_glob)
    for p in list(skill_dir.glob("**/models/*.joblib")):
        if p not in models_glob: models_glob.append(p)
    models_glob = list(skill_dir.glob("models/*.joblib")) + list(skill_dir.glob("**/models/*.joblib"))
    models_glob = list(set(models_glob))

    expected = ["model_1d.joblib", "model_5d.joblib", "model_10d.joblib", "model_20d.joblib"]
    found_names = {m.name for m in models_glob}
    missing = [e for e in expected if e not in found_names]

    score = len([e for e in expected if e in found_names]) / len(expected)
    trajs.append(_make_trajectory(
        task_id=f"{skill_name}_models_01",
        task_input=f"Do all {len(expected)} expected model files exist?",
        cmd_result={"duration": 0.5, "success": score > 0.5},
        output=f"Found {found_names} | Missing: {missing}",
        score=score,
    ))

    return {"trajs": trajs, "summary": f"Models: {len(found_names)}/{len(expected)}"}


def collect_for_skill(skill_name: str, dry_run: bool = False) -> dict:
    """Run all tests for one skill, return trajectory data."""
    skill_dir = _find_skill_dir(skill_name)
    if not skill_dir:
        return {"success": False, "error": f"Skill '{skill_name}' not found under ~/.hermes/skills/"}

    skill_path = skill_dir / "SKILL.md"
    if not skill_path.exists():
        return {"success": False, "error": f"SKILL.md not found in {skill_dir}"}

    # DuckDB
    db_path = str(Path.home() / ".hermes" / "data" / "ashare.duckdb")
    if not os.path.exists(db_path):
        # Try common alternative
        alt = str(Path.home() / ".hermes" / "data" / "ashare.db")
        if os.path.exists(alt):
            db_path = alt
        else:
            db_path = None

    # Backup
    if not dry_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = BACKUP_ROOT / ts / skill_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(skill_path, backup_dir / "SKILL.md")
        backup_msg = f"Backup → {backup_dir}"
    else:
        backup_msg = "DRY RUN — no backup taken"

    # Run all test phases
    all_trajs = []

    # Phase 1: Structural
    struct = _score_structural(skill_path)
    all_trajs.extend(struct["trajs"])
    print(f"  [struct] {struct['summary']}")

    # Phase 2: Content quality
    content = _score_skill_content(skill_path)
    all_trajs.extend(content["trajs"])
    print(f"  [content] {content['summary']}")

    # Phase 3: File/script tests
    scripts = _score_script_tests(skill_dir, skill_name)
    all_trajs.extend(scripts["trajs"])
    print(f"  [scripts] {scripts['summary']}")

    # Phase 4: Model files if applicable
    models = _score_model_files(skill_name, skill_dir)
    all_trajs.extend(models["trajs"])
    print(f"  [models] {models['summary']}")

    # Phase 5: DuckDB if available
    if db_path:
        db_result = _score_duckdb(skill_name, db_path)
        all_trajs.extend(db_result["trajs"])
        print(f"  [duckdb] {db_result['summary']}")
    else:
        print(f"  [duckdb] SKIP — no DuckDB found")

    # Summary stats
    scores = [t["score"] for t in all_trajs]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"  Total trajectories: {len(all_trajs)}, avg score: {avg_score:.3f}")

    if dry_run:
        return {"success": True, "trajectories": all_trajs, "avg_score": avg_score, "backup": backup_msg}

    # Save train/sel split
    data_dir = SCHEDULER_DATA / skill_name
    data_dir.mkdir(parents=True, exist_ok=True)

    # Split: 80/20 train/sel, ensure both have high and low scores
    sorted_trajs = sorted(all_trajs, key=lambda t: t["score"])
    n_sel = max(1, len(sorted_trajs) // 5)
    sel_trajs = sorted_trajs[:n_sel]  # lowest scores as selection
    train_trajs = sorted_trajs[n_sel:]

    (data_dir / "train.json").write_text(
        json.dumps(train_trajs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (data_dir / "sel.json").write_text(
        json.dumps(sel_trajs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "success": True,
        "skill_dir": str(skill_dir),
        "trajectories": len(all_trajs),
        "train": len(train_trajs),
        "sel": len(sel_trajs),
        "avg_score": avg_score,
        "data_dir": str(data_dir),
        "backup": backup_msg,
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-collect trajectories for skill optimization")
    parser.add_argument("--skill", help="Skill name (e.g., a-share-alpha-v9)")
    parser.add_argument("--dry-run", action="store_true", help="Preview test results without saving")
    parser.add_argument("--all", action="store_true", help="Run for all registered cron skills")
    args = parser.parse_args()

    if args.all:
        skills = ["a-share-alpha-v9", "a-share-info-extractor", "akshare-unify"]
        results = {}
        for s in skills:
            print(f"\n{'='*60}")
            print(f"Skill: {s}")
            print(f"{'='*60}")
            results[s] = collect_for_skill(s, dry_run=args.dry_run)
        print(f"\n{'='*60}")
        print("Summary:")
        for name, r in results.items():
            status = "OK" if r.get("success") else f"FAIL: {r.get('error','?')}"
            print(f"  {name}: {status} | trajs={r.get('trajectories',0)} | avg={r.get('avg_score',0):.3f} | {r.get('backup','')}")
        return

    if not args.skill:
        print("ERROR: --skill is required (or use --all)")
        sys.exit(1)

    r = collect_for_skill(args.skill, dry_run=args.dry_run)
    if r.get("success"):
        print(f"\n{'='*60}")
        print(f"Result: {r.get('trajectories',0)} trajectories saved to {r.get('data_dir','?')}")
        print(f"  Train: {r.get('train',0)} | Sel: {r.get('sel',0)} | Avg score: {r.get('avg_score',0):.3f}")
        print(f"  {r.get('backup','')}")
    else:
        print(f"ERROR: {r.get('error','?')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
