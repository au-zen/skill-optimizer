#!/usr/bin/env python3
"""
score_skill.py — Structural quality scorer for SKILL.md documents.

Evaluates a Hermes skill document against authoring conventions and
structural requirements. Used by SkillOpt's candidate-aware evaluation
to score the quality of generated candidates.

Usage:
    uv run python3 scripts/score_skill.py --skill ./SKILL.md
    uv run python3 scripts/score_skill.py --skill ./SKILL.md --verbose
    uv run python3 scripts/score_skill.py --skill ./SKILL.md --stability
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def score_skill(skill_path: str, verbose: bool = False, skip_name_check: bool = False) -> dict:
    """Score a SKILL.md document's structural quality.

    Parameters
    ----------
    skill_path : str
        Path to the SKILL.md file.
    verbose : bool
        Print per-check results to stdout.
    skip_name_check : bool
        Skip the frontmatter name vs directory check.
        Used during optimization where candidates are in temp directories.

    Returns a dict with overall_score, per-check scores, and any issues.
    """
    path = Path(skill_path)
    if not path.exists():
        return {
            "skill_path": skill_path,
            "overall_score": 0.0,
            "checks": [],
            "issues": [f"File not found: {skill_path}"],
            "error": "file_not_found",
        }

    text = path.read_text(encoding="utf-8")
    checks: list[dict] = []
    issues: list[str] = []

    # ── 1. YAML frontmatter ──
    fm_valid = False
    fm = {}
    if not text.startswith("---"):
        issues.append("File must start with --- (YAML frontmatter)")
        checks.append({"name": "frontmatter_delimiters", "score": 0.0, "detail": "Missing opening ---"})
    else:
        end_idx = text.find("---", 3)
        if end_idx == -1:
            issues.append("Unclosed YAML frontmatter")
            checks.append({"name": "frontmatter_delimiters", "score": 0.0, "detail": "No closing ---"})
        else:
            fm_text = text[3:end_idx].strip()
            if not fm_text:
                issues.append("Empty YAML frontmatter")
                checks.append({"name": "frontmatter_delimiters", "score": 0.3, "detail": "Empty frontmatter"})
            else:
                try:
                    if yaml:
                        fm = yaml.safe_load(fm_text) or {}
                    else:
                        fm = _parse_frontmatter_simple(fm_text)
                    fm_valid = True
                    checks.append({"name": "frontmatter_delimiters", "score": 1.0, "detail": "Valid YAML frontmatter"})
                except Exception as e:
                    issues.append(f"Cannot parse frontmatter: {e}")
                    checks.append({"name": "frontmatter_delimiters", "score": 0.0, "detail": str(e)})

    # ── 2. name field ──
    if fm_valid:
        name = fm.get("name", "")
        if skip_name_check:
            score = 1.0 if name else 0.0
            if not name:
                issues.append("Missing 'name' in frontmatter")
            checks.append({
                "name": "frontmatter_name",
                "score": score,
                "detail": f"name={name} (dir check skipped)",
            })
        else:
            expected = path.parent.name
            name_match = name == expected if name else False
            score = 1.0 if name_match else (0.5 if name else 0.0)
            if not name:
                issues.append("Missing 'name' in frontmatter")
            elif not name_match:
                issues.append(f"Frontmatter name '{name}' != directory name '{expected}'")
            checks.append({
                "name": "frontmatter_name",
                "score": score,
                "detail": f"name={name}, expected={expected}",
            })
    else:
        checks.append({"name": "frontmatter_name", "score": 0.0, "detail": "Skipped (no valid frontmatter)"})

    # ── 3. description field ──
    if fm_valid:
        desc = fm.get("description", "") or ""
        has_desc = len(desc) > 10
        use_when = desc.strip().startswith("Use when") if has_desc else False
        score = 1.0 if (has_desc and use_when) else (0.5 if has_desc else 0.0)
        if not has_desc:
            issues.append("Missing or too-short description")
        elif not use_when:
            issues.append("Description should start with 'Use when...'")
        checks.append({
            "name": "frontmatter_description",
            "score": score,
            "detail": f"{len(desc)} chars, starts_with_Use_when={use_when}",
        })
    else:
        checks.append({"name": "frontmatter_description", "score": 0.0, "detail": "Skipped"})

    # ── 4. version field ──
    if fm_valid:
        version = fm.get("version", "")
        has_version = bool(version)
        if not has_version:
            issues.append("Missing 'version' in frontmatter")
        checks.append({
            "name": "frontmatter_version",
            "score": 1.0 if has_version else 0.0,
            "detail": f"version={version or 'MISSING'}",
        })
    else:
        checks.append({"name": "frontmatter_version", "score": 0.0, "detail": "Skipped"})

    # ── 5. metadata tags ──
    if fm_valid:
        meta = fm.get("metadata", {})
        if isinstance(meta, dict) and "hermes" in meta:
            meta = meta["hermes"]
        tags = meta.get("tags", []) if isinstance(meta, dict) else []
        has_tags = bool(tags)
        if not has_tags:
            issues.append("Missing metadata.hermes.tags")
        checks.append({
            "name": "metadata_tags",
            "score": 1.0 if has_tags else 0.3,
            "detail": f"tags={tags}",
        })
    else:
        checks.append({"name": "metadata_tags", "score": 0.0, "detail": "Skipped"})

    # ── 6. Code blocks balanced ──
    fence_count = text.count("```")
    blocks_balanced = fence_count % 2 == 0
    if not blocks_balanced:
        issues.append(f"Unclosed fenced code block ({fence_count} fences)")
    checks.append({
        "name": "code_blocks_balanced",
        "score": 1.0 if blocks_balanced else 0.0,
        "detail": f"{fence_count} fences, balanced={blocks_balanced}",
    })

    # ── 7. Section headings present ──
    headings = re.findall(r"^#{1,3}\s+\S", text, re.MULTILINE)
    has_headings = len(headings) >= 3
    if not has_headings:
        issues.append(f"Too few section headings ({len(headings)} found, need ≥3)")
    checks.append({
        "name": "section_headings",
        "score": min(1.0, len(headings) / 6),
        "detail": f"{len(headings)} headings found",
    })

    # ── 8. Has required sections ──
    text_lower = text.lower()
    required = ["overview", "when to use", "common pitfalls"]
    found_req = sum(1 for s in required if f"## {s}" in text_lower)
    if found_req < len(required):
        missing = [s for s in required if f"## {s}" not in text_lower]
        issues.append(f"Missing required sections: {missing}")
    checks.append({
        "name": "required_sections",
        "score": found_req / len(required),
        "detail": f"found {found_req}/{len(required)} required sections",
    })

    # ── 9. No bare URLs ──
    urls = re.findall(r"https?://[^\)\s\"]+", text)
    has_bare_urls = len(urls) > 0
    if has_bare_urls:
        issues.append(f"Found {len(urls)} bare URLs")
    checks.append({
        "name": "no_bare_urls",
        "score": 0.0 if has_bare_urls else 1.0,
        "detail": f"{len(urls)} bare URLs" if has_bare_urls else "No bare URLs",
    })

    # ── Overall score (weighted average) ──
    weights = {
        "frontmatter_delimiters": 3,
        "frontmatter_name": 2,
        "frontmatter_description": 2,
        "frontmatter_version": 1,
        "metadata_tags": 1,
        "code_blocks_balanced": 2,
        "section_headings": 2,
        "required_sections": 3,
        "no_bare_urls": 2,
    }
    total_weight = sum(weights.get(c["name"], 1) for c in checks)
    weighted = sum(weights.get(c["name"], 1) * c["score"] for c in checks)
    overall = weighted / total_weight if total_weight > 0 else 0.0

    result = {
        "skill_path": str(path.resolve()),
        "overall_score": round(overall, 4),
        "checks": checks,
        "issues": issues,
        "n_checks": len(checks),
        "n_issues": len(issues),
    }

    if verbose:
        print(f"  [score_skill] {path.name}: {overall:.4f} ({len(issues)} issues)")
        for c in checks:
            status = "✓" if c["score"] >= 0.8 else ("⚠" if c["score"] >= 0.3 else "✗")
            print(f"    {status} {c['name']}: {c['score']:.2f}  ({c['detail']})")
        if issues:
            print(f"    Issues:")
            for i in issues:
                print(f"      - {i}")

    return result


def _parse_frontmatter_simple(text: str) -> dict:
    """Simple frontmatter parser when PyYAML is unavailable."""
    result = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def check_stability(skill_path: str, verbose: bool = False) -> dict:
    """Run scorer twice on the same file and report max difference."""
    r1 = score_skill(skill_path, verbose=verbose)
    r2 = score_skill(skill_path, verbose=verbose)
    diff = abs(r1["overall_score"] - r2["overall_score"])
    check_diffs = {}
    for c1, c2 in zip(r1["checks"], r2["checks"]):
        d = abs(c1["score"] - c2["score"])
        if d > 0.01:
            check_diffs[c1["name"]] = d

    stable = diff < 0.01 and not check_diffs
    result = {
        "run1_score": r1["overall_score"],
        "run2_score": r2["overall_score"],
        "max_diff": round(diff, 6),
        "stable": stable,
    }
    if check_diffs:
        result["check_diffs"] = check_diffs

    if verbose:
        status = "STABLE" if stable else "UNSTABLE"
        print(f"  [stability] {status}: run1={r1['overall_score']:.4f} run2={r2['overall_score']:.4f} diff={diff:.6f}")
        if check_diffs:
            for name, d in check_diffs.items():
                print(f"    Unstable check: {name} (diff={d:.4f})")

    return result


def main():
    parser = argparse.ArgumentParser(description="Score a SKILL.md document's structural quality")
    parser.add_argument("--skill", required=True, help="Path to SKILL.md")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-check results")
    parser.add_argument("--stability", action="store_true", help="Run twice and report stability")
    parser.add_argument("--output", help="Write JSON result to file")
    parser.add_argument("--skip-name-check", action="store_true", help="Skip frontmatter name vs directory check")
    args = parser.parse_args()

    if args.stability:
        result = check_stability(args.skill, verbose=args.verbose)
    else:
        result = score_skill(args.skill, verbose=args.verbose, skip_name_check=args.skip_name_check)

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.stability:
        sys.exit(0 if result.get("stable") else 1)
    else:
        # Exit 0 if score is meaningful, 1 on major issues
        sys.exit(0 if result["overall_score"] >= 0.3 else 1)


if __name__ == "__main__":
    main()
