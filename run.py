#!/usr/bin/env python3
"""
Convenience entry point for SkillOpt engine.

Usage:
  uv run python3 run.py optimize --skill ./SKILL.md --train-trajs trajs.json ...
  uv run python3 run.py --help
"""

from scripts.optimizer_cli import main

if __name__ == "__main__":
    main()
