import json
import tempfile
import unittest
from pathlib import Path

from scripts.reward_extractor import RewardConfig, extract_reward, load_hermes_jsonl, parse_weights


class RewardExtractorTests(unittest.TestCase):
    def test_extracts_runtime_reward_components(self):
        entry = {
            "task_id": "t1",
            "task_input": "inspect skill",
            "completed": True,
            "api_calls": 7,
            "toolsets_used": ["file_tools"],
            "tool_stats": {
                "read_file": {"count": 3, "success": 3, "failure": 0},
                "terminal": {"count": 2, "success": 1, "failure": 1},
            },
            "tool_error_counts": {"terminal.timeout": 1},
            "answer_judge": 0.8,
        }

        result = extract_reward(entry, RewardConfig())

        self.assertEqual(result.components["completed"], 1.0)
        self.assertEqual(result.components["tool_success_rate"], 0.8)
        self.assertEqual(result.components["error_penalty"], 0.8)
        self.assertIn("terminal", result.diagnostics["high_failure_tools"])
        self.assertGreater(result.score, 0.8)

    def test_converts_jsonl_to_skillopt_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory_samples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "sample-1",
                        "input": "load skill",
                        "completed": True,
                        "api_calls": 1,
                        "tool_stats": {"skill_view": {"count": 1, "success": 1, "failure": 0}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            trajectories = load_hermes_jsonl(path)

        self.assertEqual(len(trajectories), 1)
        traj = trajectories[0]
        self.assertEqual(traj.task_id, "sample-1")
        self.assertEqual(traj.metadata["reward_source"], "hermes_trajectory")
        self.assertEqual(traj.metadata["reward_components"]["completed"], 1.0)
        self.assertEqual(traj.metadata["hermes"]["tool_stats"]["skill_view"]["success"], 1)

    def test_parse_weights_overrides_defaults(self):
        weights = parse_weights("completed=0.7,tool_success_rate=0.1")
        self.assertEqual(weights["completed"], 0.7)
        self.assertEqual(weights["tool_success_rate"], 0.1)
        self.assertIn("answer_judge", weights)


if __name__ == "__main__":
    unittest.main()
