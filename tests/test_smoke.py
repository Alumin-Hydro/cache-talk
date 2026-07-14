"""Smoke test for cache_talk pipeline."""
import tempfile
import unittest
from pathlib import Path

from cache_talk.eval import score_output


class EvalTest(unittest.TestCase):
    def test_score_output_detects_reasoning(self) -> None:
        text = "First we add 17 and 25. Then we get 42. The answer is 42."
        metrics = score_output(text)
        self.assertTrue(metrics.contains_reasoning_words)
        self.assertGreaterEqual(metrics.reasoning_word_count, 2)


class DataTest(unittest.TestCase):
    def test_tasks_json_is_valid(self) -> None:
        tasks_path = Path(__file__).parent.parent / "data" / "tasks.json"
        self.assertTrue(tasks_path.exists())
        import json
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        self.assertGreater(len(tasks), 0)
        for task in tasks:
            self.assertIn("id", task)
            self.assertIn("question", task)
            self.assertIn("answer", task)


if __name__ == "__main__":
    unittest.main()
