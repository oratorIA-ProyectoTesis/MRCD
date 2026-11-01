"""Unit tests for sus_score.py. Run: python -m pytest docs/sus-kit/test_sus_score.py -p no:cacheprovider
(or: python -m unittest discover -s docs/sus-kit -p "test_*.py"). Pure Python, no data, negligible CPU."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sus_score as s  # noqa: E402


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["participant_id", *s.ITEM_COLUMNS])
        w.writerows(rows)


class TestScore(unittest.TestCase):
    def test_best_possible(self):
        # odd items 5, even items 1 -> every contribution is 4 -> 40 * 2.5 = 100
        self.assertEqual(s.sus_score([5, 1, 5, 1, 5, 1, 5, 1, 5, 1]), 100.0)

    def test_worst_possible(self):
        self.assertEqual(s.sus_score([1, 5, 1, 5, 1, 5, 1, 5, 1, 5]), 0.0)

    def test_all_neutral(self):
        # 3 everywhere: odd -> 2, even -> 2 -> 20 * 2.5 = 50
        self.assertEqual(s.sus_score([3] * 10), 50.0)

    def test_odd_even_asymmetry(self):
        # all 5: odd contribute 4 each (5 items = 20), even contribute 0 -> 50
        self.assertEqual(s.sus_score([5] * 10), 50.0)
        # all 1: odd contribute 0, even contribute 4 each (5 items = 20) -> 50
        self.assertEqual(s.sus_score([1] * 10), 50.0)

    def test_known_example(self):
        a = [
            4,
            2,
            4,
            2,
            4,
            2,
            4,
            2,
            4,
            2,
        ]  # odd: 3 each (15), even: 3 each (15) -> 30 * 2.5 = 75
        self.assertEqual(s.sus_score(a), 75.0)

    def test_scale_bounds(self):
        import random

        rng = random.Random(1)
        for _ in range(500):
            v = s.sus_score([rng.randint(1, 5) for _ in range(10)])
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 100.0)
            self.assertEqual(v % 2.5, 0.0)

    def test_invalid_inputs(self):
        for bad in (
            [3] * 9,
            [3] * 11,
            [0] + [3] * 9,
            [6] + [3] * 9,
            [3.5] + [3] * 9,
            [True] + [3] * 9,
        ):
            with self.assertRaises(ValueError):
                s.sus_score(bad)


class TestSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "r.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_template_is_not_run_not_zero(self):
        write_csv([["P01"] + [""] * 10], self.path)
        r = s.summarize(self.path)
        self.assertEqual(r["status"], "NOT_RUN")
        self.assertIsNone(r["mean"])
        self.assertIsNone(r["meets_threshold"])
        self.assertEqual(r["n_scored"], 0)

    def test_incomplete_row_excluded_with_reason(self):
        write_csv(
            [
                ["P01", 5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
                ["P02", 5, 1, "", 1, 5, 1, 5, 1, 5, 1],
            ],
            self.path,
        )
        r = s.summarize(self.path)
        self.assertEqual(r["n_scored"], 1)
        self.assertIn("P02", r["excluded"])
        self.assertIn("item_3", r["excluded"]["P02"])

    def test_below_min_users_threshold_not_evaluated(self):
        write_csv(
            [[f"P{i:02d}", 5, 1, 5, 1, 5, 1, 5, 1, 5, 1] for i in range(9)], self.path
        )
        r = s.summarize(self.path, threshold=70, min_users=10)
        self.assertEqual(r["n_scored"], 9)
        self.assertEqual(r["mean"], 100.0)
        self.assertIsNone(r["meets_threshold"])

    def test_ten_users_threshold_evaluated(self):
        write_csv(
            [[f"P{i:02d}", 4, 2, 4, 2, 4, 2, 4, 2, 4, 2] for i in range(10)], self.path
        )
        r = s.summarize(self.path, threshold=70, min_users=10)
        self.assertEqual(r["mean"], 75.0)
        self.assertTrue(r["meets_threshold"])
        write_csv(
            [[f"P{i:02d}", 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] for i in range(10)], self.path
        )
        self.assertFalse(s.summarize(self.path)["meets_threshold"])

    def test_duplicate_participant_not_double_counted(self):
        write_csv(
            [
                ["P01", 5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
                ["P01", 1, 5, 1, 5, 1, 5, 1, 5, 1, 5],
            ],
            self.path,
        )
        r = s.summarize(self.path)
        self.assertEqual(r["n_scored"], 1)
        self.assertEqual(r["n_excluded"], 1)

    def test_missing_column_rejected(self):
        self.path.write_text("participant_id,item_1\nP01,3\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            s.summarize(self.path)

    def test_ci_deterministic_with_seed(self):
        vals = [60.0, 70.0, 80.0, 90.0]
        self.assertEqual(s.bootstrap_ci(vals, seed=3), s.bootstrap_ci(vals, seed=3))
        self.assertIsNone(s.bootstrap_ci([50.0]))


if __name__ == "__main__":
    unittest.main()
