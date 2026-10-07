from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adops import dataio, metrics as M, scenarios as SC
from adops.env import AdAccountEnv


class TestCsvRoundTrip(unittest.TestCase):
    def setUp(self):
        self.env = AdAccountEnv(SC.get("S3_misallocation"), seed=2)
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "perf.csv")

    def test_round_trip_preserves_aggregates(self):
        dataio.write_csv(self.env.rows, self.path)
        rows, problems = dataio.read_csv(self.path)
        self.assertEqual(problems, [])
        self.assertEqual(len(rows), len(self.env.rows))
        before, after = M.aggregate(self.env.rows), M.aggregate(rows)
        self.assertAlmostEqual(before["spend"], after["spend"], places=2)
        self.assertAlmostEqual(before["cpa"], after["cpa"], places=4)

    def test_missing_columns_reported(self):
        with open(self.path, "w") as fh:
            fh.write("period,spend\n1,10\n")
        rows, problems = dataio.read_csv(self.path)
        self.assertEqual(rows, [])
        self.assertIn("missing required columns", problems[0])

    def test_impossible_rows_are_rejected(self):
        dataio.write_csv(self.env.rows[:3], self.path)
        with open(self.path) as fh:
            lines = fh.read().splitlines()
        parts = lines[1].split(",")
        header = lines[0].split(",")
        parts[header.index("clicks")] = "99999999"
        lines[1] = ",".join(parts)
        with open(self.path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        rows, problems = dataio.read_csv(self.path)
        self.assertEqual(len(rows), 2)
        self.assertTrue(any("exceed impressions" in p for p in problems))

    def test_supplied_derived_metrics_are_not_trusted(self):
        dataio.write_csv(self.env.rows[:2], self.path)
        with open(self.path) as fh:
            lines = fh.read().splitlines()
        header = lines[0].split(",")
        parts = lines[1].split(",")
        parts[header.index("cpa")] = "0.01"
        lines[1] = ",".join(parts)
        with open(self.path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        rows, problems = dataio.read_csv(self.path)
        self.assertEqual(len(rows), 2)            # row kept
        self.assertTrue(any("disagrees with the recomputed value" in p for p in problems))
        self.assertNotIn("cpa", rows[0])          # derived columns never stored

    def test_summarise(self):
        s = dataio.summarise(self.env.rows)
        self.assertEqual(s["ad_sets"], 7)
        self.assertEqual(s["campaigns"], 3)
        self.assertGreater(s["spend"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
