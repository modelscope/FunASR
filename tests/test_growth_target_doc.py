"""Keep the campaign's strict target distinct from an inclusive increment."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GrowthTargetDocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "docs/community_growth_20k.md").read_text(encoding="utf-8")

    def test_strict_target_and_collector_command(self):
        intro = self.text.split("\n\n")[1]
        self.assertIn("more than 20,000 additional stars", intro)
        self.assertIn("at least 51,225 combined stars", intro)
        self.assertIn("--ecosystem --baseline-stars 31224 --target-additional-stars 20001", self.text)
        self.assertIn("max(0, 51225 - total_stars)", self.text)
        self.assertIn("exactly 51,224 does not meet the strict target", self.text)

    def test_historical_snapshot_gap_uses_strict_target(self):
        snapshot = self.text.split("## Current campaign snapshot", 1)[1].split("\n\n")[1]
        total = re.search(r"has ([\d,]+) combined GitHub stars", snapshot)
        gap = re.search(r"another ([\d,]+) stars to reach at least ([\d,]+)", snapshot)
        self.assertIsNotNone(total)
        self.assertIsNotNone(gap)
        total_stars = int(total[1].replace(",", ""))
        remaining, target = (int(value.replace(",", "")) for value in gap.groups())
        self.assertEqual(target, 31224 + 20000 + 1)
        self.assertEqual(remaining, max(0, target - total_stars))

    def test_metric_and_chinese_summary_agree(self):
        self.assertIn("GitHub stars: more than +20,000", self.text)
        summary = self.text.split("## \u4e2d\u6587\u6458\u8981", 1)[1]
        self.assertIn("\u7d2f\u8ba1\u65b0\u589e\u8d85\u8fc7 20,000 stars", summary)
        self.assertIn("51,225", summary)


if __name__ == "__main__":
    unittest.main()
