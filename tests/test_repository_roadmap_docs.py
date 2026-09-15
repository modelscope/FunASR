"""Keep completed roadmap milestones separate from unresolved validation."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    ("repository_roles.md", "### Delivered", "### In progress", "### Next"),
    ("repository_roles_zh.md", "### \u5df2\u4ea4\u4ed8", "### \u8fdb\u884c\u4e2d", "### \u4e0b\u4e00\u6b65"),
)


class RepositoryRoadmapContract(unittest.TestCase):
    def sections(self, guide):
        name, delivered, active, following = guide
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        return (text.split(delivered, 1)[1].split(active, 1)[0],
                text.split(active, 1)[1].split(following, 1)[0])

    def test_native_transformers_merge_is_delivered(self):
        for guide in GUIDES:
            with self.subTest(guide=guide[0]):
                delivered, active = self.sections(guide)
                self.assertIn("huggingface/transformers/pull/46180", delivered)
                self.assertIn("FunAudioLLM/Fun-ASR-Nano-2512-hf", delivered)
                self.assertNotIn("huggingface/transformers/pull/46180", active)

    def test_reporter_closed_qwen_workflow_is_delivered(self):
        for guide in GUIDES:
            with self.subTest(guide=guide[0]):
                delivered, active = self.sections(guide)
                self.assertIn("FunASR/pull/3592", delivered)
                self.assertIn("FunASR/issues/3419", delivered)
                self.assertNotIn("FunASR/issues/3419", active)

    def test_unresolved_hardware_and_checkpoint_work_stays_open(self):
        for guide in GUIDES:
            with self.subTest(guide=guide[0]):
                _, active = self.sections(guide)
                for issue in (3496, 3528, 3479):
                    self.assertIn(f"FunASR/issues/{issue}", active)


if __name__ == "__main__":
    unittest.main()
