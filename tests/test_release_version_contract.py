from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "1.4.6"
README_FILES = ("README.md", "README_zh.md", "README_ja.md", "README_ko.md")


def test_release_version_uses_required_carry_rule():
    version = (ROOT / "funasr" / "version.txt").read_text(encoding="utf-8").strip()
    parts = version.split(".")

    assert version == RELEASE_VERSION
    assert len(parts) == 3
    assert all(part.isdecimal() for part in parts)
    assert all(0 <= int(part) <= 30 for part in parts)


def test_release_is_recorded_in_all_top_level_readmes():
    for name in README_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")

        assert f"funasr=={RELEASE_VERSION}" in text
        assert f"releases/tag/v{RELEASE_VERSION}" in text
