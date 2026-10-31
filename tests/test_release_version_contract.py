from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "1.4.18"
README_FILES = ("README.md", "README_zh.md", "README_ja.md", "README_ko.md")
REPOSITORY_ROLE_FILES = ("docs/repository_roles.md", "docs/repository_roles_zh.md")


@pytest.mark.parametrize("name,expected,incorrect", [
    ("README.md", "cross-script hotword misreplacement", "hotword-file loading"),
    ("README_zh.md", "\u8de8\u4e2d\u82f1\u6587\u70ed\u8bcd\u8bef\u66ff\u6362", "\u70ed\u8bcd\u6587\u4ef6\u8bfb\u53d6"),
    ("README_ja.md", "\u4e2d\u56fd\u8a9e\u3068\u30e9\u30c6\u30f3\u6587\u5b57\u304c\u6df7\u5728\u3059\u308b hotword \u306e\u8aa4\u7f6e\u63db", "hotword \u30d5\u30a1\u30a4\u30eb\u306e\u8aad\u307f\u8fbc\u307f"),
    ("README_ko.md", "\uc911\uad6d\uc5b4\uc640 \ub77c\ud2f4 \ubb38\uc790\uac00 \uc11e\uc778 hotword\uc758 \uc798\ubabb\ub41c \uce58\ud658", "hotword \ud30c\uc77c \uc77d\uae30"),
])
def test_release_hotword_summary_matches_cross_script_fix(name, expected, incorrect):
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    summaries = [line for line in lines if f"**FunASR {RELEASE_VERSION}**" in line]
    assert len(summaries) == 1
    assert expected in summaries[0]
    assert incorrect not in summaries[0]


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


def test_repository_role_guides_link_the_current_release():
    for name in REPOSITORY_ROLE_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")



def test_release_documents_do_not_describe_a_pending_candidate():
    candidate_labels = ("release candidate", "\u53d1\u5e03\u5019\u9009",
                        "\u30ea\u30ea\u30fc\u30b9\u5019\u88dc", "\ub9b4\ub9ac\uc2a4 \ud6c4\ubcf4")
    for name in README_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        version_lines = [line for line in text.splitlines() if RELEASE_VERSION in line]
        assert version_lines, name
        assert not any(label in line.lower() for line in version_lines
                       for label in candidate_labels), name
        assert 'pip install -U "funasr==1.4.17"' not in text, name
    for name in REPOSITORY_ROLE_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "funasr==1.4.17" not in text, name
        assert "not published yet" not in text, name
        assert "\u5c1a\u672a\u6b63\u5f0f\u53d1\u5e03" not in text, name
        assert not any(label in text.lower() for label in candidate_labels), name
        assert f"funasr=={RELEASE_VERSION}" in text
        assert f"releases/tag/v{RELEASE_VERSION}" in text
