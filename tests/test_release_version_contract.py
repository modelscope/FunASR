from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_VERSION = "1.4.17"
README_FILES = ("README.md", "README_zh.md", "README_ja.md", "README_ko.md")
REPOSITORY_ROLE_FILES = ("docs/repository_roles.md", "docs/repository_roles_zh.md")


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
        assert 'pip install -U "funasr==1.4.16"' not in text, name
    for name in REPOSITORY_ROLE_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "funasr==1.4.16" not in text, name
        assert "not published yet" not in text, name
        assert "\u5c1a\u672a\u6b63\u5f0f\u53d1\u5e03" not in text, name
        assert not any(label in text.lower() for label in candidate_labels), name
        assert f"funasr=={RELEASE_VERSION}" in text
        assert f"releases/tag/v{RELEASE_VERSION}" in text
