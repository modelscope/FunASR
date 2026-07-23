from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recent_merged_voice_input_integrations_are_listed():
    english = (ROOT / "docs/community_projects.md").read_text()
    chinese = (ROOT / "docs/community_projects_zh.md").read_text()

    for text in [english, chinese]:
        assert "crosswk/SayIt" in text
        assert "crosswk/SayIt/pull/23" in text
        assert "LeonardNJU/VocoType-linux" in text
        assert "LeonardNJU/VocoType-linux/pull/32" in text


def test_recent_merged_subtitle_integrations_are_listed():
    english = (ROOT / "docs/community_projects.md").read_text()
    chinese = (ROOT / "docs/community_projects_zh.md").read_text()

    for text in [english, chinese]:
        assert "buxuku/SmartSub" in text
        assert "renderer/components/resources/FunasrModelSection.tsx" in text
        assert "buxuku/SmartSub/pull/392" in text


def test_recent_merged_discovery_lists_are_listed():
    english = (ROOT / "docs/community_projects.md").read_text()
    chinese = (ROOT / "docs/community_projects_zh.md").read_text()

    for text in [english, chinese]:
        assert "WangRongsheng/awesome-LLM-resources" in text
        assert "WangRongsheng/awesome-LLM-resources/pull/162" in text
