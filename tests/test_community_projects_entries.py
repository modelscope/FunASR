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


def test_speech_to_speech_source_install_is_discoverable_and_release_bounded():
    revision = "9e2ed1099190a4e4bc8a972b4a3488949ff1b9f6"
    for suffix in ("", "_zh"):
        text = (ROOT / f"docs/community_projects{suffix}.md").read_text()
        assert 'id="speech-to-speech"' in text
        assert "https://github.com/huggingface/speech-to-speech/pull/319" in text
        assert f"git checkout --detach {revision}" in text
        assert 'python -m pip install -e ".[sensevoice]"' in text
        assert "speech-to-speech serve --stt sense-voice" in text
        assert "--sense_voice_stt_device cpu" in text
        assert "--sense_voice_stt_model_name FunAudioLLM/SenseVoiceSmall" in text
        assert "--help" in text
        assert "v1.0.0" in text and "2026-09-30" in text
        readme = (ROOT / f"README{suffix}.md").read_text()
        assert f"./docs/community_projects{suffix}.md#speech-to-speech" in readme
    row = next(line for line in (ROOT / "docs/community_growth_20k.md").read_text().splitlines()
               if "`huggingface/speech-to-speech#319`" in line)
    assert revision in row and "Merged" in row
    assert "Wait for maintainer review" not in row
