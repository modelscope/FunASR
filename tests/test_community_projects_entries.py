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


def test_openclaw_readme_entrypoints_disclose_unmerged_sdk_requirement():
    for suffix, marker in (("", "experimental; requires an unmerged upstream SDK patch"),
                           ("_zh", "\u5b9e\u9a8c\u6027\uff1b\u4f9d\u8d56\u5c1a\u672a\u5408\u5e76\u7684\u4e0a\u6e38 SDK \u8865\u4e01")):
        text = (ROOT / f"README{suffix}.md").read_text(encoding="utf-8")
        entry = next(line for line in text.splitlines() if "integrations/openclaw/" in line)
        assert "integrations/openclaw/#compatibility" in entry
        assert marker in entry


def test_openclaw_compatibility_records_closed_unmerged_dependency():
    text = (ROOT / "integrations/openclaw/README.md").read_text()
    compatibility = text.split("## Compatibility", 1)[1].split("\n## ", 1)[0]
    assert "2026-10-31" in compatibility
    assert "closed without merging" in compatibility
    assert "https://github.com/openclaw/openclaw/pull/118977" in compatibility
    assert "not a supported stock-install compatibility floor" in compatibility
    assert "npm and ClawHub" in compatibility
    assert "accepted and released upstream SDK contract" in compatibility


def test_openclaw_protocol_proof_is_not_released_host_acceptance():
    text = (ROOT / "integrations/openclaw/README.md").read_text()
    proof = text.split("## Verified behavior", 1)[1].split("\n## ", 1)[0]
    assert "historical local protocol proof" in proof
    assert "not validation of a released OpenClaw installation" in proof
    assert "278 audio frames" in proof
    assert "seven partial transcripts" in proof


def test_openclaw_growth_record_keeps_history_and_current_disposition_separate():
    text = (ROOT / "docs/community_growth_20k.md").read_text()
    section = text.split("### 2026-08-16 OpenClaw", 1)[1].split("\n### ", 1)[0]
    assert "154 successful, 39 skipped" in section
    assert "2026-10-31 disposition" in section
    assert "closed without merging" in section
    assert "#issuecomment-6542645321" in section
    assert "not a merged integration or adoption result" in section
