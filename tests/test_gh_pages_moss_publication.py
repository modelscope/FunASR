from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_github_pages_moss_source_is_versioned_and_discoverable():
    pages = [
        ROOT / "gh-pages-output" / "moss-transcribe-diarize.html",
        ROOT / "gh-pages-output" / "zh" / "moss-transcribe-diarize.html",
        ROOT / "gh-pages-output" / "model-selection.html",
        ROOT / "gh-pages-output" / "zh" / "model-selection.html",
        ROOT / "gh-pages-output" / "index.html",
        ROOT / "gh-pages-output" / "zh" / "index.html",
    ]
    for page in pages:
        assert page.is_file(), page

    english = (ROOT / "gh-pages-output" / "moss-transcribe-diarize.html").read_text()
    chinese = (ROOT / "gh-pages-output" / "zh" / "moss-transcribe-diarize.html").read_text()
    for page in (english, chinese):
        assert "OpenMOSS-Team/MOSS-Transcribe-Diarize" in page
        assert "OpenMOSS" in page
        assert "third-party" in page or "第三方" in page
        assert "funasr.com" in page
        assert "vLLM" in page
        assert "SGLang" in page

    for page in pages[2:]:
        content = page.read_text()
        assert "moss-transcribe-diarize.html" in content
        assert "MOSS-Transcribe-Diarize" in content

    workflow = (ROOT / ".github/workflows/update-api-docs.yml").read_text()
    assert "docs/moss_transcribe_diarize*.md" in workflow
    assert "model_zoo/**" in workflow
    assert "gh-pages-output/**" in workflow
