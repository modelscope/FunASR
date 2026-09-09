"""Exercise the downloadable article example, not an untested prose copy."""
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

SITE = Path(__file__).resolve().parents[1]
SCRIPT = SITE / "legacy/blog/transcript-audit.py"
SLUG = "meeting-transcript-acceptance.html"


@pytest.fixture
def audit():
    return runpy.run_path(str(SCRIPT))["audit"]


def segment(start=0, end=1000, spk="S01", text="hello"):
    return dict(start=start, end=end, spk=spk, text=text)


def test_reports_tail_without_claiming_missing_speech_or_quality(audit):
    report = audit({"sentence_info": [segment(end=469200)]}, 480240)
    assert report["tail_not_covered_ms"] == 11040
    assert report["quality_verified"] is False
    assert report["speaker_labels"] == ["S01"]


def test_overlap_and_cross_recording_labels_are_not_identity_proof(audit):
    data = [segment(0, 2000), segment(1000, 3000, "S02"), segment(4000, 5000)]
    result = audit({"sentence_info": data}, 5000)
    assert result["tail_not_covered_ms"] == 0
    assert result["overlap_indices"] == [1]
    assert result["quality_verified"] is False


def test_empty_and_unordered_segments_are_visible(audit):
    assert audit({"sentence_info": []}, 1000)["segment_count"] == 0
    result = audit({"sentence_info": [segment(500, 800), segment(0, 200, text=" ")]}, 1000)
    assert result["out_of_order_indices"] == [1]
    assert result["empty_text_indices"] == [1]
    assert result["last_end_ms"] == 800
    assert result["overlap_indices"] == []


def test_unordered_overlaps_preserve_original_indices(audit):
    data = [segment(1000, 3000), segment(0, 2000, "S02")]
    assert audit({"sentence_info": data}, 4000)["overlap_indices"] == [0]


def test_touching_boundaries_are_not_overlap(audit):
    data = [segment(1000, 2000), segment(0, 1000)]
    assert audit({"sentence_info": data}, 2000)["overlap_indices"] == []


@pytest.mark.parametrize("bad", [None, [], {}, {"sentence_info": None},
                                 {"sentence_info": [None]}, {"sentence_info": [{}]}])
def test_rejects_missing_schema(audit, bad):
    with pytest.raises(ValueError):
        audit(bad, 1000)


@pytest.mark.parametrize("duration", [True, 0, -1, "1000", float("nan"), float("inf"), 10**400])
def test_rejects_invalid_duration(audit, duration):
    with pytest.raises(ValueError):
        audit({"sentence_info": []}, duration)


@pytest.mark.parametrize("changes", [
    {"start": -1}, {"end": 0}, {"end": 1001}, {"start": True},
    {"end": float("inf")}, {"end": float("nan")}, {"start": "0"},
    {"spk": ""}, {"spk": 1}, {"text": None},
])
def test_rejects_bad_segments(audit, changes):
    with pytest.raises(ValueError):
        audit({"sentence_info": [segment(**changes)]}, 1000)


def test_cli_success_and_failure_do_not_modify_source(tmp_path):
    path = tmp_path / "result.json"
    original = json.dumps({"sentence_info": [segment()]}).encode()
    path.write_bytes(original)
    command = [sys.executable, str(SCRIPT), str(path), "--duration-ms", "1200"]
    good = subprocess.run(command, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["tail_not_covered_ms"] == 200
    bad = subprocess.run(command[:-1] + ["nan"], capture_output=True, text=True)
    assert bad.returncode != 0
    assert not bad.stdout
    assert path.read_bytes() == original


def test_articles_index_sources_example_and_asset_are_consistent(audit):
    snippets = []
    for language in ["zh", "en"]:
        prefix = "" if language == "zh" else "en/"
        page = SITE / "legacy" / prefix / "blog" / SLUG
        soup = BeautifulSoup(page.read_text(), "html.parser")
        metadata = json.loads(soup.select_one('script[type="application/ld+json"]').string)
        assert metadata["datePublished"] == "2026-09-09"
        assert metadata["dateModified"] == "2026-09-09"
        assert soup.select_one("h1")
        assert soup.select_one('a[href="/blog/transcript-audit.py"]')
        assert soup.select_one('a[href*="OpenMOSS/MOSS-Transcribe-Diarize"]')
        assert soup.select_one('a[href*="pyannote/pyannote-metrics"]')
        assert soup.select_one('a[href*="/deploy/moss-transcribe-diarize.html"]')
        example = json.loads(soup.select_one('[data-example="audit-input"]').get_text())
        result = audit(example, 12000)
        assert result["tail_not_covered_ms"] == 4000
        assert result["quality_verified"] is False
        snippets.append(example)
        index = BeautifulSoup((SITE / "legacy" / prefix / "blog/index.html").read_text(), "html.parser")
        assert index.select_one(f'a[href="/{prefix}blog/{SLUG}"]')
    assert snippets[0] == snippets[1]
