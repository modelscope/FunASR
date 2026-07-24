import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_GITHUB_REPO = re.compile(
    r"FunAudioLLM/(?:CosyVoice|Fun-ASR|SenseVoice)(?![A-Za-z0-9_-])"
)
LEGACY_GITHUB_OWNER_URL = re.compile(
    r"https://(?:github\.com|img\.shields\.io/github/stars)/FunAudioLLM/"
)
HUGGING_FACE_URL = re.compile(
    r"https://huggingface\.co/(?:spaces/)?FunAudioLLM/[^\s)>'\"]+"
)


def _tracked_text_files():
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = ROOT / raw_path.decode()
        try:
            yield path, path.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError, IsADirectoryError):
            continue


def test_legacy_owner_matcher_covers_repository_api_and_charts():
    legacy_examples = (
        "https://api.github.com/repos/FunAudioLLM/CosyVoice",
        "https://api.star-history.com/svg?repos=FunAudioLLM/CosyVoice&type=Date",
        "https://github.com/FunAudioLLM/CosyVoice",
        "https://img.shields.io/github/stars/FunAudioLLM/CosyVoice",
        "FunAudioLLM/Fun-ASR",
        "FunAudioLLM/SenseVoice",
    )
    for text in legacy_examples:
        without_hugging_face_urls = HUGGING_FACE_URL.sub("", text)
        assert LEGACY_GITHUB_OWNER_URL.search(text) or LEGACY_GITHUB_REPO.search(
            without_hugging_face_urls
        )

    allowed_examples = (
        "https://huggingface.co/FunAudioLLM/CosyVoice",
        "https://huggingface.co/spaces/FunAudioLLM/SenseVoice",
        "FunAudioLLM/CosyVoice-300M",
        "FunAudioLLM/Fun-ASR-Nano-2512",
        "FunAudioLLM/SenseVoiceSmall",
    )
    for text in allowed_examples:
        without_hugging_face_urls = HUGGING_FACE_URL.sub("", text)
        assert not LEGACY_GITHUB_OWNER_URL.search(text)
        assert not LEGACY_GITHUB_REPO.search(without_hugging_face_urls)


def test_tracked_files_use_qwenaudio_for_canonical_github_repositories():
    offenders = []
    for path, text in _tracked_text_files():
        if path.resolve() == Path(__file__).resolve():
            continue
        without_hugging_face_urls = HUGGING_FACE_URL.sub("", text)
        if LEGACY_GITHUB_OWNER_URL.search(text) or LEGACY_GITHUB_REPO.search(
            without_hugging_face_urls
        ):
            offenders.append(str(path.relative_to(ROOT)))

    assert not offenders, "legacy GitHub repository owner in: " + ", ".join(offenders)
