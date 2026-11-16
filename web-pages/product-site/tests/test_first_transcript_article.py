"""Executable tutorial contracts, not pretrained speech-quality tests."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import wave

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = "speech-to-text-python-transcribe-audio"
MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
TITLES = ("用 Python 把第一段录音转成文字", "Transcribe your first recording with Python")


@pytest.fixture(scope="module", params=["source", "built"])
def root(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    dest = tmp_path_factory.mktemp("first-transcript-build")
    build(dest)
    return dest


@pytest.fixture(params=["", "en/"])
def article(root, request):
    prefix = request.param
    return prefix, BeautifulSoup((root / prefix / "blog" / (SLUG + ".html")).read_text(), "html.parser")


def test_catalogue():
    data = json.loads((SITE / "data/blog.json").read_text())
    row = next(x for x in data["articles"] if x["slug"] == SLUG)
    assert row["reviewed"] is True
    assert row["category"] == "applications"
    assert (row["zh"]["title"], row["en"]["title"]) == TITLES
    assert SLUG in [data["lead"], *data["selected"]]


def test_reader_and_agent_contract(article):
    prefix, soup = article
    assert soup.h1.get_text() == TITLES[bool(prefix)]
    metadata = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert metadata["datePublished"] == "2026-06-18"
    assert metadata["dateModified"] == "2026-11-17"
    assert metadata["headline"] == soup.h1.get_text()
    assert soup.select_one('link[rel="canonical"]')["href"] == f"https://www.funasr.com/{prefix}blog/{SLUG}.html"
    for section in ("ecosystem", "requirements", "installation", "audio", "transcribe", "result", "troubleshooting", "agent-contract", "verification"):
        assert soup.select_one(f"article #{section}"), section
    text = soup.select_one("article").get_text()
    for claim in ("代码全部实测", "每段代码都已实测", "换模型只改", "CPU 也能实时", "All code tested", "just change model="):
        assert claim not in text
    assert "1.4.26" in text and "30" in text and "16000" in text
    assert soup.select_one('article [data-editorial="next-step"] a[href]')
    assert not soup.select("article .post-list")
    assert "pretrained" in soup.select_one("#verification").parent.get_text().lower() if prefix else "预训练" in soup.select_one("#verification").parent.get_text()


@pytest.fixture
def recipe(article):
    _, soup = article
    node = soup.select_one('code[data-example="transcribe-first"]')
    assert node is not None
    ns = {"__name__": "tutorial_example"}
    exec(compile(node.get_text(), "article/transcribe_first.py", "exec"), ns)
    return ns["main"]


def wav_file(tmp_path, rate=16000, channels=1, seconds=1, width=2):
    p = tmp_path / "sample.wav"
    with wave.open(str(p), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(b"\x00" * int(rate * seconds) * channels * width)
    return p


@pytest.fixture
def fake_model(monkeypatch):
    calls = []
    result = [{"text": "测试结果"}]
    class Model:
        model_path = "/reviewed/model"
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs))
        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return copy.deepcopy(result)
    module = types.ModuleType("funasr")
    module.AutoModel = Model
    module.__version__ = "1.4.26"
    monkeypatch.setitem(sys.modules, "funasr", module)
    return calls, result


def args_for(tmp_path, audio):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    return ["--audio", str(audio), "--model-dir", str(model), "--output", str(tmp_path / "transcript.json")]


def test_local_model_output_and_immutable_input(recipe, tmp_path, fake_model):
    audio = wav_file(tmp_path)
    before = hashlib.sha256(audio.read_bytes()).hexdigest()
    args = args_for(tmp_path, audio)
    recipe(args)
    calls, _ = fake_model
    assert calls == [
        ("construct", dict(model=str(tmp_path / "model"), hub="ms", device="cpu", ncpu=4, disable_update=True, trust_remote_code=False)),
        ("generate", dict(input=str(audio), batch_size=1)),
    ]
    out = json.loads((tmp_path / "transcript.json").read_text())
    assert out["text"] == "测试结果" and out["funasr_version"] == "1.4.26"
    assert out["audio"] == str(audio) and out["model_dir"] == "/reviewed/model"
    assert out["duration_seconds"] == 1.0
    assert hashlib.sha256(audio.read_bytes()).hexdigest() == before


def test_download_requires_explicit_opt_in(recipe, tmp_path, fake_model):
    audio = wav_file(tmp_path)
    with pytest.raises(SystemExit) as e:
        recipe(["--audio", str(audio), "--output", str(tmp_path / "out.json")])
    assert e.value.code == 2 and fake_model[0] == []
    recipe(["--audio", str(audio), "--allow-download", "--output", str(tmp_path / "out.json")])
    assert fake_model[0][0][1]["model"] == MODEL


@pytest.mark.parametrize("case", ["missing", "not-wav", "stereo", "rate", "empty", "long", "width", "truncated", "model-missing", "output-exists", "output-parent"])
def test_invalid_inputs_stop_before_model_load(recipe, tmp_path, fake_model, case):
    options = {"stereo": {"channels": 2}, "rate": {"rate": 8000}, "empty": {"seconds": 0}, "long": {"seconds": 31}, "width": {"width": 1}}
    audio = wav_file(tmp_path, **options.get(case, {}))
    args = args_for(tmp_path, audio)
    if case == "missing":
        audio.unlink()
    elif case == "not-wav":
        audio.write_bytes(b"not a wav")
    elif case == "truncated":
        audio.write_bytes(audio.read_bytes()[:-4])
    elif case == "model-missing":
        (tmp_path / "model").rmdir()
    elif case == "output-exists":
        (tmp_path / "transcript.json").write_text("KEEP")
    elif case == "output-parent":
        args[-1] = str(tmp_path / "absent" / "out.json")
    with pytest.raises((SystemExit, OSError, ValueError, EOFError, wave.Error)):
        recipe(args)
    assert fake_model[0] == []
    if case == "output-exists":
        assert (tmp_path / "transcript.json").read_text() == "KEEP"
    else:
        assert not (tmp_path / "transcript.json").exists()


@pytest.mark.parametrize("response", [[], [{}], [{"text": ""}], [{"text": 12}], [{"text": " "}], [{"text": "one"}, {"text": "two"}]])
def test_invalid_results_do_not_create_success_file(recipe, tmp_path, fake_model, response):
    audio = wav_file(tmp_path)
    fake_model[1][:] = response
    with pytest.raises((SystemExit, ValueError, RuntimeError)):
        recipe(args_for(tmp_path, audio))
    assert not (tmp_path / "transcript.json").exists()


def test_bilingual_code_is_identical(root):
    codes = []
    for prefix in ("", "en/"):
        soup = BeautifulSoup((root / prefix / "blog" / (SLUG + ".html")).read_text(), "html.parser")
        codes.append(soup.select_one('code[data-example="transcribe-first"]').get_text())
    assert codes[0] == codes[1]
