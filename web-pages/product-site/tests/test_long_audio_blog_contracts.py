"""Long-file documentation contracts, without acoustic models or network access."""

import ast
import copy
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from bs4 import BeautifulSoup

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = "funasr-transcribe-long-audio.html"
WHISPER_REVISION = "86098128c0b4f24f0e2aa2994de830614b474227"
FALSE_CLAIMS = (
    r"Whisper\s+(?:caps?\s+at|限)\s*30\s*(?:s\b|seconds?\b|秒)",
    r"(?:you\s+(?:have|need)\s+to|a\s+1-hour\s+file\s+means)\s+(?:manually\s+)?chunk(?:ing)?",
    r"你(?:得|必须|需要)自己把音频切片",
    r"(?:ingests?\s+audio\s+of|handles?)\s+any\s+length",
    r"(?:吃下|处理)\s*(?:任意|无限)时长",
    r"(?:GPU\s+memory\s+is\s+)?independent\s+of\s+total\s+file\s+length",
    r"(?:显存占用\s*)?与文件总长无关",
    r"1\s*hour\s+uses\s+about\s+as\s+much\s+as\s+1\s*minute",
    r"1\s*小时和\s*1\s*分钟占用相近",
)
NEGATION = r"\b(?:not|no|never|cannot|doesn't|isn't)\b|不能|并非|不是|不保证|不代表|不意味着|没有"
HISTORICAL_NUMBER = r"\b791\b|\b4\.3\s*(?:s\b|秒)|\b186\s*[x倍]|\b2104\b"
MULTIPLICATION = r"multiply|multiplied|multiplication|times|product|×|\*|乘"
CPU_PER_SEGMENT = r"per.segment|one.*segment|individual.*segment|segments?\s+individually|逐段|单段|每段|按片段逐个"


@pytest.fixture(scope="module", params=("source", "built"))
def site_tree(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    output = tmp_path_factory.mktemp("long-audio-contracts")
    build(output)
    return output


def page(root, prefix):
    return BeautifulSoup((root / prefix / "blog" / SLUG).read_text(), "html.parser")


def visible_text(node):
    assert node is not None, "Missing visible long-audio explanation"
    for parent in (node, *node.parents):
        assert not parent.has_attr("hidden")
        assert parent.get("aria-hidden") != "true"
        assert not re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", parent.get("style", ""))
    text = " ".join(node.get_text(" ", strip=True).split())
    assert text, "A marker without readable content is not a contract"
    return text


def contract(soup, name):
    return soup.select_one(f'article [data-long-audio-contract="{name}"]')


def require_concepts(text, *patterns):
    for pattern in patterns:
        assert re.search(pattern, text, re.I), (pattern, text)


def reject_positive_false_claims(text):
    text = " ".join(text.split())
    for pattern in FALSE_CLAIMS:
        for match in re.finditer(pattern, text, re.I):
            # A negation from a previous sentence/clause must not excuse a new claim.
            before = re.split(r"[.!?。！？;；]|\bbut\b|\bhowever\b|但是|但", text[:match.start()], flags=re.I)[-1]
            if re.search(NEGATION, before[-65:], re.I):
                continue
            pytest.fail(f"Unqualified long-audio claim: {match.group()}")


def promotional_texts(soup):
    yield soup.title.get_text() if soup.title else ""
    yield visible_text(soup.select_one("article h1"))
    for meta in soup.select('meta[name="description"], meta[name="keywords"], meta[property="og:title"], meta[property="og:description"]'):
        yield meta.get("content", "")
    for script in soup.select('script[type="application/ld+json"]'):
        records = json.loads(script.get_text())
        for record in records if isinstance(records, list) else [records]:
            for key in ("headline", "description"):
                yield record.get(key, "")


def recipe_tree(soup):
    blocks = [pre for pre in soup.select("article pre")
              if re.search(r"from\s+funasr\s+import\s+AutoModel", pre.get_text())]
    assert len(blocks) == 1, "Retain one complete, unambiguous Python SDK recipe"
    return ast.parse(blocks[0].get_text())


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_claims_are_bounded_in_article_seo_and_actual_index_card(site_tree, prefix):
    soup = page(site_tree, prefix)
    texts = [*promotional_texts(soup), visible_text(soup.select_one("article"))]
    index = BeautifulSoup((site_tree / prefix / "blog/index.html").read_text(), "html.parser")
    cards = index.select(f'a.post-card[href="/{prefix}blog/{SLUG}"]')
    assert len(cards) == 1
    texts.append(visible_text(cards[0]))
    for text in texts:
        reject_positive_false_claims(text)


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_whisper_window_is_distinguished_from_file_and_hosted_limits(site_tree, prefix):
    text = visible_text(contract(page(site_tree, prefix), "window"))
    require_concepts(text, r"Whisper", r"transcribe", r"30[\s-]*(?:second|s\b|秒)",
                     r"internal|internally|内部", r"window|窗口", r"file|文件", NEGATION,
                     r"hosted|cloud|托管|云", r"upload|上传")


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_resources_distinguish_full_decode_batches_and_streaming(site_tree, prefix):
    text = visible_text(contract(page(site_tree, prefix), "resources"))
    require_concepts(text, r"CPU|RAM|host|主机|内存", r"full|whole|entire|完整|整段|整个",
                     r"decode|waveform|解码|波形")
    require_concepts(text, r"GPU|VRAM|显存", r"batch|批", r"model|模型", NEGATION,
                     r"stream|流式", r"offline|file|离线|文件",
                     r"duration|length|时长|长度", r"limit|bound|上限|边界|限制|有界")


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_batch_and_endpoint_units_match_the_specific_cpu_recipe(site_tree, prefix):
    text = visible_text(contract(page(site_tree, prefix), "batching"))
    require_concepts(text, r"max_single_segment_time", r"30000|30,000",
                     r"millisecond|\bms\b|毫秒", r"threshold|端点|阈值", NEGATION,
                     r"sentence|phoneme|linguistic|句|音素|语言")
    require_concepts(text, r"batch_size_s", r"\b300\b", r"second|秒",
                     r"longest|max|最长", r"count|number|数量|段数", MULTIPLICATION)
    require_concepts(text, r"CPU", CPU_PER_SEGMENT,
                     r"throughput|performance|吞吐|性能", NEGATION)


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_prepared_sdk_environment_is_not_an_install_or_inference_guarantee(site_tree, prefix):
    text = visible_text(contract(page(site_tree, prefix), "prerequisites"))
    require_concepts(text, r"1\.4\.15", r"prepared|ready|准备|配置",
                     r"separate|isolated|独立|隔离", r"torch", r"audio|音频",
                     r"model|weight|模型|权重", r"inference|transcription|推理|转写", NEGATION)


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_immutable_sources_and_same_language_acceptance_links(site_tree, prefix):
    soup = page(site_tree, prefix)
    hrefs = [link["href"] for link in soup.select("article a[href]") if visible_text(link)]
    for path in ("README.md", "whisper/transcribe.py"):
        assert any(urlsplit(h).netloc == "github.com" and urlsplit(h).path ==
                   f"/openai/whisper/blob/{WHISPER_REVISION}/{path}" for h in hrefs)
    for path in ("funasr/auto/auto_model.py", "funasr/utils/load_utils.py"):
        assert any(re.fullmatch(r"/modelscope/FunASR/blob/[0-9a-f]{40}/" + re.escape(path),
                                urlsplit(h).path) and urlsplit(h).netloc == "github.com" for h in hrefs)
    for route in (f"/{prefix}docs/installation.html", f"/{prefix}docs/python-api.html",
                  f"/{prefix}blog/meeting-transcript-acceptance.html"):
        links = [h for h in hrefs if not urlsplit(h).netloc and urlsplit(h).path == route]
        assert links, route
        if site_tree != SITE / "legacy":
            target = BeautifulSoup((site_tree / route.lstrip("/")).read_text(), "html.parser")
            for href in links:
                fragment = unquote(urlsplit(href).fragment)
                if fragment:
                    assert target.find(id=fragment), (route, fragment)
    for rel in ("canonical",):
        assert soup.find("link", rel=rel)["href"] == f"https://www.funasr.com/{prefix}blog/{SLUG}"
    for lang, route_prefix in (("zh", ""), ("en", "en/")):
        rendered_lang = "zh-CN" if lang == "zh" and site_tree != SITE / "legacy" else lang
        alternates = soup.find_all("link", rel="alternate", hreflang=rendered_lang)
        assert len(alternates) == 1, (rendered_lang, alternates)
        assert alternates[0]["href"] == f"https://www.funasr.com/{route_prefix}blog/{SLUG}"


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_historical_numbers_are_not_promoted_as_new_capacity_or_coverage(site_tree, prefix):
    soup = page(site_tree, prefix)
    index = BeautifulSoup((site_tree / prefix / "blog/index.html").read_text(), "html.parser")
    card = index.select_one(f'a.post-card[href="/{prefix}blog/{SLUG}"]')
    for text in [*promotional_texts(soup), visible_text(card)]:
        assert not re.search(HISTORICAL_NUMBER, text, re.I), text
    article = copy.deepcopy(soup.select_one("article"))
    history = article.select_one('[data-long-audio-contract="evidence"]')
    if history and re.search(HISTORICAL_NUMBER, visible_text(history), re.I):
        text = visible_text(history)
        require_concepts(text, r"historical|archiv|历史", r"unverified|not.*verif|未.*核|未.*验证",
                         r"coverage|tail|覆盖|尾", NEGATION)
        history.decompose()
    assert not re.search(HISTORICAL_NUMBER, article.get_text(" ", strip=True), re.I)


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_recipe_has_explicit_model_vad_device_and_matching_units(site_tree, prefix):
    tree = recipe_tree(page(site_tree, prefix))
    constructors = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "AutoModel"]
    assert len(constructors) == 1
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in constructors[0].keywords}
    assert kwargs["model"] in ("iic/SenseVoiceSmall", "SenseVoiceSmall")
    assert kwargs["vad_model"] == "fsmn-vad"
    assert kwargs.get("device") == "cpu", "The demonstrated device must not depend on host GPU availability"
    assert kwargs["vad_kwargs"]["max_single_segment_time"] == 30000
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "generate"]
    assert len(calls) == 1
    args = {kw.arg: kw.value for kw in calls[0].keywords}
    assert ast.literal_eval(args["batch_size_s"]) == 300
    assert "input" in args


def test_bilingual_python_recipe_has_identical_ast(site_tree):
    assert ast.dump(recipe_tree(page(site_tree, ""))) == ast.dump(recipe_tree(page(site_tree, "en/")))


@pytest.mark.parametrize("prefix", ("", "en/"))
@pytest.mark.parametrize("result", ([], [{"text": ""}], [{"text": "controlled transcript"}]))
def test_actual_recipe_handles_empty_results_without_loading_models(site_tree, prefix, result, capsys, tmp_path, monkeypatch):
    tree = recipe_tree(page(site_tree, prefix))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    model_imports = [node for node in imports if isinstance(node, ast.ImportFrom) and node.module == "funasr"]
    assert len(model_imports) == 1
    assert [(n.name, n.asname) for n in model_imports[0].names] == [("AutoModel", None)]
    for node in imports:
        if node is model_imports[0]:
            continue
        if isinstance(node, ast.ImportFrom):
            assert node.module in ("pathlib", "importlib.metadata")
        else:
            assert all(n.name in ("pathlib", "importlib.metadata") for n in node.names)
    # Keep real Path/file checks and control flow; isolate only model and metadata boundaries.
    tree.body.remove(model_imports[0])
    monkeypatch.chdir(tmp_path)
    (tmp_path / "meeting.wav").write_bytes(b"")
    metadata_calls = []
    real_version = importlib.metadata.version

    def controlled_version(name):
        metadata_calls.append(name)
        return "1.4.15" if name == "funasr" else real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", controlled_version)
    calls = []

    class ControlledModel:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return copy.deepcopy(result)

    code = compile(tree, SLUG, "exec")
    namespace = {"AutoModel": ControlledModel, "__name__": "__main__"}
    if not result or not result[0]["text"]:
        with pytest.raises((ValueError, RuntimeError, SystemExit)) as rejected:
            exec(code, namespace)
        assert str(rejected.value), "Empty results need an explicit, readable failure"
    else:
        exec(code, namespace)
    assert "funasr" in metadata_calls
    assert [name for name, _ in calls] == ["init", "generate"]
    audio = calls[-1][1]["input"]
    assert isinstance(audio, str) and not urlsplit(audio).scheme and Path(audio).is_file()
    output = capsys.readouterr().out
    if result and result[0]["text"]:
        assert result[0]["text"] in output, "Do not make empty handling pass by discarding successful output"


@pytest.mark.parametrize("text", (
    "Whisper caps at 30s; manually split the file.",
    "FunASR ingests audio of any length.",
    "GPU memory is independent of total file length.",
    "显存占用与文件总长无关。",
    "No benchmark was run. FunASR handles any length.",
))
def test_false_claim_detector_rejects_positive_promises(text):
    with pytest.raises(pytest.fail.Exception):
        reject_positive_false_claims(text)


@pytest.mark.parametrize("text", (
    "Whisper transcribe internally processes 30-second windows, not a 30-second file limit.",
    "It is not true that FunASR handles any length.",
    "Do not assume GPU memory is independent of total file length.",
    "不能保证显存占用与文件总长无关。",
    "Full-file CPU decoding remains; smaller ASR batches do not turn this into streaming.",
))
def test_false_claim_detector_accepts_explicit_corrections(text):
    reject_positive_false_claims(text)


@pytest.mark.parametrize("pattern,text", (
    (MULTIPLICATION, "longest duration multiplied by the number of segments"),
    (CPU_PER_SEGMENT, "CPU handles segments individually"),
    (CPU_PER_SEGMENT, "CPU 按片段逐个处理"),
))
def test_equivalent_batching_grammar_is_accepted(pattern, text):
    require_concepts(text, pattern)
