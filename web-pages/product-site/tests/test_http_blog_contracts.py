"""Keep legacy HTTP articles consistent with the maintained packaged API."""

import ast
import json
import re
import shlex
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from bs4 import BeautifulSoup

SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parents[1]
sys.path.insert(0, str(SITE))
from build import build


SLUGS = (
    "self-hosted-openai-whisper-api-alternative.html",
    "self-hosted-deepgram-assemblyai-alternative.html",
)
PAGES = [(prefix, slug) for prefix in ("", "en/") for slug in SLUGS]
NEGATIVE = r"\b(?:no|not|without|unsupported|unimplemented)\b|不|未|没有|缺少"
FORBIDDEN = (
    r"接口完全一致",
    r"\bidentical\s+interface\b",
    r"\binterface\s+(?:is\s+)?identical\b",
    r"(?:只改|只需(?:修改|更改)|仅改|仅需(?:修改|更改)).{0,20}base[_ ]url.{0,15}(?:即可|就能|便可)",
    r"(?:just|only)\s+(?:change|changing)\s+(?:a\s+single\s+|the\s+)?base[_ ]url",
    r"changing\s+(?:only\s+|a\s+single\s+)base[_ ]url",
    r"\beverything\s+else\s*:\s*the\s+client\.audio\.transcriptions\.create.*?same",
    r"其余不变.*?client\.audio\.transcriptions\.create",
    r"(?:usage\s+bills?\s+go(?:es)?\s+to\s+zero|按量账单归零)",
)


@pytest.fixture(scope="module", params=("source", "built"))
def site_tree(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    output = tmp_path_factory.mktemp("http-blog-contracts")
    build(output)
    return output


def read_page(root, prefix, slug):
    return BeautifulSoup((root / prefix / "blog" / slug).read_text(), "html.parser")


def visible_text(node):
    assert node is not None, "Missing visible contract explanation"
    for ancestor in [node, *node.parents]:
        assert not ancestor.has_attr("hidden")
        assert ancestor.get("aria-hidden") != "true"
        assert not re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", ancestor.get("style", ""))
    text = " ".join(node.get_text(" ", strip=True).split())
    assert text, "A data attribute alone is not a contract explanation"
    return text


def startup_commands(soup):
    commands = []
    for pre in soup.select("article pre"):
        for line in pre.get_text().replace("\\\n", " ").splitlines():
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
            lexer.whitespace_split = True
            tokens = list(lexer)
            for index, token in enumerate(tokens):
                if Path(token).name != "funasr-server":
                    continue
                command = []
                for argument in tokens[index:]:
                    if argument and set(argument) <= set(";&|()"):
                        break
                    command.append(argument)
                commands.append((pre, command))
    assert commands, "Retain at least one usable local startup recipe"
    return commands


def assert_loopback(command):
    hosts = []
    for index, argument in enumerate(command):
        if argument == "--host":
            assert index + 1 < len(command), command
            hosts.append(command[index + 1])
        elif argument.startswith("--host="):
            hosts.append(argument.partition("=")[2])
    assert hosts == ["127.0.0.1"], command


def prose_and_metadata(soup):
    chunks = [soup.title.get_text() if soup.title else ""]
    chunks.extend(node.get("content", "") for node in soup.select(
        'meta[name="description"], meta[property="og:description"], meta[property="og:title"]'
    ))
    for script in soup.select('script[type="application/ld+json"]'):
        data = json.loads(script.get_text())
        records = data if isinstance(data, list) else [data]
        for record in records:
            chunks.extend(str(record.get(key, "")) for key in ("headline", "description"))
    chunks.append(soup.select_one("article").get_text(" ", strip=True))
    return [" ".join(chunk.split()) for chunk in chunks]


def assert_no_blanket_claims(text):
    for pattern in FORBIDDEN:
        for match in re.finditer(pattern, text, re.I):
            # Explicit rejection of an old claim is not a positive promise.
            before = text[max(0, match.start() - 35):match.start()]
            if re.search(r"(?:not|never)\s+(?:an?\s+)?$|(?:并非|不是|不能|不意味着)\s*$", before, re.I):
                continue
            pytest.fail(f"Unqualified migration claim: {match.group(0)}")


@pytest.mark.parametrize("prefix,slug", PAGES)
def test_all_local_startup_commands_explicitly_bind_loopback(site_tree, prefix, slug):
    soup = read_page(site_tree, prefix, slug)
    for _, command in startup_commands(soup):
        assert_loopback(command)


@pytest.mark.parametrize("prefix,slug", PAGES)
def test_security_boundary_precedes_first_startup(site_tree, prefix, slug):
    soup = read_page(site_tree, prefix, slug)
    warning = soup.select_one('article [data-http-contract="security"]')
    visible_text(warning)
    first_pre = startup_commands(soup)[0][0]
    order = {id(node): index for index, node in enumerate(soup.find_all(True))}
    for risk, concept in (
        ("authentication", r"auth|认证|鉴权"),
        ("upload-limit", r"upload|request.body|上传|请求体"),
        ("backend-bypass", r"bypass|绕过"),
    ):
        item = warning.select_one(f'[data-risk="{risk}"]')
        text = visible_text(item)
        assert re.search(concept, text, re.I), (risk, text)
        assert order[id(item)] < order[id(first_pre)], "Security explanation must precede startup"
        if risk != "backend-bypass":
            assert re.search(NEGATIVE, text, re.I), (risk, text)
            assert re.search(r"built.in|implement|FunASR|内置|实现", text, re.I), text
            if risk == "upload-limit":
                assert re.search(r"limit|cap|上限|限制", text, re.I), text
        else:
            assert re.search(r"backend|back.end|后端", text, re.I), text
            assert re.search(r"gateway|proxy|网关|代理", text, re.I), text


@pytest.mark.parametrize("prefix,slug", PAGES)
def test_links_reach_maintained_same_language_contracts(site_tree, prefix, slug):
    soup = read_page(site_tree, prefix, slug)
    links = soup.select("article a[href]")
    for route in (f"/{prefix}docs/security.html", f"/{prefix}docs/http-server.html"):
        matching = [link for link in links if urlsplit(link["href"]).path == route
                    and not urlsplit(link["href"]).netloc]
        assert matching, route
        for link in matching:
            visible_text(link)
            if site_tree != SITE / "legacy":
                target = BeautifulSoup((site_tree / route.lstrip("/")).read_text(), "html.parser")
                fragment = unquote(urlsplit(link["href"]).fragment)
                if fragment:
                    assert target.find(id=fragment), (route, fragment)
                source = target.select_one("[data-source-link]")
                expected = "SECURITY" if "security.html" in route else "README"
                suffix = "" if prefix else "_zh"
                assert source["href"].endswith(f"/examples/openai_api/{expected}{suffix}.md")


@pytest.mark.parametrize("prefix,slug", PAGES)
def test_migration_claims_are_bounded_in_article_metadata_and_index(site_tree, prefix, slug):
    soup = read_page(site_tree, prefix, slug)
    for text in prose_and_metadata(soup):
        assert_no_blanket_claims(text)
    index = read_page(site_tree, prefix, "index.html")
    cards = index.select(f'a.post-card[href="/{prefix}blog/{slug}"]')
    assert len(cards) == 1
    assert_no_blanket_claims(visible_text(cards[0]))


@pytest.mark.parametrize("prefix,slug", PAGES)
def test_visible_response_format_limits_match_packaged_handler(site_tree, prefix, slug):
    soup = read_page(site_tree, prefix, slug)
    section = soup.select_one('article [data-http-contract="formats"]')
    visible_text(section)
    for name in ("json", "verbose_json", "text", "srt", "vtt"):
        item = section.select_one(f'[data-response-format="{name}"]')
        text = visible_text(item)
        assert re.search(rf"\b{re.escape(name)}\b", text, re.I), text
        if name in ("srt", "vtt"):
            assert re.search(r"unsupported|not\s+(?:currently\s+)?supported|does\s+not\s+(?:produce|support)|不支持|未支持|不提供|不生成", text, re.I), text
        else:
            assert "json" in text.lower(), text
            if name == "text":
                assert re.search(r"string|quoted|字符串|引号", text, re.I), text
            elif name == "verbose_json":
                assert "segments" in text and "duration" in text, text
            else:
                assert "text" in text, text


def shell_argvs(section):
    for pre in section.select("pre"):
        for line in pre.get_text().replace("\\\n", " ").splitlines():
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
            lexer.whitespace_split = True
            argv = []
            for token in lexer:
                if token and set(token) <= set(";&|()"):
                    if argv:
                        yield argv
                    argv = []
                else:
                    argv.append(token)
            if argv:
                yield argv


def asserts_installed_funasr_version(source, release):
    tree = ast.parse(source)
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports[name.asname or name.name.split(".")[0]] = (
                    name.name if name.asname else name.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            for name in node.names:
                imports[name.asname or name.name] = f"{node.module}.{name.name}"

    def qualified(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{qualified(node.value)}.{node.attr}"
        return ""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        comparison = node.test
        if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
            continue
        pair = (comparison.left, comparison.comparators[0])
        for call, expected in (pair, pair[::-1]):
            if (isinstance(call, ast.Call)
                    and qualified(call.func) == "importlib.metadata.version"
                    and len(call.args) == 1
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "funasr"
                    and isinstance(expected, ast.Constant)
                    and expected.value == release):
                return True
    return False


@pytest.mark.parametrize("prefix,slug", PAGES)
def test_packaged_prerequisites_are_verified_before_startup(site_tree, prefix, slug):
    soup = read_page(site_tree, prefix, slug)
    section = soup.select_one('article [data-http-contract="prerequisites"]')
    text = visible_text(section)
    for concept in (
        r"separate|isolated|dedicated|独立|隔离",
        r"prepared|ready|准备好|已准备|已配置",
        r"torch|PyTorch",
        r"audio|音频",
        r"model|checkpoint|模型|权重",
    ):
        assert re.search(concept, text, re.I), (concept, text)
    assert any(re.search(r"inference|transcription|推理|转写", sentence, re.I)
               and re.search(NEGATIVE, sentence, re.I)
               for sentence in re.split(r"[.!?。！？]\s*", text)), (
        "Explain that package checks do not prove inference"
    )
    assert re.search(r"access|available|可用|访问|获取", text, re.I), text
    order = {id(node): index for index, node in enumerate(soup.find_all(True))}
    first_start = startup_commands(soup)[0][0]
    assert order[id(section)] < order[id(first_start)]
    for pre in section.select("pre"):
        assert order[id(pre)] < order[id(first_start)]
    release = "1.4.15"
    commands = list(shell_argvs(section))
    python_commands = [args for args in commands if re.fullmatch(
        r"python(?:\d+(?:\.\d+)*)?", Path(args[0]).name
    )]
    assert any(args[1:] == ["-m", "pip", "check"] for args in python_commands)
    checks = [args[2] for args in python_commands if len(args) == 3 and args[1] == "-c"]
    assert any(asserts_installed_funasr_version(code, release) for code in checks), (
        "Check installed distribution metadata against the documented release, not a print or literal"
    )
    assert release in text, text


@pytest.mark.parametrize("prefix,slug", PAGES)
def test_example_links_do_not_own_packaged_installation_or_schema(site_tree, prefix, slug):
    soup = read_page(site_tree, prefix, slug)
    for basename in ("http-server", "service-api"):
        route = f"/{prefix}docs/{basename}.html"
        links = [link for link in soup.select("article a[href]")
                 if urlsplit(link["href"]).path == route
                 and not urlsplit(link["href"]).netloc]
        assert links, route
        for link in links:
            assert re.search(r"example|示例", visible_text(link), re.I), (
                "Label the linked implementation rather than implying packaged ownership", route
            )
    authority = soup.select_one('article [data-http-contract="schema-authority"]')
    text = visible_text(authority)
    assert "/openapi.json" in text
    assert re.search(r"private|restricted|internal|私有|受限|内网|内部", text, re.I), text
    assert re.search(r"packaged|包内", text, re.I), text
    sources = [link for link in authority.select("a[href]") if re.fullmatch(
        r"https://github\.com/modelscope/FunASR/blob/[0-9a-f]{40}/funasr/bin/_server_app\.py",
        link["href"].split("#", 1)[0],
    )]
    assert sources, "Keep an immutable packaged-handler source next to the live-schema authority"
    for link in sources:
        visible_text(link)


def test_packaged_format_implementation_still_matches_documented_boundary():
    tree = ast.parse((REPO / "funasr/bin/_server_app.py").read_text())
    handler = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.AsyncFunctionDef) and node.name == "transcribe")
    comparisons = [node for node in ast.walk(handler) if isinstance(node, ast.Compare)
                   and isinstance(node.left, ast.Name) and node.left.id == "response_format"]
    assert {node.comparators[0].value for node in comparisons} == {"verbose_json", "text"}
    text_branch = next(node for node in ast.walk(handler) if isinstance(node, ast.If)
                       and isinstance(node.test, ast.Compare)
                       and isinstance(node.test.left, ast.Name)
                       and node.test.left.id == "response_format"
                       and node.test.comparators[0].value == "text")
    returned = text_branch.body[0].value
    assert isinstance(returned, ast.Call) and returned.func.id == "JSONResponse"
    assert ast.dump(returned.args[0]) == ast.dump(ast.parse('result["text"]', mode="eval").body)


@pytest.mark.parametrize("command", [
    "funasr-server --model sensevoice",
    "funasr-server --host 0.0.0.0",
    "funasr-server --host 127.0.0.1; funasr-server --model sensevoice",
    "funasr-server --host 127.0.0.1 --host=0.0.0.0",
])
def test_command_parser_does_not_hide_unsafe_later_startups(command):
    soup = BeautifulSoup(f"<article><pre>{command}</pre></article>", "html.parser")
    with pytest.raises(AssertionError):
        for _, args in startup_commands(soup):
            assert_loopback(args)
