#!/usr/bin/env python3
"""Generate API page with source code preview and GitHub links.
Run this script from the repo root on main branch.
Output: gh-pages-output/api.html for deployment to gh-pages.

This script is designed to be run by GitHub Actions on every push to main,
automatically regenerating the API docs.
"""
import ast
import glob
import html as htmlmod
import os
import re
import shutil
from pathlib import Path

REPO_URL = "https://github.com/modelscope/FunASR"
BRANCH = "main"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "gh-pages-output"

os.chdir(REPO_ROOT)

tree_structure = {
    "funasr.auto": {"auto_model": "funasr/auto/auto_model.py"},
    "funasr.register": {"register": "funasr/register.py"},
    "funasr.models": {},
    "funasr.frontends": {},
    "funasr.tokenizer": {},
    "funasr.utils": {},
    "funasr.bin": {},
    "funasr.download": {},
    "funasr.train_utils": {},
    "funasr.datasets": {},
    "funasr.losses": {},
    "funasr.schedulers": {},
}

# Auto-discover
for f in sorted(glob.glob("funasr/models/*/model.py")):
    if "whisper_lib" in f:
        continue
    tree_structure["funasr.models"][f.split("/")[-2]] = f

for top in ["frontends", "tokenizer", "utils", "bin", "download", "train_utils", "losses", "schedulers"]:
    for f in sorted(glob.glob(f"funasr/{top}/*.py")):
        if "__init__" in f or "abs_" in f:
            continue
        tree_structure[f"funasr.{top}"][os.path.basename(f).replace(".py", "")] = f

for f in sorted(glob.glob("funasr/datasets/*.py") + glob.glob("funasr/datasets/audio_datasets/*.py")):
    if "__init__" in f or "__pycache__" in f:
        continue
    tree_structure["funasr.datasets"][os.path.basename(f).replace(".py", "")] = f

tree_structure = {k: v for k, v in tree_structure.items() if v}

SKIP_CLASSES = {
    "SinusoidalPositionEncoder", "StreamSinusoidalPositionEncoder",
    "PositionwiseFeedForward", "MultiHeadedAttentionSANM", "LayerNorm",
    "EncoderLayerSANM", "VadStateMachine", "FrameState", "AudioChangeState",
    "VadDetectMode", "VADXOptions", "E2EVadSpeechBufWithDoa",
    "E2EVadFrameProb", "WindowDetector", "Stats",
}

def esc(s):
    return htmlmod.escape(s) if s else ""

def format_doc(doc):
    if not doc:
        return '<p class="muted">No documentation yet.</p>'
    lines = doc.strip().split("\n")
    out = ""
    in_list = False
    for line in lines:
        s = line.strip()
        if s in ("Args:", "Returns:", "Raises:", "Examples:", "Note:", "Notes:", "Features:", "Models:", "Requirements:", "Output:", "Output format:"):
            if in_list: out += "</ul>\n"; in_list = False
            out += f'<h4>{s}</h4>\n'
            continue
        if s.startswith("- "):
            if not in_list: out += '<ul>\n'; in_list = True
            out += f'<li>{esc(s[2:])}</li>\n'
            continue
        m = re.match(r'^(\*{0,2}\w[\w.*]*(?:\s*\([^)]*\))?)\s*[:—\-]\s*(.+)', s)
        if m and not s.startswith("http"):
            if not in_list: out += '<ul>\n'; in_list = True
            out += f'<li><code>{esc(m.group(1))}</code> — {esc(m.group(2))}</li>\n'
            continue
        if (line.startswith("        ") or line.startswith("            ")) and in_list and s:
            out += f'<li class="sub">{esc(s)}</li>\n'
            continue
        if s:
            if in_list: out += "</ul>\n"; in_list = False
            out += f'<p>{esc(s)}</p>\n'
    if in_list: out += "</ul>\n"
    return out

def get_source_lines(filepath, node):
    """Get source code lines for an AST node."""
    with open(filepath) as f:
        all_lines = f.readlines()
    start = node.lineno - 1
    end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 20
    # Limit preview to 30 lines
    preview_end = min(start + 30, end)
    source = "".join(all_lines[start:preview_end])
    truncated = preview_end < end
    return source, start + 1, truncated

def github_url(filepath, lineno):
    return f"{REPO_URL}/blob/{BRANCH}/{filepath}#L{lineno}"

# Extract all data
all_data = {}
for top_level, sub_modules in tree_structure.items():
    for sub_name, filepath in sub_modules.items():
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=filepath)
        except:
            continue
        key = f"{top_level}.{sub_name}"
        entries = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if node.name in SKIP_CLASSES:
                    continue
                class_doc = ast.get_docstring(node) or ""
                src, lineno, trunc = get_source_lines(filepath, node)
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith("_"):
                            continue
                        mdoc = ast.get_docstring(item) or ""
                        args = [a.arg for a in item.args.args if a.arg != "self"]
                        if item.args.vararg: args.append("*" + item.args.vararg.arg)
                        if item.args.kwarg: args.append("**" + item.args.kwarg.arg)
                        msrc, mline, mtrunc = get_source_lines(filepath, item)
                        methods.append({"name": item.name, "args": args, "doc": mdoc,
                                       "source": msrc, "lineno": mline, "truncated": mtrunc})
                entries.append({"type": "class", "name": node.name, "doc": class_doc,
                               "methods": methods, "source": src, "lineno": lineno,
                               "truncated": trunc, "filepath": filepath})
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                fdoc = ast.get_docstring(node) or ""
                args = [a.arg for a in node.args.args]
                if node.args.vararg: args.append("*" + node.args.vararg.arg)
                if node.args.kwarg: args.append("**" + node.args.kwarg.arg)
                fsrc, fline, ftrunc = get_source_lines(filepath, node)
                entries.append({"type": "function", "name": node.name, "args": args,
                               "doc": fdoc, "source": fsrc, "lineno": fline,
                               "truncated": ftrunc, "filepath": filepath})
        if entries:
            all_data[key] = entries

total_entries = sum(len(e) for e in all_data.values())
print(f"Extracted: {len(all_data)} modules, {total_entries} entries")

# Generate HTML
sidebar = ""
content = ""
eid = 0

for top_level in tree_structure:
    sub_modules = tree_structure[top_level]
    sidebar += f'<div class="l1"><div class="l1-title" onclick="toggleL1(this)"><span>{esc(top_level)}</span><span class="cnt">{sum(1 for s in sub_modules if f"{top_level}.{s}" in all_data)}</span><span class="arr">▸</span></div><div class="l1-children">\n'
    for sub_name in sub_modules:
        key = f"{top_level}.{sub_name}"
        if key not in all_data:
            continue
        sidebar += f'<div class="l2"><div class="l2-title" onclick="toggleL2(this)"><span>{esc(sub_name)}</span><span class="arr">▸</span></div><div class="l2-children">\n'
        for entry in all_data[key]:
            eid += 1
            eid_str = f"e{eid}"
            filepath = entry.get("filepath", "")
            if entry["type"] == "class":
                sidebar += f'<a class="l3-item" data-target="{eid_str}" onclick="showEntry(\'{eid_str}\')"><span class="mb badge-c">C</span>{esc(entry["name"])}</a>\n'
                gh_link = github_url(filepath, entry["lineno"])
                content += f'<div class="api-detail" id="{eid_str}"><span class="dtag badge-c">class</span><h2>{esc(entry["name"])}</h2><div class="dmod">{key} · <a href="{gh_link}" target="_blank">View on GitHub ↗</a></div><div class="ddoc">{format_doc(entry["doc"])}</div>'
                content += f'<details class="src-block"><summary>📄 Source code</summary><pre><code>{esc(entry["source"])}</code></pre>'
                if entry["truncated"]:
                    content += f'<a href="{gh_link}" target="_blank" class="src-more">View full source on GitHub →</a>'
                content += '</details>'
                if entry["methods"]:
                    content += '<h3 class="mt">Methods</h3>'
                    for m in entry["methods"]:
                        sig = f'.{m["name"]}({", ".join(m["args"])})'
                        mgh = github_url(filepath, m["lineno"])
                        content += f'<div class="mblk"><code>{esc(sig)}</code> <a href="{mgh}" target="_blank" class="gh-link">L{m["lineno"]}</a><div class="mdoc">{format_doc(m["doc"])}</div>'
                        content += f'<details class="src-block"><summary>📄 Source</summary><pre><code>{esc(m["source"])}</code></pre>'
                        if m["truncated"]:
                            content += f'<a href="{mgh}" target="_blank" class="src-more">View full source →</a>'
                        content += '</details></div>'
                content += '</div>\n'
                for m in entry["methods"]:
                    eid += 1
                    mid = f"e{eid}"
                    sidebar += f'<a class="l3-item l3m" data-target="{mid}" onclick="showEntry(\'{mid}\')"><span class="mb badge-m">M</span>.{esc(m["name"])}()</a>\n'
                    sig = f'{entry["name"]}.{m["name"]}({", ".join(m["args"])})'
                    mgh = github_url(filepath, m["lineno"])
                    content += f'<div class="api-detail" id="{mid}"><span class="dtag badge-m">method</span><h2><code>{esc(sig)}</code></h2><div class="dmod">{key}.{entry["name"]} · <a href="{mgh}" target="_blank">View on GitHub ↗</a></div><div class="ddoc">{format_doc(m["doc"])}</div>'
                    content += f'<details class="src-block"><summary>📄 Source code</summary><pre><code>{esc(m["source"])}</code></pre>'
                    if m["truncated"]:
                        content += f'<a href="{mgh}" target="_blank" class="src-more">View full source on GitHub →</a>'
                    content += '</details></div>\n'
            else:
                args_str = ", ".join(entry.get("args", []))
                gh_link = github_url(filepath, entry["lineno"])
                sidebar += f'<a class="l3-item" data-target="{eid_str}" onclick="showEntry(\'{eid_str}\')"><span class="mb badge-f">F</span>{esc(entry["name"])}</a>\n'
                content += f'<div class="api-detail" id="{eid_str}"><span class="dtag badge-f">function</span><h2><code>{esc(entry["name"])}({esc(args_str)})</code></h2><div class="dmod">{key} · <a href="{gh_link}" target="_blank">View on GitHub ↗</a></div><div class="ddoc">{format_doc(entry["doc"])}</div>'
                content += f'<details class="src-block"><summary>📄 Source code</summary><pre><code>{esc(entry["source"])}</code></pre>'
                if entry["truncated"]:
                    content += f'<a href="{gh_link}" target="_blank" class="src-more">View full source on GitHub →</a>'
                content += '</details></div>\n'
        sidebar += '</div></div>\n'
    sidebar += '</div></div>\n'

# Write the full page (keeping same CSS structure + source styles)
html_page = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>FunASR API</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
<style>
body{{display:flex;flex-direction:column;min-height:100vh;background:var(--surface)}}
.api-layout{{display:flex;flex:1}}
.api-sidebar{{width:260px;min-width:260px;height:calc(100vh - 57px);position:sticky;top:57px;overflow-y:auto;border-right:1px solid var(--line);background:var(--page);padding:8px 0}}
.sb-search{{margin:4px 8px 10px;padding:7px 10px;width:calc(100% - 16px);border:1px solid var(--line);border-radius:5px;font-size:0.78rem}}
.sb-search:focus{{outline:none;border-color:var(--primary)}}
.l1{{border-bottom:1px solid var(--line)}}.l1-title{{padding:9px 12px;font-size:0.78rem;font-weight:700;color:var(--ink-soft);cursor:pointer;display:flex;align-items:center;gap:6px}}
.l1-title:hover{{background:var(--surface-2)}}.l1-title .cnt{{font-size:0.65rem;background:var(--surface-2);color:var(--muted);padding:1px 5px;border-radius:8px;margin-left:auto}}
.l1-title .arr{{font-size:0.6rem;color:var(--muted);transition:transform 0.2s}}.l1-children{{max-height:0;overflow:hidden;transition:max-height 0.3s}}
.l1.open>.l1-children{{max-height:20000px}}.l1.open>.l1-title .arr{{transform:rotate(90deg)}}
.l2-title{{padding:4px 12px 4px 22px;font-size:0.74rem;font-weight:600;color:var(--muted);cursor:pointer;display:flex;align-items:center}}
.l2-title:hover{{background:var(--surface-2);color:var(--ink-soft)}}.l2-title .arr{{margin-left:auto;font-size:0.55rem;transition:transform 0.2s}}
.l2-children{{max-height:0;overflow:hidden;transition:max-height 0.25s}}.l2.open>.l2-children{{max-height:20000px}}.l2.open>.l2-title .arr{{transform:rotate(90deg)}}
.l3-item{{display:flex;align-items:center;gap:5px;padding:2px 12px 2px 34px;font-size:0.73rem;color:var(--ink-soft);text-decoration:none;cursor:pointer;border-left:2px solid transparent}}
.l3-item:hover{{background:var(--surface-2)}}.l3-item.active{{background:var(--primary-soft);color:var(--primary-dark);border-left-color:var(--primary);font-weight:600}}
.l3m{{padding-left:44px;font-size:0.7rem}}
.mb{{font-size:0.5rem;font-weight:700;width:13px;height:13px;display:inline-flex;align-items:center;justify-content:center;border-radius:2px;flex-shrink:0}}
.badge-c{{background:#dbeafe;color:#1d4ed8}}.badge-m{{background:#d1fae5;color:#059669}}.badge-f{{background:#fef3c7;color:#b45309}}
.api-content{{flex:1;padding:32px 40px;overflow-y:auto;max-width:800px}}
.api-detail{{display:none}}.api-detail.active{{display:block}}
.api-welcome{{color:var(--muted);padding:60px 20px;text-align:center}}.api-welcome h2{{color:var(--ink-soft);margin-bottom:8px}}
.dtag{{font-size:0.63rem;font-weight:700;padding:2px 7px;border-radius:3px;text-transform:uppercase}}
h2{{font-size:1.1rem;margin:8px 0}}h2 code{{font-family:'JetBrains Mono',monospace;font-size:0.92rem;background:none;color:var(--ink);padding:0}}
.dmod{{font-size:0.7rem;color:var(--muted);font-family:monospace;margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid var(--line)}}
.dmod a{{color:var(--primary);text-decoration:none;font-size:0.7rem}}.dmod a:hover{{text-decoration:underline}}
.ddoc{{line-height:1.7}}.ddoc p{{margin-bottom:8px;font-size:0.87rem}}.ddoc h4{{font-size:0.82rem;font-weight:700;margin:16px 0 4px;color:var(--ink)}}
.ddoc ul{{list-style:none;padding:0;margin:0 0 12px}}.ddoc li{{padding:3px 0 3px 12px;border-left:2px solid var(--line);font-size:0.83rem;color:var(--ink-soft)}}
.ddoc li code{{background:var(--surface-2);padding:1px 4px;border-radius:3px;font-size:0.77rem;color:#b91c1c}}
.ddoc li.sub{{padding-left:24px;font-size:0.79rem;color:var(--muted);border-left-color:transparent}}
.mt{{font-size:0.85rem;font-weight:700;margin:24px 0 10px;padding-top:14px;border-top:1px solid var(--line)}}
.mblk{{margin:8px 0;padding:10px 14px;background:var(--page);border-radius:6px;border:1px solid var(--line)}}
.mblk code{{font-family:'JetBrains Mono',monospace;font-size:0.84rem}}
.mdoc p{{font-size:0.83rem}}.mdoc h4{{font-size:0.78rem}}
.gh-link{{font-size:0.68rem;color:var(--muted);text-decoration:none;margin-left:8px}}.gh-link:hover{{color:var(--primary)}}
.src-block{{margin:12px 0;border:1px solid var(--line);border-radius:6px;overflow:hidden}}
.src-block summary{{padding:8px 14px;font-size:0.78rem;cursor:pointer;color:var(--muted);background:var(--surface-2)}}
.src-block summary:hover{{color:var(--ink-soft)}}
.src-block pre{{margin:0;border-radius:0;font-size:0.75rem;max-height:400px;overflow-y:auto}}
.src-more{{display:block;padding:8px 14px;font-size:0.75rem;color:var(--primary);text-decoration:none;border-top:1px solid var(--line)}}
.src-more:hover{{background:var(--surface-2)}}
.muted{{color:var(--muted-2);font-style:italic}}
.hidden{{display:none!important}}
</style></head><body>
<nav class="nav"><div class="container">
<a href="index.html" class="nav-logo">FunASR</a>
<div class="nav-links"><a href="index.html">Home</a><a href="tutorial.html">Tutorial</a><a href="training.html">Training</a><a href="model-registration.html">Develop</a><a href="api.html" class="active">API</a></div>
<a href="https://github.com/modelscope/FunASR" class="nav-github">GitHub</a>
</div></nav>
<div class="api-layout">
<div class="api-sidebar">
<input type="text" class="sb-search" placeholder="Search..." oninput="filterSidebar(this.value)">
{sidebar}
</div>
<div class="api-content">
<div class="api-welcome" id="api-welcome"><h2>API Reference</h2><p>{total_entries} entries · auto-generated from source<br>Click source code to expand. Links point to latest GitHub code.</p></div>
{content}
</div></div>
<script>
function toggleL1(el){{el.closest('.l1').classList.toggle('open')}}
function toggleL2(el){{el.closest('.l2').classList.toggle('open')}}
function showEntry(id){{
document.querySelectorAll('.api-detail').forEach(e=>e.classList.remove('active'));
document.querySelectorAll('.l3-item').forEach(e=>e.classList.remove('active'));
document.getElementById('api-welcome').style.display='none';
var t=document.getElementById(id);if(t)t.classList.add('active');
var l=document.querySelector('[data-target="'+id+'"]');if(l)l.classList.add('active');
document.querySelector('.api-content').scrollTop=0;
}}
function filterSidebar(q){{
q=q.toLowerCase();
document.querySelectorAll('.l3-item').forEach(e=>e.classList.toggle('hidden',q&&!e.textContent.toLowerCase().includes(q)));
if(q)document.querySelectorAll('.l1,.l2').forEach(e=>e.classList.add('open'));
}}
if(window.location.hash)showEntry(window.location.hash.slice(1));
</script></body></html>'''

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with (OUTPUT_DIR / "api.html").open("w", encoding="utf-8") as f:
    f.write(html_page)

training_src = REPO_ROOT / "training.html"
if training_src.exists():
    shutil.copy2(training_src, OUTPUT_DIR / "training.html")

print("Generated gh-pages-output/api.html")
