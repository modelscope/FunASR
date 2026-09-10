"""Lightweight documentation contracts; no SDK import, network, or model weights."""

import ast
from pathlib import Path
import re
import types
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = (ROOT / "docs/python_api.md", ROOT / "docs/python_api_zh.md")
AUTO_MODEL = ROOT / "funasr/auto/auto_model.py"


def python_blocks(path):
    return re.findall(r"^```python\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S)


def source_tree(relative_path):
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def literal_gets(tree):
    values = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "get" or not node.args:
            continue
        try:
            key = ast.literal_eval(node.args[0])
            default = ast.literal_eval(node.args[1]) if len(node.args) > 1 else None
            values.add((key, repr(default)))
        except (ValueError, TypeError):
            pass
    return values


class PythonApiDocsContract(unittest.TestCase):
    def test_examples_compile_and_use_source_backed_keywords(self):
        tree = source_tree("funasr/auto/auto_model.py")
        known_keys = {key for key, _ in literal_gets(tree)}
        known_keys.update({"model", "input", "input_len", "progress_callback"})
        known_keys.update(key for key, _ in literal_gets(source_tree(
            "funasr/models/paraformer_streaming/model.py")))
        for path in DOCS:
            blocks = python_blocks(path)
            self.assertEqual(len(blocks), 4, path.name)
            for index, block in enumerate(blocks):
                with self.subTest(doc=path.name, block=index):
                    compile(block, str(path), "exec")
                    for node in ast.walk(ast.parse(block)):
                        if not isinstance(node, ast.Call):
                            continue
                        is_constructor = isinstance(node.func, ast.Name) and node.func.id == "AutoModel"
                        is_generate = isinstance(node.func, ast.Attribute) and node.func.attr == "generate"
                        if is_constructor or is_generate:
                            for keyword in node.keywords:
                                if keyword.arg is not None:
                                    self.assertIn(keyword.arg, known_keys)

    def test_translations_keep_identical_executable_examples(self):
        self.assertEqual(python_blocks(DOCS[0]), python_blocks(DOCS[1]))

    def alignment_helper(self, path):
        blocks = [block for block in python_blocks(path)
                  if "def sensevoice_word_intervals(" in block]
        self.assertEqual(len(blocks), 1, path.name)
        namespace = {}
        exec(compile(blocks[0], str(path), "exec"), namespace)
        return namespace["sensevoice_word_intervals"]

    def test_sensevoice_example_pairs_words_not_display_characters(self):
        # SenseVoice215: display characters are not the alignment units.
        result = {"text": "<|en|>hello world", "words": ["hello", "world"],
                  "timestamp": [[0, 500], [700, 1000]]}
        for path in DOCS:
            with self.subTest(doc=path.name):
                helper = self.alignment_helper(path)
                self.assertEqual(helper(result), [
                    {"word": "hello", "start_ms": 0, "end_ms": 500},
                    {"word": "world", "start_ms": 700, "end_ms": 1000},
                ])
                self.assertEqual(helper({"words": [], "timestamp": []}), [])
                self.assertEqual(result["words"], ["hello", "world"])

    def test_sensevoice_example_rejects_missing_or_truncated_alignment(self):
        for path in DOCS:
            helper = self.alignment_helper(path)
            for result in ({"text": ""}, {"words": [], "timestamp": None},
                           {"words": ["hello"], "timestamp": []},
                           {"words": [], "timestamp": [[0, 500]]}):
                with self.subTest(doc=path.name, result=result):
                    with self.assertRaises(ValueError):
                        helper(result)

    def test_local_links_resolve_without_repository_scan(self):
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
            self.assertTrue(links)
            for link in links:
                parts = urlsplit(link)
                if parts.scheme or not parts.path:
                    continue
                target = (path.parent / unquote(parts.path)).resolve()
                self.assertIn(ROOT.resolve(), target.parents)
                self.assertNotIn(".mcp-tasks", target.parts)
                self.assertTrue(target.exists(), f"{path.name}: {link}")

    def test_assigned_files_have_no_whitespace_errors(self):
        for path in (*DOCS, Path(__file__)):
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"), path.name)
            self.assertFalse(text.endswith("\n\n"), path.name)
            for number, line in enumerate(text.splitlines(), 1):
                self.assertEqual(line, line.rstrip(), f"{path.name}:{number}")

    def test_documented_wrapper_defaults_match_source(self):
        gets = literal_gets(source_tree("funasr/auto/auto_model.py"))
        expected = {
            "device": "cuda", "hub": "ms", "ngpu": 1, "ncpu": 4,
            "disable_update": False, "disable_pbar": False,
            "batch_size": 1, "batch_size_s": 300,
            "batch_size_threshold_s": 60, "merge_vad": False,
            "merge_length_s": 15, "spk_mode": "punc_segment",
            "return_spk_res": True, "return_spk_center": False,
            "preset_spk_num": None, "sentence_timestamp": False,
            "return_raw_text": False,
        }
        for key, value in expected.items():
            self.assertIn((key, repr(value)), gets, key)
            for path in DOCS:
                self.assertIn(f"`{key}`", path.read_text(encoding="utf-8"))

    def test_model_specific_defaults_match_source(self):
        expectations = {
            "funasr/models/sense_voice/model.py": {
                "language": "auto", "use_itn": False, "output_timestamp": False,
            },
            "funasr/models/fun_asr_nano/model.py": {
                "hotwords": [], "language": None, "itn": True,
            },
            "funasr/models/qwen3_asr/model.py": {
                "context": "", "return_time_stamps": False,
            },
            "funasr/models/paraformer_streaming/model.py": {
                "chunk_size": [0, 10, 5], "encoder_chunk_look_back": 0,
                "decoder_chunk_look_back": 0, "is_final": False,
            },
        }
        for path, defaults in expectations.items():
            gets = literal_gets(source_tree(path))
            for key, value in defaults.items():
                self.assertIn((key, repr(value)), gets, f"{path}: {key}")

    def test_hf_revision_caveat_matches_download_helper(self):
        tree = source_tree("funasr/download/download_model_from_hub.py")
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "get_or_download_model_dir_hf")
        calls = [node for node in ast.walk(helper) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "snapshot_download"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].keywords, [])
        for path in DOCS:
            text = path.read_text(encoding="utf-8")
            for marker in ("snapshot_download(model)", "model_revision", "disable_update",
                           "check_latest", "trust_remote_code", "OpenMOSS", "MIT",
                           "funasr-server", "AutoModelVLLM", "llama.cpp", "CLIENTS.md"):
                self.assertIn(marker, text)

    def test_generate_dispatch_preserves_caller_cache(self):
        # Execute only the real wrapper method with fake inference endpoints.
        tree = ast.parse(AUTO_MODEL.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "AutoModel")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "generate")
        namespace = {"apply_postprocess_hotwords_to_results": lambda results, cfg: results}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(AUTO_MODEL), "exec"), namespace)
        for use_vad in (False, True):
            calls = []
            resets = []
            expected = [{"key": "test", "text": "stub"}]

            def inference(input, **cfg):
                calls.append(("direct", input, cfg))
                return expected

            def inference_with_vad(input, **cfg):
                calls.append(("vad", input, cfg))
                return expected

            model = types.SimpleNamespace(
                vad_model=object() if use_vad else None, punc_model=None,
                _reset_runtime_configs=lambda: resets.append(True),
                inference=inference, inference_with_vad=inference_with_vad,
            )
            cache = {}
            result = namespace["generate"](model, "audio.wav", cache=cache, is_final=True)
            self.assertIs(result, expected)
            self.assertEqual(resets, [True])
            self.assertEqual(calls[0][0], "vad" if use_vad else "direct")
            self.assertIs(calls[0][2]["cache"], cache)
            self.assertTrue(calls[0][2]["is_final"])


if __name__ == "__main__":
    unittest.main()
