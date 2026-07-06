import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_MODULE_PATH = Path(__file__).resolve().parents[1] / "funasr" / "utils" / "postprocess_hotwords.py"
_SPEC = importlib.util.spec_from_file_location("postprocess_hotwords", _MODULE_PATH)
postprocess_hotwords = importlib.util.module_from_spec(_SPEC)
sys.modules["postprocess_hotwords"] = postprocess_hotwords
assert _SPEC.loader is not None
_SPEC.loader.exec_module(postprocess_hotwords)

HotwordMatch = postprocess_hotwords.HotwordMatch
PostprocessHotwordMatcher = postprocess_hotwords.PostprocessHotwordMatcher
apply_postprocess_hotwords_to_results = postprocess_hotwords.apply_postprocess_hotwords_to_results
build_postprocess_hotword_matcher = postprocess_hotwords.build_postprocess_hotword_matcher
parse_hotword_file = postprocess_hotwords.parse_hotword_file
parse_postprocess_hotwords = postprocess_hotwords.parse_postprocess_hotwords


class TestPostprocessHotwordParsing(unittest.TestCase):
    def test_parse_list_and_dict(self):
        explicit, fuzzy = parse_postprocess_hotwords(["科大讯飞", "东方财富"])
        self.assertEqual(explicit, {})
        self.assertEqual(fuzzy, ["科大讯飞", "东方财富"])

        explicit, fuzzy = parse_postprocess_hotwords({"科大迅飞": "科大讯飞", "东方财富": "东方财富"})
        self.assertEqual(explicit, {"科大迅飞": "科大讯飞"})
        self.assertEqual(fuzzy, ["东方财富"])

    def test_parse_inline_mapping(self):
        explicit, fuzzy = parse_postprocess_hotwords(["撒贝你=>撒贝宁", "康辉"])
        self.assertEqual(explicit, {"撒贝你": "撒贝宁"})
        self.assertEqual(fuzzy, ["康辉"])

    def test_parse_hotword_file(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("# comment\n")
            f.write("科大讯飞\n")
            f.write("科大迅飞=>科大讯飞\n")
            path = f.name

        try:
            explicit, fuzzy = parse_hotword_file(path)
            self.assertEqual(explicit, {"科大迅飞": "科大讯飞"})
            self.assertEqual(fuzzy, ["科大讯飞"])
        finally:
            os.unlink(path)


class TestPostprocessHotwordMatcher(unittest.TestCase):
    def test_explicit_replace(self):
        matcher = PostprocessHotwordMatcher(
            explicit_map={"撒贝你": "撒贝宁"},
            enable_fuzzy=False,
        )
        text, matches = matcher.apply_text("我非常喜欢撒贝你说的新闻")
        self.assertEqual(text, "我非常喜欢撒贝宁说的新闻")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].replacement, "撒贝宁")
        self.assertEqual(matches[0].score, 1.0)

    def test_fuzzy_replace_with_optional_deps(self):
        try:
            import pypinyin  # noqa: F401
            import rapidfuzz  # noqa: F401
        except ImportError:
            self.skipTest("pypinyin and rapidfuzz are required for fuzzy tests")

        matcher = PostprocessHotwordMatcher(
            fuzzy_targets=["科大讯飞"],
            threshold=0.75,
        )
        text, matches = matcher.apply_text("科大迅飞的语音识别很强")
        self.assertIn("科大讯飞", text)
        self.assertTrue(matches)
        self.assertEqual(matches[0].replacement, "科大讯飞")

    def test_missing_fuzzy_dependency_raises(self):
        with mock.patch.object(
            postprocess_hotwords,
            "_require_pypinyin",
            side_effect=ImportError("missing pypinyin"),
        ):
            with self.assertRaises(ImportError):
                PostprocessHotwordMatcher(
                    fuzzy_targets=["科大讯飞"],
                    threshold=0.85,
                )

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            PostprocessHotwordMatcher(explicit_map={"a": "b"}, threshold=1.5)

    def test_sentence_info_and_timestamp_preserved(self):
        matcher = PostprocessHotwordMatcher(
            explicit_map={"撒贝你": "撒贝宁"},
            enable_fuzzy=False,
        )
        result = {
            "text": "撒贝你主持节目",
            "timestamp": [[0, 100], [100, 200]],
            "sentence_info": [
                {"text": "撒贝你主持", "sentence": "撒贝你主持", "start": 0, "end": 1000},
                {"text": "节目", "sentence": "节目", "start": 1000, "end": 1500},
            ],
        }
        matcher.apply_result(result, return_matches=True)
        self.assertEqual(result["text"], "撒贝宁主持节目")
        self.assertEqual(result["sentence_info"][0]["text"], "撒贝宁主持")
        self.assertEqual(result["sentence_info"][0]["sentence"], "撒贝宁主持")
        self.assertEqual(result["timestamp"], [[0, 100], [100, 200]])
        self.assertEqual(result["postprocess_hotword_matches"][0]["replacement"], "撒贝宁")

    def test_matcher_reused_for_multiple_results(self):
        build_calls = {"count": 0}
        original_build = build_postprocess_hotword_matcher

        def counting_build(*args, **kwargs):
            build_calls["count"] += 1
            return original_build(*args, **kwargs)

        results = [
            {"text": "撒贝你主持", "timestamp": [1]},
            {"text": "康灰播报", "timestamp": [2]},
        ]
        cfg = {
            "postprocess_hotwords": {"撒贝你": "撒贝宁", "康灰": "康辉"},
            "return_postprocess_hotword_matches": True,
        }
        with mock.patch.object(
            postprocess_hotwords,
            "build_postprocess_hotword_matcher",
            side_effect=counting_build,
        ):
            updated = apply_postprocess_hotwords_to_results(results, cfg)

        self.assertEqual(build_calls["count"], 1)
        self.assertEqual(updated[0]["text"], "撒贝宁主持")
        self.assertEqual(updated[1]["text"], "康辉播报")
        self.assertEqual(len(updated[0]["postprocess_hotword_matches"]), 1)

    def test_noop_when_not_configured(self):
        results = [{"text": "不变", "timestamp": [1]}]
        updated = apply_postprocess_hotwords_to_results(results, {})
        self.assertIs(updated, results)
        self.assertEqual(updated[0]["text"], "不变")


class TestHotwordMatch(unittest.TestCase):
    def test_as_dict(self):
        match = HotwordMatch("a", "b", 0.9, 1, 2)
        self.assertEqual(
            match.as_dict(),
            {"original": "a", "replacement": "b", "score": 0.9, "start": 1, "end": 2},
        )


if __name__ == "__main__":
    unittest.main()
