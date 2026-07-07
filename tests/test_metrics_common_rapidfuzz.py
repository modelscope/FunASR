import importlib.util
import sys
import types
from pathlib import Path


def _install_fake_rapidfuzz():
    rapidfuzz = types.ModuleType("rapidfuzz")
    distance = types.ModuleType("rapidfuzz.distance")

    class Levenshtein:
        @staticmethod
        def distance(left, right):
            left = list(left)
            right = list(right)
            dp = list(range(len(right) + 1))
            for i, lval in enumerate(left, 1):
                prev = dp[0]
                dp[0] = i
                for j, rval in enumerate(right, 1):
                    old = dp[j]
                    dp[j] = min(
                        dp[j] + 1,
                        dp[j - 1] + 1,
                        prev + (0 if lval == rval else 1),
                    )
                    prev = old
            return dp[-1]

    distance.Levenshtein = Levenshtein
    rapidfuzz.distance = distance
    sys.modules.setdefault("rapidfuzz", rapidfuzz)
    sys.modules.setdefault("rapidfuzz.distance", distance)


def _load_metrics_common():
    _install_fake_rapidfuzz()
    module_path = Path(__file__).resolve().parents[1] / "funasr" / "metrics" / "common.py"
    spec = importlib.util.spec_from_file_location("metrics_common_probe", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["metrics_common_probe"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cer_and_wer_return_none_when_reference_length_is_zero():
    common = _load_metrics_common()
    calc = common.ErrorCalculator(
        char_list=["<blank>", "a", "b", "c", "h", "e", "l", "o", "w", "r", "d", " "],
        sym_space=" ",
        sym_blank="<blank>",
        report_cer=True,
        report_wer=True,
    )

    assert calc.calculate_cer(["abc"], ["   "]) is None
    assert calc.calculate_wer(["hello world"], ["   "]) is None


def test_cer_and_wer_keep_normal_levenshtein_semantics():
    common = _load_metrics_common()
    calc = common.ErrorCalculator(
        char_list=["<blank>", "a", "b", "c", "x", "y", " "],
        sym_space=" ",
        sym_blank="<blank>",
        report_cer=True,
        report_wer=True,
    )

    assert calc.calculate_cer(["abc"], ["axc"]) == 1 / 3
    assert calc.calculate_wer(["foo bar baz"], ["foo qux baz"]) == 1 / 3
