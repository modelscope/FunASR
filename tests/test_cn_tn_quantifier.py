import importlib.util
from pathlib import Path


module_path = (
    Path(__file__).parents[1] / "funasr" / "models" / "fun_asr_nano" / "tools" / "cn_tn.py"
)
spec = importlib.util.spec_from_file_location("cn_tn", module_path)
cn_tn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cn_tn)


def test_quantifier_does_not_replace_digits_in_an_earlier_number():
    assert cn_tn.normalize_nsw("100和1个") == "一百和一个"


def test_simple_quantifier():
    assert cn_tn.normalize_nsw("拿3个") == "拿三个"
