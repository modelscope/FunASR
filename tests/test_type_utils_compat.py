"""Keep the command-line boolean parser compatible without distutils."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "funasr/utils/type_utils.py"


@pytest.fixture(scope="module")
def parser():
    spec = importlib.util.spec_from_file_location("boolean_parser_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.str2bool


@pytest.mark.parametrize(
    ("value", "expected"),
    [("y", True), ("yes", True), ("t", True), ("true", True), ("on", True),
     ("1", True), ("n", False), ("no", False), ("f", False), ("false", False),
     ("off", False), ("0", False), ("YeS", True), ("TRUE", True),
     ("nO", False), ("OFF", False)],
)
def test_boolean_argument_preserves_values_and_boolean_type(parser, value, expected):
    assert parser(value) is expected


@pytest.mark.parametrize(
    ("value", "message"),
    [("MAYBE", "invalid truth value 'maybe'"),
     ("FaLsE!", "invalid truth value 'false!'"),
     (" TRUE ", "invalid truth value ' true '"),
     ("", "invalid truth value ''"),
     ("2", "invalid truth value '2'"),
     ("false!", "invalid truth value 'false!'")],
)
def test_invalid_boolean_argument_preserves_normalized_error(parser, value, message):
    with pytest.raises(ValueError) as caught:
        parser(value)
    assert str(caught.value) == message


def test_boolean_parser_imports_when_distutils_is_unavailable():
    code = """
import importlib.abc
import runpy
import sys

class MissingDistutils(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'distutils' or fullname.startswith('distutils.'):
            raise ImportError('distutils is unavailable')

sys.meta_path.insert(0, MissingDistutils())
parse = runpy.run_path(sys.argv[1])['str2bool']
assert parse('yes') is True
assert parse('no') is False
"""
    result = subprocess.run([sys.executable, "-c", code, str(SOURCE)],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
