import pytest

from funasr.bin.tokenize_text import field2slice


@pytest.mark.parametrize(
    "field,expected,words",
    [
        ("2", slice(1, 2), ["hello"]),
        ("2-2", slice(1, 2), ["hello"]),
        ("1-3", slice(0, 3), ["utt1", "hello", "world"]),
        ("2-", slice(1, None), ["hello", "world"]),
        ("-3", slice(None, 3), ["utt1", "hello", "world"]),
    ],
)
def test_field2slice_selects_expected_columns(field, expected, words):
    result = field2slice(field)
    assert result == expected
    assert "utt1 hello world".split()[result] == words


def test_field2slice_rejects_zero():
    with pytest.raises(RuntimeError, match="Format error"):
        field2slice("0")
