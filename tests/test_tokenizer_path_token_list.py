import pytest

from funasr.tokenizer.char_tokenizer import CharTokenizer


@pytest.mark.parametrize(
    "filename,content",
    [
        ("tokens.txt", "<unk>\na\nb\n"),
        ("tokens.json", '["<unk>", "a", "b"]'),
    ],
)
@pytest.mark.parametrize("as_string", [False, True])
def test_token_list_file(tmp_path, filename, content, as_string):
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")

    tokenizer = CharTokenizer(token_list=str(path) if as_string else path)

    assert tokenizer.token_list == ["<unk>", "a", "b"]
