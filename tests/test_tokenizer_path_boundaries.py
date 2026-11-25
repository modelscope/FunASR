import json

import pytest

from funasr.tokenizer.char_tokenizer import CharTokenizer


@pytest.mark.parametrize("suffix", [".txt", ".json"])
@pytest.mark.parametrize("as_string", [False, True])
def test_file_roundtrip_and_unknown_token(tmp_path, suffix, as_string):
    path = tmp_path / ("tokens" + suffix)
    tokens = ["<unk>", "a", "b"]
    path.write_text(json.dumps(tokens) if suffix == ".json" else "\n".join(tokens) + "\n",
                    encoding="utf-8")
    tokenizer = CharTokenizer(token_list=str(path) if as_string else path)
    assert tokenizer.encode("abx") == [1, 2, 0]
    assert tokenizer.decode([1, 2]) == "ab"
    assert tokenizer.get_num_vocabulary_size() == 3
    assert tokenizer.token_list_repr == str(path)


@pytest.mark.parametrize("suffix", [".txt", ".json"])
def test_missing_file_keeps_file_error(tmp_path, suffix):
    with pytest.raises(FileNotFoundError):
        CharTokenizer(token_list=tmp_path / ("missing" + suffix))


def test_invalid_json_keeps_decode_error(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("[", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        CharTokenizer(token_list=path)


@pytest.mark.parametrize("tokens,message", [(["<unk>", "a", "a"], "duplicated"),
                                            (["a", "b"], "Unknown symbol")])
@pytest.mark.parametrize("suffix", [".txt", ".json"])
def test_invalid_vocabulary_keeps_validation(tmp_path, tokens, message, suffix):
    path = tmp_path / ("invalid" + suffix)
    path.write_text(json.dumps(tokens) if suffix == ".json" else "\n".join(tokens),
                    encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        CharTokenizer(token_list=path)


@pytest.mark.parametrize("container", [list, tuple, iter])
def test_iterable_vocabulary_unchanged(container):
    tokenizer = CharTokenizer(token_list=container(["<unk>", "a", "b"]))
    assert tokenizer.encode("abx") == [1, 2, 0]
    assert tokenizer.decode([1, 2]) == "ab"
