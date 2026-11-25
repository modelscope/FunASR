"""Model vocabulary IDs must reach callers as pieces, not BPE-decoded text."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import sentencepiece as spm
import torch

from funasr.models.paraformer.model import Paraformer
from funasr.tokenizer.build_tokenizer import build_tokenizer
from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer


@pytest.fixture(scope="module")
def bpe_model(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("sentencepiece") / "tiny"
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(
            ["hello world", "hello there", "he tried to think how it could be", "你好 world"] * 20
        ),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=48,
        user_defined_symbols=["<|en|>"],
        shuffle_input_sentence=False,
        num_threads=1,
        minloglevel=2,
    )
    return str(prefix.with_suffix(".model"))


def vocabulary(bpe_model, reordered=True):
    sp = spm.SentencePieceProcessor(model_file=bpe_model)
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    return list(reversed(pieces)) if reordered else pieces


@pytest.mark.parametrize("storage", ["list", "tuple", "json", "txt"])
def test_reordered_vocabulary_decodes_expected_text(bpe_model, tmp_path, storage):
    tokens = vocabulary(bpe_model)
    token_list = tokens
    if storage == "tuple":
        token_list = tuple(tokens)
    elif storage in ("json", "txt"):
        path = tmp_path / ("tokens." + storage)
        content = json.dumps(tokens) if storage == "json" else "\n".join(tokens) + "\n"
        path.write_text(content, encoding="utf-8")
        token_list = str(path)
    tokenizer = SentencepiecesTokenizer(bpe_model, token_list=token_list)
    pieces = tokenizer.text2tokens("hello world")
    model_ids = [tokens.index(piece) for piece in pieces]

    # Establish that the two vocabularies really assign different meanings to IDs.
    assert tokenizer.decode(model_ids) != "hello world"
    result = tokenizer.ids2tokens(model_ids)

    assert result == pieces
    assert tokenizer.tokens2text(result) == "hello world"


@pytest.mark.parametrize("text", ["hello world", "你好 world", "<|en|> hello world", ""])
def test_matching_vocabulary_preserves_text_and_piece_boundaries(bpe_model, text):
    tokenizer = SentencepiecesTokenizer(bpe_model, token_list=vocabulary(bpe_model, False))
    ids = tokenizer.encode(text)

    pieces = tokenizer.ids2tokens(ids)

    assert pieces == tokenizer.text2tokens(text)
    assert tokenizer.tokens2text(pieces) == tokenizer.decode(ids) == text


@pytest.mark.parametrize("container", [list, tuple, iter, np.asarray])
def test_model_ids_accept_base_tokenizer_containers(bpe_model, container):
    tokens = vocabulary(bpe_model)
    tokenizer = SentencepiecesTokenizer(bpe_model, token_list=tokens)
    ids = [tokens.index(piece) for piece in tokenizer.text2tokens("hello world")]

    assert tokenizer.tokens2text(tokenizer.ids2tokens(container(ids))) == "hello world"


def test_special_tokens_and_model_only_ids_are_preserved(bpe_model):
    tokens = vocabulary(bpe_model) + ["<blank>"]
    tokenizer = SentencepiecesTokenizer(bpe_model, token_list=tokens)
    pieces = ["<s>", "<unk>", "<|en|>", "▁", "▁", "</s>", "<blank>"]

    assert tokenizer.ids2tokens([tokens.index(piece) for piece in pieces]) == pieces
    assert tokenizer.ids2tokens([]) == []


def test_model_ids_require_one_dimensional_numpy_array(bpe_model):
    tokenizer = SentencepiecesTokenizer(bpe_model, token_list=vocabulary(bpe_model))

    with pytest.raises(ValueError, match="Must be 1 dim ndarray"):
        tokenizer.ids2tokens(np.array([[1, 2]]))


@pytest.mark.parametrize("construction", ["omitted", "none", "build_tokenizer"])
def test_without_vocabulary_preserves_legacy_text_decode(bpe_model, construction):
    if construction == "build_tokenizer":
        tokenizer = build_tokenizer("bpe", bpemodel=bpe_model)
    elif construction == "none":
        tokenizer = SentencepiecesTokenizer(bpe_model, token_list=None)
    else:
        tokenizer = SentencepiecesTokenizer(bpe_model)
    ids = tokenizer.encode("hello world")

    assert tokenizer.ids2tokens(ids) == "hello world"
    assert tokenizer.ids2tokens(line=ids) == "hello world"
    assert tokenizer.ids2tokens([ids, ids]) == ["hello world", "hello world"]
    assert tokenizer.ids2tokens([]) == ""


def test_encode_decode_and_legacy_keyword_remain_sentencepiece_operations(bpe_model):
    tokenizer = SentencepiecesTokenizer(bpe_model, token_list=vocabulary(bpe_model))
    ids = tokenizer.encode("hello world")
    model_ids = [tokenizer.token_list.index(p) for p in tokenizer.text2tokens("hello world")]

    assert tokenizer.tokens2ids("hello world") == ids
    assert tokenizer.decode(ids) == "hello world"
    assert tokenizer.tokens2text(tokenizer.ids2tokens(line=model_ids, unused=True)) == "hello world"


def test_paraformer_inference_uses_model_pieces_and_filters_control_ids(bpe_model, tmp_path):
    tokens = ["<blank>", "<s>", "</s>"] + [
        p for p in vocabulary(bpe_model) if p not in ("<s>", "</s>")
    ]
    tokenizer = SentencepiecesTokenizer(bpe_model, token_list=tokens)
    pieces = tokenizer.text2tokens("hello world")
    ids = [0, 1] + [tokens.index(p) for p in pieces] + [2, 0]
    scores = torch.full((1, len(ids), len(tokens)), -10.0)
    scores[0, torch.arange(len(ids)), ids] = 10.0
    lengths = torch.tensor([len(ids)])
    speech = torch.zeros(1, len(ids), 2)
    # Only the acoustic stages are controlled; the actual inference method,
    # tokenizer, special-ID filtering, text conversion and writer all run.
    state = SimpleNamespace(
        beam_search=None,
        sos=1,
        eos=2,
        blank_id=0,
        encode=lambda *args: (speech, lengths),
        calc_predictor=lambda *args: (speech, lengths.float(), None, None),
        cal_decoder_with_predictor=lambda *args: (scores, lengths),
    )

    results, _ = Paraformer.inference(
        state,
        speech,
        data_lengths=lengths,
        key=["sample"],
        tokenizer=tokenizer,
        data_type="fbank",
        device="cpu",
        output_dir=str(tmp_path),
    )

    assert results == [{"key": "sample", "text": "hello world"}]
    assert (tmp_path / "1best_recog" / "token").read_text().strip() == (
        "sample " + " ".join(pieces)
    )
