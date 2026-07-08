from types import SimpleNamespace

import torch

from funasr.models.fun_asr_nano import model as nano_model


class _FakeLLM:
    def __init__(self):
        self.config = SimpleNamespace(pad_token_id=None, eos_token_id=0)
        self.calls = 0

    def to(self, _dtype):
        return self

    def generate(self, **_kwargs):
        self.calls += 1
        return torch.tensor([[10 + self.calls]], dtype=torch.long)


class _FakeTokenizer:
    def batch_decode(self, generated_ids, **_kwargs):
        return [f"text-{int(generated_ids[0, -1])}"]


class _FakeCTCDecoder:
    def __call__(self, encoder_out, encoder_out_lens):
        logits = torch.tensor(
            [[[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]],
            dtype=torch.float32,
        )
        return logits, encoder_out_lens


class _FakeCTC:
    def log_softmax(self, decoder_out):
        return decoder_out


class _FakeCTCTokenizer:
    def decode(self, token_ids):
        return "".join(str(token_id) for token_id in token_ids)

    def encode(self, text):
        return [1] if text else []


def test_ctc_decoder_multi_segment_input_uses_single_segment_fallback(monkeypatch):
    instance = object.__new__(nano_model.FunASRNano)
    instance.ctc_decoder = _FakeCTCDecoder()
    instance.ctc = _FakeCTC()
    instance.ctc_tokenizer = _FakeCTCTokenizer()
    instance.blank_id = 0
    instance.llm = _FakeLLM()

    prepare_calls = []

    def fake_inference_prepare(
        data_in,
        data_lengths=None,
        key=None,
        tokenizer=None,
        frontend=None,
        **_kwargs,
    ):
        if len(data_in) > 1:
            raise NotImplementedError("batch decoding is not implemented")
        prepare_calls.append((data_in[0], key[0]))
        return (
            torch.zeros(1, 2, 4),
            {"assistant": [f"label-{data_in[0]}"]},
            {"attention_mask": torch.ones(1, 2, dtype=torch.long)},
            torch.empty(1, 0, dtype=torch.long),
            {
                "encoder_out": torch.zeros(1, 2, 3),
                "encoder_out_lens": torch.tensor([2]),
                "batch_data_time": 1.0,
            },
        )

    monkeypatch.setattr(instance, "inference_prepare", fake_inference_prepare)
    monkeypatch.setattr(
        nano_model,
        "forced_align",
        lambda _logits, target_ids, _blank_id: [
            {"token": int(target_ids[0]), "start_time": 0.0, "end_time": 1.0}
        ],
    )

    results, meta = instance.inference_llm(
        ["seg-a", "seg-b"],
        key=["key-a", "key-b"],
        tokenizer=_FakeTokenizer(),
        frontend=None,
        device="cpu",
        llm_dtype="fp32",
    )

    assert prepare_calls == [("seg-a", "key-a"), ("seg-b", "key-b")]
    assert [result["key"] for result in results] == ["key-a", "key-b"]
    assert [result["text"] for result in results] == ["text-11", "text-12"]
    assert all("timestamps" in result for result in results)
    assert meta["batch_data_time"] == 2.0
