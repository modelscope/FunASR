import io
import sys
import types
from contextlib import redirect_stdout
from unittest.mock import patch

from funasr import cli


class DummyAutoModel:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances.append(self)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [{"text": "hello"}]


class SubtitleAutoModel(DummyAutoModel):
    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if kwargs.get("sentence_timestamp") and kwargs.get("output_timestamp"):
            return [
                {
                    "text": "<|zh|>第一句。第二句。",
                    "sentence_info": [
                        {"start": 0, "end": 1200, "text": "<|zh|>第一句。"},
                        {"start": 1200, "end": 2600, "sentence": "第二句。"},
                    ],
                }
            ]
        return [{"text": "<|zh|>第一句。第二句。"}]


class ContinuationSubtitleAutoModel(DummyAutoModel):
    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [
            {
                "text": "甲，乙。",
                "sentence_info": [
                    {"start": 0, "end": 500, "text": "甲，"},
                    {"start": 600, "end": 1200, "text": "乙。"},
                ],
            }
        ]


def test_cli_passes_hub_to_auto_model(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    DummyAutoModel.instances = []
    argv = ["funasr", "--hub", "hf", str(audio_path)]

    with (
        patch.object(sys, "argv", argv),
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("funasr.AutoModel", DummyAutoModel),
        redirect_stdout(io.StringIO()) as stdout,
    ):
        cli.main()

    assert DummyAutoModel.instances[0].kwargs["hub"] == "hf"
    assert stdout.getvalue().strip() == "hello"


def test_cli_routes_multiple_hotwords_to_paraformer_hotword(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    DummyAutoModel.instances = []
    argv = [
        "funasr",
        "--model",
        "paraformer",
        "--hotwords",
        "FunASR, ModelScope",
        str(audio_path),
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("funasr.AutoModel", DummyAutoModel),
        redirect_stdout(io.StringIO()),
    ):
        cli.main()

    generate_kwargs = DummyAutoModel.instances[0].generate_kwargs
    assert generate_kwargs["hotword"] == "FunASR ModelScope"
    assert "hotwords" not in generate_kwargs


def test_timestamp_bounds_skip_malformed_entries():
    assert cli._timestamp_bounds_ms(
        {
            "timestamp": [
                [None, 1000],
                ["bad", "2000"],
                ["1200.0", "2600.0"],
                {"start": "0.5", "end": "1.0"},
                {"start_time": "oops", "end_time": "2.0"},
            ]
        }
    ) == (500, 2600)


def test_cli_srt_requests_sentence_timestamps_and_writes_segmented_output(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    out_dir = tmp_path / "subs"
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )

    SubtitleAutoModel.instances = []
    argv = [
        "funasr",
        str(audio_path),
        "--output-format",
        "srt",
        "--output-dir",
        str(out_dir),
        "--lang",
        "zh",
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.dict(sys.modules, {"torch": fake_torch}),
        patch("funasr.AutoModel", SubtitleAutoModel),
        redirect_stdout(io.StringIO()),
    ):
        cli.main()

    instance = SubtitleAutoModel.instances[0]
    assert instance.kwargs["punc_model"] == "ct-punc"
    assert instance.generate_kwargs["language"] == "zh"
    assert instance.generate_kwargs["sentence_timestamp"] is True
    assert instance.generate_kwargs["output_timestamp"] is True
    assert instance.generate_kwargs["return_time_stamps"] is True
    assert (out_dir / "sample.srt").read_text(encoding="utf-8") == (
        "1\n"
        "00:00:00,000 --> 00:00:01,200\n"
        "第一句。\n\n"
        "2\n"
        "00:00:01,200 --> 00:00:02,600\n"
        "第二句。\n"
    )


def test_cli_srt_supports_readable_and_sentence_segment_modes(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not a real wav")
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )

    outputs = {}
    for mode in ("readable", "sentence"):
        out_dir = tmp_path / mode
        argv = [
            "funasr",
            str(audio_path),
            "--output-format",
            "srt",
            "--output-dir",
            str(out_dir),
        ]
        if mode == "sentence":
            argv.extend(["--subtitle-segment-mode", "sentence"])

        with (
            patch.object(sys, "argv", argv),
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("funasr.AutoModel", ContinuationSubtitleAutoModel),
            redirect_stdout(io.StringIO()),
        ):
            cli.main()
        outputs[mode] = (out_dir / "sample.srt").read_text(encoding="utf-8")

    assert outputs["readable"] == "1\n00:00:00,000 --> 00:00:01,200\n甲，乙。\n"
    assert outputs["sentence"] == (
        "1\n00:00:00,000 --> 00:00:00,500\n甲，\n\n"
        "2\n00:00:00,600 --> 00:00:01,200\n乙。\n"
    )


def test_merge_subtitle_segments_groups_continuation_cues():
    segments = [
        {"start": 98520, "end": 99900, "text": "救护车上常用的呼吸机、"},
        {"start": 100140, "end": 100500, "text": "除颤仪，"},
        {"start": 100620, "end": 101580, "text": "属于二类医疗器械。"},
    ]

    assert cli.merge_subtitle_segments(segments) == [
        {
            "start": 98520,
            "end": 101580,
            "text": "救护车上常用的呼吸机、除颤仪，属于二类医疗器械。",
        }
    ]


def test_merge_subtitle_segments_keeps_continuation_chain_with_its_ending():
    segments = [
        {
            "start": 93000,
            "end": 95500,
            "text": "而君逸公司连医疗器械经营资质都不具备，",
        },
        {"start": 95700, "end": 99900, "text": "救护车上常用的呼吸机、"},
        {"start": 100140, "end": 100500, "text": "除颤仪，"},
        {"start": 100620, "end": 101580, "text": "属于二类医疗器械。"},
    ]

    assert cli.merge_subtitle_segments(segments) == [
        {
            "start": 93000,
            "end": 95500,
            "text": "而君逸公司连医疗器械经营资质都不具备，",
        },
        {
            "start": 95700,
            "end": 101580,
            "text": "救护车上常用的呼吸机、除颤仪，属于二类医疗器械。",
        },
    ]


def test_merge_subtitle_segments_repacks_false_sentence_boundary_after_tiny_gap():
    segments = [
        {"start": 148930, "end": 149830, "text": "就在八月二十五日，"},
        {
            "start": 150010,
            "end": 151930,
            "text": "上汽大通直接向平台发起投诉，",
        },
        {
            "start": 152170,
            "end": 153970,
            "text": "投诉对象是那些转述基金会。",
        },
        {"start": 154030, "end": 155050, "text": "声明的自媒体文章，"},
        {
            "start": 155470,
            "end": 157330,
            "text": "上汽大通的投诉内容写的很清楚，",
        },
    ]

    assert cli.merge_subtitle_segments(segments) == [
        {
            "start": 148930,
            "end": 151930,
            "text": "就在八月二十五日，上汽大通直接向平台发起投诉，",
        },
        {
            "start": 152170,
            "end": 157330,
            "text": "投诉对象是那些转述基金会。声明的自媒体文章，上汽大通的投诉内容写的很清楚，",
        },
    ]


def test_merge_subtitle_segments_groups_two_character_sentence():
    segments = [
        {"start": 12450, "end": 12690, "text": "突然。"},
        {"start": 12750, "end": 15150, "text": "捐十万立刻被全网吹成仗义好人。"},
    ]

    assert cli.merge_subtitle_segments(segments) == [
        {
            "start": 12450,
            "end": 15150,
            "text": "突然。捐十万立刻被全网吹成仗义好人。",
        }
    ]


def test_merge_subtitle_segments_preserves_hard_boundaries():
    segments = [
        {"start": 0, "end": 1200, "text": "第一句。", "speaker": 0},
        {"start": 1200, "end": 2600, "text": "第二句。", "speaker": 0},
        {"start": 2700, "end": 3000, "text": "甲，", "speaker": 0},
        {"start": 3100, "end": 3600, "text": "乙。", "speaker": 1},
        {"start": 5000, "end": 5300, "text": "丙，", "speaker": 1},
        {"start": 6000, "end": 6500, "text": "丁。", "speaker": 1},
    ]

    assert cli.merge_subtitle_segments(segments) == segments
