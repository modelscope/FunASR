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


def test_merge_subtitle_segments_splits_overlong_source_with_token_timestamps():
    text = "甲" * 60
    timestamps = [[index * 300, index * 300 + 120] for index in range(len(text))]
    segments = [
        {
            "start": timestamps[0][0],
            "end": timestamps[-1][1],
            "text": text,
            "timestamp": timestamps,
        }
    ]

    cues = cli.merge_subtitle_segments(segments)

    assert len(cues) > 1
    assert "".join(cue["text"] for cue in cues) == text
    assert [timestamp for cue in cues for timestamp in cue["timestamp"]] == timestamps
    assert all(cue["end"] - cue["start"] <= 8000 for cue in cues)
    assert all(len(cue["text"]) <= 42 for cue in cues)


def test_merge_subtitle_segments_keeps_word_timestamp_boundaries():
    timestamps = [[0, 900], [1000, 1900], [2000, 2900]]
    segments = [
        {
            "start": 0,
            "end": 2900,
            "text": "hello, world! again",
            "timestamp": timestamps,
        }
    ]

    assert cli.merge_subtitle_segments(
        segments, max_duration_ms=2000, max_chars=13
    ) == [
        {
            "start": 0,
            "end": 1900,
            "text": "hello, world!",
            "timestamp": timestamps[:2],
        },
        {
            "start": 2000,
            "end": 2900,
            "text": "again",
            "timestamp": timestamps[2:],
        },
    ]


def test_merge_subtitle_segments_preserves_overlong_source_without_timestamps():
    segment = {"start": 0, "end": 12000, "text": "没有时间戳的长句" * 8}

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_preserves_unalignable_token_timestamps():
    segment = {
        "start": 0,
        "end": 12000,
        "text": "어디 통화하고 있는 거네 지금",
        "timestamp": [[index * 1000, index * 1000 + 900] for index in range(8)],
    }

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_keeps_indivisible_overlong_token():
    segment = {
        "start": 0,
        "end": 12000,
        "text": "supercalifragilisticexpialidocious",
        "timestamps": [[0, 12000]],
        "words": ["supercalifragilisticexpialidocious"],
    }

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_preserves_source_around_indivisible_token():
    segment = {
        "start": 0,
        "end": 14000,
        "text": "before supercalifragilisticexpialidocious after",
        "timestamps": [[0, 900], [1000, 13000], [13100, 14000]],
        "words": ["before", "supercalifragilisticexpialidocious", "after"],
    }

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_rejects_partially_invalid_timestamps():
    segment = {
        "start": 0,
        "end": 12000,
        "text": "hello world",
        "timestamp": [[0, 900], [1000], [11100, 12000]],
    }

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_rejects_out_of_order_timestamps():
    segment = {
        "start": 0,
        "end": 12000,
        "text": "hello world again",
        "timestamp": [[0, 1000], [5000, 6000], [2000, 3000]],
    }

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_rejects_negative_timestamps():
    segment = {
        "start": 0,
        "end": 12000,
        "text": "hello world",
        "timestamp": [[-100, 3900], [4000, 7900]],
    }

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_rejects_non_finite_timestamps():
    segment = {
        "start": 0,
        "end": 12000,
        "text": "hello",
        "timestamp": [[0, float("inf")]],
    }

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_does_not_infer_unsupported_script_surfaces():
    segment = {
        "start": 0,
        "end": 12000,
        "text": "مرح",
        "timestamp": [[0, 3900], [4000, 7900], [8000, 12000]],
    }

    assert cli.merge_subtitle_segments([segment]) == [segment]


def test_merge_subtitle_segments_uses_explicit_word_surfaces():
    timestamps = [[index * 1000, index * 1000 + 900] for index in range(8)]
    segment = {
        "start": 0,
        "end": 7900,
        "text": "어디 통화하고 있는 거네 지금",
        "timestamp": timestamps,
        "words": ["어", "디", "통화", "하고", "있는", "거", "네", "지금"],
    }

    assert cli.merge_subtitle_segments(
        [segment], max_duration_ms=3900, max_chars=42
    ) == [
        {
            "start": 0,
            "end": 3900,
            "text": "어디 통화하고",
            "timestamp": timestamps[:4],
        },
        {
            "start": 4000,
            "end": 7900,
            "text": "있는 거네 지금",
            "timestamp": timestamps[4:],
        },
    ]


def test_sentence_timestamp_words_tracks_global_word_surfaces():
    result = {
        "words": ["hello", "世", "界", "▁진짜"],
        "timestamp": [[0, 100], [110, 210], [220, 320], [330, 430]],
        "sentence_info": [
            {"timestamp": [[0, 100], [110, 210], [220, 320]]},
            {"timestamp": [[330, 430]]},
        ],
    }

    assert cli._sentence_timestamp_words(result) == [
        ["hello", "世", "界"],
        ["▁진짜"],
    ]


def test_merge_subtitle_segments_combines_explicit_word_surfaces():
    segments = [
        {
            "start": 0,
            "end": 900,
            "text": "hello,",
            "timestamp": [[0, 900]],
            "words": ["hello"],
        },
        {
            "start": 1000,
            "end": 1900,
            "text": "world",
            "timestamp": [[1000, 1900]],
            "words": ["world"],
        },
    ]

    assert cli.merge_subtitle_segments(segments) == [
        {
            "start": 0,
            "end": 1900,
            "text": "hello,world",
            "timestamp": [[0, 900], [1000, 1900]],
            "words": ["hello", "world"],
        }
    ]


def test_merge_subtitle_segments_drops_incomplete_word_surfaces():
    segments = [
        {
            "start": 0,
            "end": 900,
            "text": "hello,",
            "timestamp": [[0, 900]],
            "words": ["hello"],
        },
        {
            "start": 1000,
            "end": 1900,
            "text": "world",
            "timestamp": [[1000, 1900]],
            "words": [],
        },
    ]

    assert cli.merge_subtitle_segments(segments) == [
        {
            "start": 0,
            "end": 1900,
            "text": "hello,world",
            "timestamp": [[0, 900], [1000, 1900]],
        }
    ]


def test_merge_subtitle_segments_keeps_chinese_words_and_balances_long_source():
    text = (
        "百分之九十九看过寄生虫的人都极易忽略一个藏着阶级真相与人性褶皱的细节"
        "当金家母亲钟书成为朴家保姆后他给正在辅导多会英语的儿子基语送水果时"
        "不仅毫无顾忌的未敲门还轻易的偷偷扭了一下儿子的耳朵可转身给辅导多送的"
        "女儿基婷送水果时却被基婷厉声斥责质问他为何不敲门就擅自闯入这段看似"
        "无关紧要的日常是凤俊浩用极简镜头精准刻画人物特质阶级心里的神来之笔"
        "基婷斥责母亲从"
    )
    timestamps = [[index * 180, index * 180 + 120] for index in range(len(text))]

    cues = cli.merge_subtitle_segments(
        [
            {
                "start": timestamps[0][0],
                "end": timestamps[-1][1],
                "text": text,
                "timestamp": timestamps,
            }
        ]
    )

    assert "".join(cue["text"] for cue in cues) == text
    assert all(len(cue["text"]) <= 42 for cue in cues)
    assert all(cue["end"] - cue["start"] <= 8000 for cue in cues)
    assert min(len(cue["text"]) for cue in cues) >= 24

    boundaries = {
        "".join(cue["text"] for cue in cues[:index])
        for index in range(1, len(cues))
    }
    for phrase in ("成为", "扭了一下", "无关紧要", "神来之笔"):
        start = text.index(phrase)
        assert not any(start < len(boundary) < start + len(phrase) for boundary in boundaries)


def test_merge_subtitle_segments_avoids_single_character_phrase_breaks():
    phrase = "钟书成为朴家保姆后他轻易的偷偷扭了一下儿子的耳朵"
    text = phrase * 4
    timestamps = [[index * 180, index * 180 + 120] for index in range(len(text))]

    cues = cli.merge_subtitle_segments(
        [{"start": 0, "end": timestamps[-1][1], "text": text, "timestamp": timestamps}]
    )

    boundary_offsets = []
    offset = 0
    for cue in cues[:-1]:
        offset += len(cue["text"])
        boundary_offsets.append(offset)

    search_from = 0
    while True:
        phrase_start = text.find("扭了一下", search_from)
        if phrase_start < 0:
            break
        assert not any(
            phrase_start < boundary < phrase_start + len("扭了一下")
            for boundary in boundary_offsets
        )
        search_from = phrase_start + 1
