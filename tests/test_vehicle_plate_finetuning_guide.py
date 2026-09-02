from pathlib import Path


DOC = Path("docs/vehicle_plate_finetuning_zh.md")


def test_vehicle_plate_guide_keeps_asr_and_extraction_boundaries_explicit():
    text = DOC.read_text(encoding="utf-8")

    for expected in (
        "2-12 秒",
        "逐字转写",
        "不是确定性纠错",
        "1,000",
        "3,000-5,000",
        "CER",
        "车牌片段 exact match",
        "整通话",
    ):
        assert expected in text
