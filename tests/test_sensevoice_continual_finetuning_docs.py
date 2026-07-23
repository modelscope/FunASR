from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SENSEVOICE_DIR = ROOT / "examples" / "industrial_data_pretraining" / "sense_voice"


def test_readmes_link_continual_finetuning_guides():
    readme = (SENSEVOICE_DIR / "README.md").read_text(encoding="utf-8")
    readme_zh = (SENSEVOICE_DIR / "README_zh.md").read_text(encoding="utf-8")

    assert "[Continual fine-tuning](CONTINUAL_FINETUNING.md)" in readme
    assert "[持续微调指南](CONTINUAL_FINETUNING_zh.md)" in readme_zh


def test_english_guide_covers_retention_controls_and_worked_example():
    guide = (SENSEVOICE_DIR / "CONTINUAL_FINETUNING.md").read_text(encoding="utf-8")

    required = [
        "cannot guarantee zero regression",
        "speaker-disjoint",
        "pseudo-label",
        "replay:new",
        "1:1",
        "2:1",
        "3:1",
        '++freeze_param="encoder"',
        '++freeze_param="encoder.encoders0,encoder.encoders.0,encoder.encoders.1,encoder.encoders.2"',
        "++optim_conf.lr=0.00002",
        "<|zh|>",
        "<|teochew|>",
        "8.45",
        "16.9",
        "Mandarin",
        "Cantonese",
        "Teochew",
    ]
    assert all(item in guide for item in required)


def test_chinese_guide_covers_retention_controls_and_worked_example():
    guide = (SENSEVOICE_DIR / "CONTINUAL_FINETUNING_zh.md").read_text(encoding="utf-8")

    required = [
        "不能保证零退化",
        "说话人不重叠",
        "伪标签",
        "replay:new",
        "1:1",
        "2:1",
        "3:1",
        '++freeze_param="encoder"',
        '++freeze_param="encoder.encoders0,encoder.encoders.0,encoder.encoders.1,encoder.encoders.2"',
        "++optim_conf.lr=0.00002",
        "<|zh|>",
        "<|teochew|>",
        "8.45",
        "16.9",
        "普通话",
        "粤语",
        "潮汕话",
    ]
    assert all(item in guide for item in required)
