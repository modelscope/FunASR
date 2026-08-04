from funasr.download.name_maps_from_hub import name_maps_hf


def test_hf_paraformer_en_alias_targets_english_checkpoint():
    assert name_maps_hf["paraformer-en"] == "funasr/paraformer-en"
