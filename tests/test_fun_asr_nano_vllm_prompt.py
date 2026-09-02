from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM


def test_iso_language_aliases_use_documented_prompt_names():
    model = object.__new__(FunASRNanoVLLM)

    assert model._build_prompt_text(language="zh") == "语音转写成中文："
    assert model._build_prompt_text(language="en") == "语音转写成英文："


def test_custom_language_prompt_is_preserved():
    model = object.__new__(FunASRNanoVLLM)

    assert model._build_prompt_text(language="粤语") == "语音转写成粤语："
