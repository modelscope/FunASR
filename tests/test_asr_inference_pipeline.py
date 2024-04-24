import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class TestConformerInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_aishell1(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_aishell2(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))


class TestData2vecInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_transformer(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))
        assert rec_result["text"] == "每一天都要快乐喔"

    def test_paraformer(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_data2vec_pretrain-paraformer-zh-cn-aishell2-16k",
        )
        rec_result = inference_pipeline(
            audio_in="https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))
        assert rec_result["text"] == "每一天都要快乐喔"


class TestMfccaInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_alimeeting(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950",
            model_revision="v3.0.0",
        )
        rec_result = inference_pipeline(
            audio_in="https://pre.modelscope.cn/api/v1/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/repo?Revision=master&FilePath=example/asr_example_mc.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))


class TestParaformerInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_paraformer_large_contextual_common(self):
        param_dict = dict()
        param_dict["hotword"] = (
            "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"
        )
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
            param_dict=param_dict,
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))
        assert rec_result["text"] == "国务院发展研究中心市场经济研究所副所长邓郁松认为"

    def test_paraformer_large_aishell1(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell1-vocab8404-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))
        assert rec_result["text"] == "欢迎大家来体验达摩院推出的语音识别模型"

    def test_paraformer_large_aishell2(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell2-vocab8404-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))
        assert rec_result["text"] == "欢迎大家来体验达摩院推出的语音识别模型"

    def test_paraformer_large_common(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))
        assert rec_result["text"] == "欢迎大家来体验达摩院推出的语音识别模型"

    def test_paraformer_large_online_common(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
            model_revision="v1.0.6",
            update_model=False,
            mode="paraformer_fake_streaming",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))
        assert rec_result["text"] == "欢迎大家来体验达摩院推出的语音识别模型"

    def test_paraformer_online_common(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online",
            model_revision="v1.0.6",
            update_model=False,
            mode="paraformer_fake_streaming",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))
        assert rec_result["text"] == "欢迎大家来体验达摩院推出的语音识别模型"

    def test_paraformer_tiny_commandword(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh_command.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_paraformer_8k(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer_asr_nat-zh-cn-8k-common-vocab8358-tensorflow1",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_8K.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_paraformer_aishell1(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_paraformer_aishell2(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))


class TestParaformerBertInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_aishell1(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_aishell2(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))


class TestUniasrInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_uniasr_2pass_cantonese_chs_16k_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cantonese-CHS.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_cantonese_chs_16k_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_cantonese-CHS.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_cn_dialect_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-cn-dialect-16k-vocab8358-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_cn_dialect_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-cn-dialect-16k-vocab8358-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_de_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_de.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_de_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_de.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_en_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_en_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-en-16k-common-vocab1080-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_es_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_es.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_es_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-es-16k-common-vocab3445-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_es.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_fa_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_fa.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_fa_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_fa.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_fr_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_fr.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_fr_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-fr-16k-common-vocab3472-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_fr.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_id_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_id.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_id_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-id-16k-common-vocab1067-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_id.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_ja_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ja.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_ja_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-ja-16k-common-vocab93-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ja.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_ko_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ko.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_ko_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-ko-16k-common-vocab6400-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ko.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_minnan_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-minnan-16k-common-vocab3825",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_pt_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_pt.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_pt_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-pt-16k-common-vocab1617-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_pt.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_ru_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ru.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_ru_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-ru-16k-common-vocab1664-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_ru.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_vi_common_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_vi.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_vi_common_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-vi-16k-common-vocab1001-pytorch-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_vi.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_zhcn_8k_common_vocab3445_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_zhcn_8k_common_vocab3445_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_zhcn_8k_common_vocab8358_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab8358-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_zhcn_8k_common_vocab8358_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab8358-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_zhcn_16k_common_vocab8358_offline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
            param_dict={"decoding_model": "offline"},
        )
        logger.info("asr inference result: {0}".format(rec_result))

    def test_uniasr_2pass_zhcn_16k_common_vocab8358_online(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
            param_dict={"decoding_model": "normal"},
        )
        logger.info("asr inference result: {0}".format(rec_result))


if __name__ == "__main__":
    unittest.main()
