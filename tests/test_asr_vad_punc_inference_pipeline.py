import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class TestParaformerInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_inference_pipeline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            model_revision="v1.2.1",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            vad_model_revision="v1.1.8",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            punc_model_revision="v1.1.6",
            ngpu=1,
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
        )
        logger.info("asr_vad_punc inference result: {0}".format(rec_result))
        assert rec_result["text"] == "欢迎大家来体验达摩院推出的语音识别模型。"


if __name__ == "__main__":
    unittest.main()
