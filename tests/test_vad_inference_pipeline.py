import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

class TestFSMNInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os
        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_8k(self):
        inference_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model="damo/speech_fsmn_vad_zh-cn-8k-common",
        )
        rec_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example_8k.wav')
        logger.info("vad inference result: {0}".format(rec_result))

    def test_16k(self):
        inference_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav')
        logger.info("vad inference result: {0}".format(rec_result))


if __name__ == '__main__':
    unittest.main()
