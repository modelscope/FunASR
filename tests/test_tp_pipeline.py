import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

class TestTimestampPredictionPipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os
        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_inference_pipeline(self):
        inference_pipeline = pipeline(
            task=Tasks.speech_timestamp,
            model='damo/speech_timestamp_prediction-v1-16k-offline',
            model_revision='v1.1.0')

        rec_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_timestamps.wav',
            text_in='一 个 东 太 平 洋 国 家 为 什 么 跑 到 西 太 平 洋 来 了 呢',)
        print(rec_result)
        logger.info("punctuation inference result: {0}".format(rec_result))


if __name__ == '__main__':
    unittest.main()