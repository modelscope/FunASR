import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

class TestData2vecInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os
        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_transformer(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch')
        rec_result = inference_pipeline(
            audio_in='https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav')
        logger.info("asr inference result: {0}".format(rec_result))

    def test_paraformer(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_data2vec_pretrain-paraformer-zh-cn-aishell2-16k')
        rec_result = inference_pipeline(
            audio_in='https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav')
        logger.info("asr inference result: {0}".format(rec_result))


if __name__ == '__main__':
    unittest.main()
