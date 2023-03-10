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
            model='damo/speech_conformer_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch')
        rec_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
        logger.info("asr inference result: {0}".format(rec_result))

    def test_aishell2(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch')
        rec_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
        logger.info("asr inference result: {0}".format(rec_result))


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


class TestMfccaInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os
        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_alimeeting(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950',
            model_revision='v3.0.0')
        rec_result = inference_pipeline(
            audio_in='16:32https://pre.modelscope.cn/api/v1/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/repo?Revision=master&FilePath=example/asr_example_mc.wav')
        logger.info("asr inference result: {0}".format(rec_result))


class TestParaformerInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os
        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_paraformer_large_contextual_common(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404')
        rec_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.wav')
        logger.info("asr inference result: {0}".format(rec_result))


if __name__ == '__main__':
    unittest.main()
