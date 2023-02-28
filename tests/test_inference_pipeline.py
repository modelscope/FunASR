import unittest
import logging

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class TestInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os
        print("JIANGYU: run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_asr_inference_pipeline(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
        rec_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
        logging.info("asr inference result: {0}".format(rec_result))

    def test_asr_inference_pipeline_with_vad_punc(self):
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model='damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
            vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
            vad_model_revision="v1.1.8",
            punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
            punc_model_revision="v1.1.6")
        rec_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_vad_punc_example.wav')
        logging.info("asr inference with vad punc result: {0}".format(rec_result))

    def test_vad_inference_pipeline(self):
        inference_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
            model_revision='v1.1.8',
        )
        segments_result = inference_pipeline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav')
        logging.info("vad inference result: {0}".format(segments_result))


if __name__ == '__main__':
    unittest.main()
