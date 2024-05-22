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
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example_8k.wav"
        )
        logger.info("vad inference result: {0}".format(rec_result))
        assert rec_result["text"] == [
            [0, 1960],
            [2870, 6730],
            [7960, 10180],
            [12140, 14830],
            [15740, 19400],
            [20220, 24230],
            [25540, 27290],
            [30070, 30970],
            [32070, 34280],
            [35990, 37050],
            [39400, 41020],
            [41810, 47320],
            [48120, 52150],
            [53560, 58310],
            [59290, 62210],
            [63110, 66420],
            [67300, 68280],
            [69670, 71770],
            [73100, 75550],
            [76850, 78500],
            [79380, 83280],
            [85000, 92320],
            [93560, 94110],
            [94990, 95620],
            [96940, 97590],
            [98400, 100530],
            [101600, 104890],
            [108780, 110900],
            [112020, 113460],
            [114210, 115030],
        ]

    def test_16k(self):
        inference_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        )
        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav"
        )
        logger.info("vad inference result: {0}".format(rec_result))
        assert rec_result["text"] == [
            [70, 2340],
            [2620, 6200],
            [6480, 23670],
            [23950, 26250],
            [26780, 28990],
            [29950, 31430],
            [31750, 37600],
            [38210, 46900],
            [47310, 49630],
            [49910, 56460],
            [56740, 59540],
            [59820, 70450],
        ]


if __name__ == "__main__":
    unittest.main()
