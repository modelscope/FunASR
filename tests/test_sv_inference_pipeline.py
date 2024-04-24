import unittest

import numpy as np
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class TestXVectorInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_inference_pipeline(self):
        inference_sv_pipline = pipeline(
            task=Tasks.speaker_verification,
            model="damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch",
        )

        # the same speaker
        rec_result = inference_sv_pipline(
            audio_in=(
                "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav",
                "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_same.wav",
            )
        )
        assert (
            abs(rec_result["scores"][0] - 0.85) < 0.1 and abs(rec_result["scores"][1] - 0.14) < 0.1
        )
        logger.info(f"Similarity {rec_result['scores']}")

        # different speaker
        rec_result = inference_sv_pipline(
            audio_in=(
                "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav",
                "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_different.wav",
            )
        )
        assert abs(rec_result["scores"][0] - 0.0) < 0.1 and abs(rec_result["scores"][1] - 1.0) < 0.1
        logger.info(f"Similarity {rec_result['scores']}")


if __name__ == "__main__":
    unittest.main()
