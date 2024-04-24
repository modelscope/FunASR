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
            model="damo/speech_timestamp_prediction-v1-16k-offline",
            model_revision="v1.1.0",
        )

        rec_result = inference_pipeline(
            audio_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_timestamps.wav",
            text_in="一 个 东 太 平 洋 国 家 为 什 么 跑 到 西 太 平 洋 来 了 呢",
        )
        print(rec_result)
        logger.info("punctuation inference result: {0}".format(rec_result))
        assert rec_result == {
            "text": "<sil> 0.000 0.380;一 0.380 0.560;个 0.560 0.800;东 0.800 0.980;太 0.980 1.140;平 1.140 1.260;洋 1.260 1.440;国 1.440 1.680;家 1.680 1.920;<sil> 1.920 2.040;为 2.040 2.200;什 2.200 2.320;么 2.320 2.500;跑 2.500 2.680;到 2.680 2.860;西 2.860 3.040;太 3.040 3.200;平 3.200 3.380;洋 3.380 3.500;来 3.500 3.640;了 3.640 3.800;呢 3.800 4.150;<sil> 4.150 4.440;",
            "timestamp": [
                [380, 560],
                [560, 800],
                [800, 980],
                [980, 1140],
                [1140, 1260],
                [1260, 1440],
                [1440, 1680],
                [1680, 1920],
                [2040, 2200],
                [2200, 2320],
                [2320, 2500],
                [2500, 2680],
                [2680, 2860],
                [2860, 3040],
                [3040, 3200],
                [3200, 3380],
                [3380, 3500],
                [3500, 3640],
                [3640, 3800],
                [3800, 4150],
            ],
        }


if __name__ == "__main__":
    unittest.main()
