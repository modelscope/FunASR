import unittest

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class TestTransformerInferencePipelines(unittest.TestCase):
    def test_funasr_path(self):
        import funasr
        import os

        logger.info("run_dir:{0} ; funasr_path: {1}".format(os.getcwd(), funasr.__file__))

    def test_inference_pipeline(self):
        inference_pipeline = pipeline(
            task=Tasks.language_score_prediction,
            model="damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch",
        )
        rec_result = inference_pipeline(text_in="hello 大 家 好 呀")
        logger.info("lm inference result: {0}".format(rec_result))


if __name__ == "__main__":
    unittest.main()
