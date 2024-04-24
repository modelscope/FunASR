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
            task=Tasks.punctuation,
            model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            model_revision="v1.1.7",
        )
        inputs = "./egs_modelscope/punctuation/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/data/punc_example.txt"
        rec_result = inference_pipeline(text_in=inputs)
        logger.info("punctuation inference result: {0}".format(rec_result))

    def test_vadrealtime_inference_pipeline(self):
        inference_pipeline = pipeline(
            task=Tasks.punctuation,
            model="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
        )
        inputs = "跨境河流是养育沿岸|人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员|在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险|向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切|愿意进一步完善双方联合工作机制|凡是|中方能做的我们|都会去做而且会做得更好我请印度朋友们放心中国在上游的|任何开发利用都会经过科学|规划和论证兼顾上下游的利益"
        vads = inputs.split("|")
        rec_result_all = "outputs:"
        param_dict = {"cache": []}
        for vad in vads:
            rec_result = inference_pipeline(text_in=vad, param_dict=param_dict)
            rec_result_all += rec_result["text"]
        logger.info("punctuation inference result: {0}".format(rec_result_all))


if __name__ == "__main__":
    unittest.main()
