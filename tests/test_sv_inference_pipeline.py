import unittest

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
            model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
        )
        # 提取不同句子的说话人嵌入码
        rec_result = inference_sv_pipline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav')
        enroll = rec_result["spk_embedding"]

        rec_result = inference_sv_pipline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_same.wav')
        same = rec_result["spk_embedding"]

        rec_result = inference_sv_pipline(
            audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_different.wav')
        different = rec_result["spk_embedding"]

        # 对相同的说话人计算余弦相似度
        sv_threshold = 0.9465
        same_cos = np.sum(enroll * same) / (np.linalg.norm(enroll) * np.linalg.norm(same))
        same_cos = max(same_cos - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
        logger.info("Similarity: {}".format(same_cos))

        # 对不同的说话人计算余弦相似度
        diff_cos = np.sum(enroll * different) / (np.linalg.norm(enroll) * np.linalg.norm(different))
        diff_cos = max(diff_cos - sv_threshold, 0.0) / (1.0 - sv_threshold) * 100.0
        logger.info("Similarity: {}".format(diff_cos))


if __name__ == '__main__':
    unittest.main()
