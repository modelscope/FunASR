

##################text二进制数据#####################
inputs = "hello 大 家 好 呀"

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.language_score_prediction,
    model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
    output_dir="./tmp/"
)

rec_result = inference_pipeline(text_in=inputs)
print(rec_result)

