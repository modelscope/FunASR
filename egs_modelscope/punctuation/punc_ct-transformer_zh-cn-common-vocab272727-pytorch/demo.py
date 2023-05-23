
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    model_revision="v1.1.7",
    output_dir="./tmp/"
)

##################text.scp###################
# inputs = "./egs_modelscope/punctuation/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/data/punc_example.txt"

##################text#####################
#inputs = "我们都是木头人不会讲话不会动"

##################text file url#######################
inputs = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt"

rec_result = inference_pipeline(text_in=inputs)
print(rec_result)
