
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_cn-en-common-vocab471067-large',
    model_revision="v1.0.0",
    output_dir="./tmp/"
)

##################text.scp###################
# inputs = "./egs_modelscope/punctuation/punc_ct-transformer_cn-en-common-vocab471067-large/data/punc_example.txt"

##################text#####################
#inputs = "我们都是木头人不会讲话不会动"

##################text file url#######################
inputs = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt"

rec_result = inference_pipeline(text_in=inputs)
print(rec_result)
