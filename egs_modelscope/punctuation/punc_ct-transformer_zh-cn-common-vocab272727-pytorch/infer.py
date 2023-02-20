
##################text.scp文件路径###################
inputs = "./egs_modelscope/punctuation/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/data/punc_example.txt"

##################text二进制数据#####################
#inputs = "我们都是木头人不会讲话不会动"

##################text文件url#######################
#inputs = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt"


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    model_revision="v1.1.6",
    output_dir="./tmp/"
)

rec_result = inference_pipline(text_in=inputs)
print(rec_result)
