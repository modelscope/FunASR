(简体中文|[English](./README.md))
# 标点预测

> **Note**: 
> Pipeline 支持在[modelscope模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope)中的所有模型进行推理和微调。在这里，我们以 CT-Transformer 模型为例来演示使用方法。

## 推理

### 快速使用
#### [CT-Transformer 模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    model_revision=None)

rec_result = inference_pipeline(text_in='example/punc_example.txt')
print(rec_result)
```
- text二进制数据，例如：用户直接从文件里读出bytes数据
```python
rec_result = inference_pipeline(text_in='我们都是木头人不会讲话不会动')
```
- text文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt
```python
rec_result = inference_pipeline(text_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/punc_example.txt')
```

#### [CT-Transformer 实时模型](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727',
    model_revision=None,
)

inputs = "跨境河流是养育沿岸|人民的生命之源长期以来为帮助下游地区防灾减灾中方技术人员|在上游地区极为恶劣的自然条件下克服巨大困难甚至冒着生命危险|向印方提供汛期水文资料处理紧急事件中方重视印方在跨境河流问题上的关切|愿意进一步完善双方联合工作机制|凡是|中方能做的我们|都会去做而且会做得更好我请印度朋友们放心中国在上游的|任何开发利用都会经过科学|规划和论证兼顾上下游的利益"
vads = inputs.split("|")
rec_result_all="outputs:"
param_dict = {"cache": []}
for vad in vads:
    rec_result = inference_pipeline(text_in=vad, param_dict=param_dict)
    rec_result_all += rec_result['text']

print(rec_result_all)
```
演示例子完整代码请参考 [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/238)

### API接口说明
#### pipeline定义
- `task`: `Tasks.punctuation`
- `model`: [模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) 中的模型名称，或本地磁盘中的模型路径
- `ngpu`: `1`（默认），使用 GPU 进行推理。如果 ngpu=0，则使用 CPU 进行推理
- `ncpu`: `1` （默认），设置用于 CPU 内部操作并行性的线程数
- `output_dir`: `None` （默认），如果设置，输出结果的输出路径
- `model_revision`: `None`（默认），modelscope中版本版本号


#### pipeline推理
- `text_in`: 需要进行推理的输入，支持一下输入：
  - 文本字符，例如："我们都是木头人不会讲话不会动"
  - 文本文件，例如：example/punc_example.txt。
  在使用文本文件 输入时，必须设置 `output_dir` 以保存输出结果。
- `param_dict`: 在实时模式下必要的缓存。

### Inference with multi-thread CPUs or multi GPUs
FunASR 还提供了 [egs_modelscope/punctuation/TEMPLATE/infer.sh](infer.sh) 脚本，以使用多线程 CPU 或多个 GPU 进行解码。

#### `infer.sh` 设置
- `model`: [modelscope模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope)中的模型名称，或本地磁盘中的模型路径
- `data_dir`: 数据集目录需要包括 `punc.txt` 文件
- `output_dir`: 识别结果的输出目录
- `batch_size`: `1`（默认），在 GPU 上进行推理的批处理大小
- `gpu_inference`: `true` （默认），是否执行 GPU 解码，如果进行 CPU 推理，则设置为 `false`
- `gpuid_list`: `0,1` （默认），用于推理的 GPU ID
- `njob`: 仅用于 CPU 推理（`gpu_inference=false`），`64`（默认），CPU 解码的作业数


#### 使用多个 GPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 1 \
    --gpu_inference true \
    --gpuid_list "0,1"
```
#### 使用多线程 CPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 1
```

## Finetune with pipeline

### Quick start

### Finetune with your data

## Inference with your finetuned model

