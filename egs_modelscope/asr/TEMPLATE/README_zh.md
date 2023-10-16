(简体中文|[English](./README.md))

# 语音识别

> **注意**:
> pipeline 支持 [modelscope模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) 中的所有模型进行推理和微调。这里我们以典型模型作为示例来演示使用方法。

## 推理

### 快速使用
#### [Paraformer 模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
```
#### [Paraformer-实时模型](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary)
##### 实时推理
```python
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision='v1.0.7',
    update_model=False,
    mode='paraformer_streaming'
    )
import soundfile
speech, sample_rate = soundfile.read("example/asr_example.wav")

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention
param_dict = {"cache": dict(), "is_final": False, "chunk_size": chunk_size,
              "encoder_chunk_look_back": encoder_chunk_look_back, "decoder_chunk_look_back": decoder_chunk_look_back}
chunk_stride = chunk_size[1] * 960 # 600ms、480ms
# first chunk, 600ms
speech_chunk = speech[0:chunk_stride]
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
# next chunk, 600ms
speech_chunk = speech[chunk_stride:chunk_stride+chunk_stride]
rec_result = inference_pipeline(audio_in=speech_chunk, param_dict=param_dict)
print(rec_result)
```

##### 伪实时推理
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision='v1.0.7',
    update_model=False,
    mode="paraformer_fake_streaming"
)
audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav'
rec_result = inference_pipeline(audio_in=audio_in)
print(rec_result)
```
演示代码完整版本，请参考[demo](https://github.com/alibaba-damo-academy/FunASR/discussions/241)

#### [Paraformer-contextual Model](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

param_dict = dict()
# param_dict['hotword'] = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"
param_dict['hotword']="邓郁松 王颖春 王晔君"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    param_dict=param_dict)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_hotword.wav')
print(rec_result)
```

#### [UniASR 模型](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online/summary)
UniASR 模型有三种解码模式(fast、normal、offline)，更多模型细节请参考[文档](https://www.modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online/summary)
```python
decoding_model = "fast" # "fast"、"normal"、"offline"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_UniASR_asr_2pass-minnan-16k-common-vocab3825',
    param_dict={"decoding_model": decoding_model})

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
```
fast 和 normal 的解码模式是假流式解码，可用于评估识别准确性。
演示的完整代码，请参见 [demo](https://github.com/alibaba-damo-academy/FunASR/discussions/151)

#### [Paraformer-Spk model](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary)
返回识别结果的同时返回每个子句的说话人分类结果。关于说话人日志模型的详情请见[CAM++](https://modelscope.cn/models/damo/speech_campplus_speaker-diarization_common/summary)。

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    audio_in = 'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_speaker_demo.wav'
    output_dir = "./results"
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn',
        model_revision='v0.0.2',
        vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        punc_model='damo/punc_ct-transformer_cn-en-common-vocab471067-large',
        output_dir=output_dir,
    )
    rec_result = inference_pipeline(audio_in=audio_in, batch_size_token=5000, batch_size_token_threshold_s=40, max_single_segment_time=6000)
    print(rec_result)
```


#### [RNN-T-online 模型]()
Undo

#### [MFCCA 模型](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary)

更多模型细节请参考[文档](https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950',
    model_revision='v3.0.0'
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
```

### API接口说明
#### pipeline定义
- `task`: `Tasks.auto_speech_recognition`
- `model`: [模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) 中的模型名称，或本地磁盘中的模型路径
- `ngpu`: `1`（默认），使用 GPU 进行推理。如果 ngpu=0，则使用 CPU 进行推理
- `ncpu`: `1` （默认），设置用于 CPU 内部操作并行性的线程数
- `output_dir`: `None` （默认），如果设置，输出结果的输出路径
- `batch_size`: `1` （默认），解码时的批处理大小
#### pipeline 推理
- `audio_in`: 要解码的输入，可以是：
  - wav文件路径, 例如: asr_example.wav,
  - pcm文件路径, 例如: asr_example.pcm,
  - 音频字节数流，例如：麦克风的字节数数据
  - 音频采样点，例如：`audio, rate = soundfile.read("asr_example_zh.wav")`, 数据类型为 numpy.ndarray 或者 torch.Tensor
  - wav.scp，kaldi 样式的 wav 列表 (`wav_id \t wav_path`), 例如:
  ```text
  asr_example1  ./audios/asr_example1.wav
  asr_example2  ./audios/asr_example2.wav
  ```
  在这种输入 `wav.scp` 的情况下，必须设置 `output_dir` 以保存输出结果
- `audio_fs`: 音频采样率，仅在 audio_in 为 pcm 音频时设置
- `output_dir`: None （默认），如果设置，输出结果的输出路径

### 使用多线程 CPU 或多个 GPU 进行推理
FunASR 还提供了 [egs_modelscope/asr/TEMPLATE/infer.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE/infer.sh) 脚本，以使用多线程 CPU 或多个 GPU 进行解码。

#### `infer.sh` 设置
- `model`: [modelscope模型仓库](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope)中的模型名称，或本地磁盘中的模型路径
- `data_dir`: 数据集目录需要包括 `wav.scp` 文件。如果 `${data_dir}/text` 也存在，则将计算 CER
- `output_dir`: 识别结果的输出目录
- `batch_size`: `64`（默认），在 GPU 上进行推理的批处理大小
- `gpu_inference`: `true` （默认），是否执行 GPU 解码，如果进行 CPU 推理，则设置为 `false`
- `gpuid_list`: `0,1` （默认），用于推理的 GPU ID
- `njob`: 仅用于 CPU 推理（`gpu_inference=false`），`64`（默认），CPU 解码的作业数
- `checkpoint_dir`: 仅用于推理微调模型，微调模型的路径目录
- `checkpoint_name`: 仅用于推理微调模型，`valid.cer_ctc.ave.pb`（默认），用于推理的检查点
- `decoding_mode`: `normal`（默认），UniASR 模型的解码模式（`fast`、`normal`、`offline`）
- `hotword_txt`: `None` （默认），上下文语料库模型的热词文件（热词文件名以 .txt 结尾）

#### 使用多个 GPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 64 \
    --gpu_inference true \
    --gpuid_list "0,1"
```
#### 使用多线程 CPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 64
```

#### 推理结果
解码结果可以在 `$output_dir/1best_recog/text.cer` 中找到，其中包括每个样本的识别结果和整个测试集的 CER 指标。
如果您对 SpeechIO 测试集进行解码，则可以使用 `stage=3` 的 textnorm，`DETAILS.txt` 和 `RESULTS.txt` 记录了文本标准化后的结果和 CER。

## 使用pipeline进行微调

### 快速上手
[finetune.py](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE/finetune.py)
```python
import os

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

from funasr.datasets.ms_dataset import MsDataset
from funasr.utils.modelscope_param import modelscope_args


def modelscope_finetune(params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = MsDataset.load(params.data_path)
    kwargs = dict(
        model=params.model,
        data_dir=ds_dict,
        dataset_type=params.dataset_type,
        work_dir=params.output_dir,
        batch_bins=params.batch_bins,
        max_epoch=params.max_epoch,
        lr=params.lr,
        mate_params=params.param_dict)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    params = modelscope_args(model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    params.output_dir = "./checkpoint"                 # m模型保存路径
    params.data_path = "speech_asr_aishell1_trainsets" # 数据路径
    params.dataset_type = "small"                      # 小数据量设置small，若数据量大于1000小时，请使用large
    params.batch_bins = 2000                           # batch size，如果dataset_type="small"，batch_bins单位为fbank特征帧数，如果dataset_type="large"，batch_bins单位为毫秒，
    params.max_epoch = 20                              # 最大训练轮数
    params.lr = 0.00005                                # 设置学习率
    init_param = []                                    # 初始模型路径，默认加载modelscope模型初始化，例如: ["checkpoint/20epoch.pb"]
    freeze_param = []                                  # 模型参数freeze, 例如: ["encoder"]
    ignore_init_mismatch = True                        # 是否忽略模型参数初始化不匹配
    use_lora = False                                   # 是否使用lora进行模型微调
    params.param_dict = {"init_param":init_param, "freeze_param": freeze_param, "ignore_init_mismatch": ignore_init_mismatch}
    if use_lora:
        enable_lora = True
        lora_bias = "all"
        lora_params = {"lora_list":['q','v'], "lora_rank":8, "lora_alpha":16, "lora_dropout":0.1}
        lora_config = {"enable_lora": enable_lora, "lora_bias": lora_bias, "lora_params": lora_params}
        params.param_dict.update(lora_config)

    modelscope_finetune(params)
```

```shell
python finetune.py &> log.txt &
```

### 使用私有数据进行微调

- 修改 [finetune.py](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE/finetune.py) 中微调训练相关参数
    - `output_dir`: 微调模型保存路径
    - `data_dir`: 数据集目录需要包括以下文件：`train/wav.scp`, `train/text`; `validation/wav.scp`, `validation/text`
    - `dataset_type`: 对于大于 1000 小时的数据集，设置为 `large`，否则设置为 `small`
    - `batch_bins`: 批处理大小。对于 `dataset_type` 为 `small`，`batch_bins` 表示特征帧数。对于 `dataset_type` 为 `large`，`batch_bins` 表示以毫秒为单位的持续时间
    - `max_epoch`: 最大训练 epoch 数量
    - `lr`: 学习率
    - `init_param`: `[]`（默认值），初始化模型路径，按默认设置加载 modelscope 模型初始化。例如：["checkpoint/20epoch.pb"]
    - `freeze_param`: `[]`（默认值），冻结模型参数。例如：["encoder"]
    - `ignore_init_mismatch`: `True`（默认值），在加载预训练模型时忽略大小不匹配
    - `use_lora`: `False`（默认值），微调模型使用 LORA，请参阅 [LORA论文](https://arxiv.org/pdf/2106.09685.pdf)

- 训练数据格式
```sh
cat ./example_data/text
BAC009S0002W0122 而 对 楼 市 成 交 抑 制 作 用 最 大 的 限 购
BAC009S0002W0123 也 成 为 地 方 政 府 的 眼 中 钉
english_example_1 hello world
english_example_2 go swim 去 游 泳

cat ./example_data/wav.scp
BAC009S0002W0122 /mnt/data/wav/train/S0002/BAC009S0002W0122.wav
BAC009S0002W0123 /mnt/data/wav/train/S0002/BAC009S0002W0123.wav
english_example_1 /mnt/data/wav/train/S0002/english_example_1.wav
english_example_2 /mnt/data/wav/train/S0002/english_example_2.wav
```

- 然后，您可以使用以下命令运行pipeline进行微调：
```shell
python finetune.py
```
如果您想使用多个 GPU 进行微调，可以使用以下命令：
```shell
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 finetune.py > log.txt 2>&1
```
## 使用微调模型进行推理

[egs_modelscope/asr/TEMPLATE/infer.sh](https://github.com/alibaba-damo-academy/FunASR/blob/main/egs_modelscope/asr/TEMPLATE/infer.sh) 参数设置与上面`infer.sh`相同

- 使用多个 GPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --batch_size 64 \
    --gpu_inference true \
    --gpuid_list "0,1" \
    --checkpoint_dir "./checkpoint" \
    --checkpoint_name "valid.cer_ctc.ave.pb"
```
- 使用多线程 CPU 进行解码：
```shell
    bash infer.sh \
    --model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" \
    --data_dir "./data/test" \
    --output_dir "./results" \
    --gpu_inference false \
    --njob 64 \
    --checkpoint_dir "./checkpoint" \
    --checkpoint_name "valid.cer_ctc.ave.pb"
```
