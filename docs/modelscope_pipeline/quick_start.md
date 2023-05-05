# Quick Start

> **Note**: 
> The modelscope pipeline supports all the models in [model zoo](https://alibaba-damo-academy.github.io/FunASR/en/model_zoo/modelscope_models.html#pretrained-models-on-modelscope) to inference and finetine. Here we take typic model as example to demonstrate the usage.


## Inference with pipeline

### Speech Recognition
#### Paraformer Model
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
)

rec_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
# {'text': '欢迎大家来体验达摩院推出的语音识别模型'}
```

### Voice Activity Detection
#### FSMN-VAD Model
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
import logging
logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)

inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    )

segments_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav')
print(segments_result)
# {'text': [[70, 2340], [2620, 6200], [6480, 23670], [23950, 26250], [26780, 28990], [29950, 31430], [31750, 37600], [38210, 46900], [47310, 49630], [49910, 56460], [56740, 59540], [59820, 70450]]}
```

### Punctuation Restoration
#### CT_Transformer Model
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    )

rec_result = inference_pipeline(text_in='我们都是木头人不会讲话不会动')
print(rec_result)
# {'text': '我们都是木头人，不会讲话，不会动。'}
```

### Timestamp Prediction
#### TP-Aligner Model
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.speech_timestamp,
    model='damo/speech_timestamp_prediction-v1-16k-offline',)

rec_result = inference_pipeline(
    audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_timestamps.wav',
    text_in='一 个 东 太 平 洋 国 家 为 什 么 跑 到 西 太 平 洋 来 了 呢',)
print(rec_result)
# {'text': '<sil> 0.000 0.380;一 0.380 0.560;个 0.560 0.800;东 0.800 0.980;太 0.980 1.140;平 1.140 1.260;洋 1.260 1.440;国 1.440 1.680;家 1.680 1.920;<sil> 1.920 2.040;为 2.040 2.200;什 2.200 2.320;么 2.320 2.500;跑 2.500 2.680;到 2.680 2.860;西 2.860 3.040;太 3.040 3.200;平 3.200 3.380;洋 3.380 3.500;来 3.500 3.640;了 3.640 3.800;呢 3.800 4.150;<sil> 4.150 4.440;', 'timestamp': [[380, 560], [560, 800], [800, 980], [980, 1140], [1140, 1260], [1260, 1440], [1440, 1680], [1680, 1920], [2040, 2200], [2200, 2320], [2320, 2500], [2500, 2680], [2680, 2860], [2860, 3040], [3040, 3200], [3200, 3380], [3380, 3500], [3500, 3640], [3640, 3800], [3800, 4150]]}
```

### Speaker Verification
#### X-vector Model
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np

inference_sv_pipline = pipeline(
    task=Tasks.speaker_verification,
    model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
)

# embedding extract
spk_embedding = inference_sv_pipline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav')["spk_embedding"]

# speaker verification
rec_result = inference_sv_pipline(audio_in=('https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_enroll.wav','https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/sv_example_same.wav'))
print(rec_result["scores"][0])
# 0.8540499500025098
```

### Speaker Diarization
#### SOND Model
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_diar_pipline = pipeline(
    mode="sond_demo",
    num_workers=0,
    task=Tasks.speaker_diarization,
    diar_model_config="sond.yaml",
    model='damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch',
    model_revision="v1.0.3",
    sv_model="damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch",
    sv_model_revision="v1.0.0",
)

audio_list=[
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/record.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_A.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_B.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_B1.wav"
]

results = inference_diar_pipline(audio_in=audio_list)
print(results)
# {'text': 'spk1 [(0.8, 1.84), (2.8, 6.16), (7.04, 10.64), (12.08, 12.8), (14.24, 15.6)]\nspk2 [(0.0, 1.12), (1.68, 3.2), (4.48, 7.12), (8.48, 9.04), (10.56, 14.48), (15.44, 16.0)]'}
```

### FAQ
#### How to switch device from GPU to CPU with pipeline

The pipeline defaults to decoding with GPU (`ngpu=1`) when GPU is available. If you want to switch to CPU, you could set `ngpu=0`
```python
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    ngpu=0,
)
```

#### How to infer from local model path
Download model to local dir, by modelscope-sdk

```python
from modelscope.hub.snapshot_download import snapshot_download

local_dir_root = "./models_from_modelscope"
model_dir = snapshot_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', cache_dir=local_dir_root)
```

Or download model to local dir, by git lfs
```shell
git lfs install
# git clone https://www.modelscope.cn/<namespace>/<model-name>.git
git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git
```

Infer with local model path
```python
local_dir_root = "./models_from_modelscope/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=local_dir_root,
)
```

## Finetune with pipeline
### Speech Recognition
#### Paraformer Model

finetune.py
```python
import os
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.msdatasets.audio.asr_dataset import ASRDataset

def modelscope_finetune(params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = ASRDataset.load(params.data_path, namespace='speech_asr')
    kwargs = dict(
        model=params.model,
        data_dir=ds_dict,
        dataset_type=params.dataset_type,
        work_dir=params.output_dir,
        batch_bins=params.batch_bins,
        max_epoch=params.max_epoch,
        lr=params.lr)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    from funasr.utils.modelscope_param import modelscope_args
    params = modelscope_args(model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    params.output_dir = "./checkpoint"                      # 模型保存路径
    params.data_path = "speech_asr_aishell1_trainsets"      # 数据路径，可以为modelscope中已上传数据，也可以是本地数据
    params.dataset_type = "small"                           # 小数据量设置small，若数据量大于1000小时，请使用large
    params.batch_bins = 2000                                # batch size，如果dataset_type="small"，batch_bins单位为fbank特征帧数，如果dataset_type="large"，batch_bins单位为毫秒，
    params.max_epoch = 50                                   # 最大训练轮数
    params.lr = 0.00005                                     # 设置学习率
    
    modelscope_finetune(params)
```

```shell
python finetune.py &> log.txt &
```
tail log.txt
```
[bach-gpu011024008134] 2023-04-23 18:59:13,976 (e2e_asr_paraformer:467) INFO: enable sampler in paraformer, sampling_ratio: 0.75
[bach-gpu011024008134] 2023-04-23 18:59:48,924 (trainer:777) INFO: 2epoch:train:1-50batch:50num_updates: iter_time=0.008, forward_time=0.302, loss_att=0.186, acc=0.942, loss_pre=0.005, loss=0.192, backward_time=0.231, optim_step_time=0.117, optim0_lr0=7.484e-06, train_time=0.753
[bach-gpu011024008134] 2023-04-23 19:00:23,869 (trainer:777) INFO: 2epoch:train:51-100batch:100num_updates: iter_time=1.152e-04, forward_time=0.275, loss_att=0.184, acc=0.945, loss_pre=0.005, loss=0.189, backward_time=0.234, optim_step_time=0.117, optim0_lr0=7.567e-06, train_time=0.699
[bach-gpu011024008134] 2023-04-23 19:00:58,463 (trainer:777) INFO: 2epoch:train:101-150batch:150num_updates: iter_time=1.123e-04, forward_time=0.271, loss_att=0.204, acc=0.942, loss_pre=0.005, loss=0.210, backward_time=0.231, optim_step_time=0.116, optim0_lr0=7.651e-06, train_time=0.692
```

### FAQ
### Multi GPUs training and distributed training

If you want finetune with multi-GPUs, you could:
```shell
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 finetune.py > log.txt 2>&1
```

