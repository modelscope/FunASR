# Quick Start

## Inference with pipeline

### Speech Recognition
#### Paraformer model
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

### Voice Activity Detection
#### FSMN-VAD
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
```

### Punctuation Restoration
#### CT_Transformer
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    )

rec_result = inference_pipeline(text_in='我们都是木头人不会讲话不会动')
print(rec_result)
```

### Timestamp Prediction
#### TP-Aligner
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
```

### Speaker Verification
#### X-vector
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
```

### Speaker diarization
#### SOND
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_diar_pipline = pipeline(
    mode="sond_demo",
    num_workers=0,
    task=Tasks.speaker_diarization,
    diar_model_config="sond.yaml",
    model='damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch',
    sv_model="damo/speech_xvector_sv-en-us-callhome-8k-spk6135-pytorch",
    sv_model_revision="master",
)

audio_list=[
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/record.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_A.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_B.wav",
    "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_data/spk_B1.wav"
]

results = inference_diar_pipline(audio_in=audio_list)
print(results)
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
#### Paraformer model

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

### FAQ
### Multi GPUs training and distributed training

If you want finetune with multi-GPUs, you could:
```shell
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node 2 finetune.py > log.txt 2>&1
```

