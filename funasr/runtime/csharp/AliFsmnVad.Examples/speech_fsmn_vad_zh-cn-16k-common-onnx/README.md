---
tasks:
- voice-activity-detection
domain:
- audio
model-type:
- VAD model
frameworks:
- pytorch
backbone:
- fsmn
metrics:
- f1_score
license: Apache License 2.0
language: 
- cn
tags:
- FSMN
- Alibaba
- Online
datasets:
  train:
  - 20,000 hour industrial Mandarin task
  test:
  - 20,000 hour industrial Mandarin task
widgets:
  - task: voice-activity-detection
    inputs:
      - type: audio
        name: input
        title: 音频
    examples:
      - name: 1
        title: 示例1
        inputs:
          - name: input
            data: git://example/vad_example.wav 
    inferencespec:
      cpu: 1 #CPU数量
      memory: 4096
---

# FSMN-Monophone VAD 模型介绍

[//]: # (FSMN-Monophone VAD 模型)

## Highlight
- 16k中文通用VAD模型：可用于检测长语音片段中有效语音的起止时间点。
  - 基于[Paraformer-large长音频模型](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)场景的使用
  - 基于[FunASR框架](https://github.com/alibaba-damo-academy/FunASR)，可进行ASR，VAD，[中文标点](https://www.modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)的自由组合
  - 基于音频数据的有效语音片段起止时间点检测

## <strong>[ModelScope-FunASR](https://github.com/alibaba-damo-academy/FunASR)</strong>
<strong>[FunASR](https://github.com/alibaba-damo-academy/FunASR)</strong>希望在语音识别方面建立学术研究和工业应用之间的桥梁。通过支持在ModelScope上发布的工业级语音识别模型的训练和微调，研究人员和开发人员可以更方便地进行语音识别模型的研究和生产，并促进语音识别生态系统的发展。

[**最新动态**](https://github.com/alibaba-damo-academy/FunASR#whats-new) 
| [**环境安装**](https://github.com/alibaba-damo-academy/FunASR#installation)
| [**介绍文档**](https://alibaba-damo-academy.github.io/FunASR/en/index.html)
| [**中文教程**](https://github.com/alibaba-damo-academy/FunASR/wiki#funasr%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C)
| [**服务部署**](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime)
| [**模型库**](https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/modelscope_models.md)
| [**联系我们**](https://github.com/alibaba-damo-academy/FunASR#contact)


## 项目介绍

FSMN-Monophone VAD是达摩院语音团队提出的高效语音端点检测模型，用于检测输入音频中有效语音的起止时间点信息，并将检测出来的有效音频片段输入识别引擎进行识别，减少无效语音带来的识别错误。

<p align="center">
<img src="fig/struct.png" alt="VAD模型结构"  width="500" />

FSMN-Monophone VAD模型结构如上图所示：模型结构层面，FSMN模型结构建模时可考虑上下文信息，训练和推理速度快，且时延可控；同时根据VAD模型size以及低时延的要求，对FSMN的网络结构、右看帧数进行了适配。在建模单元层面，speech信息比较丰富，仅用单类来表征学习能力有限，我们将单一speech类升级为Monophone。建模单元细分，可以避免参数平均，抽象学习能力增强，区分性更好。


## 如何使用与训练自己的模型


本项目提供的预训练模型是基于大数据训练的通用领域VAD模型，开发者可以基于此模型进一步利用ModelScope的微调功能或者本项目对应的Github代码仓库[FunASR](https://github.com/alibaba-damo-academy/FunASR)进一步进行模型的效果优化。

### 在Notebook中开发

对于有开发需求的使用者，特别推荐您使用Notebook进行离线处理。先登录ModelScope账号，点击模型页面右上角的“在Notebook中打开”按钮出现对话框，首次使用会提示您关联阿里云账号，按提示操作即可。关联账号后可进入选择启动实例界面，选择计算资源，建立实例，待实例创建完成后进入开发环境，输入api调用实例。

#### 基于ModelScope进行推理

- 推理支持音频格式如下：
  - wav文件路径，例如：data/test/audios/vad_example.wav
  - wav文件url，例如：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav
  - wav二进制数据，格式bytes，例如：用户直接从文件里读出bytes数据或者是麦克风录出bytes数据。
  - 已解析的audio音频，例如：audio, rate = soundfile.read("vad_example_zh.wav")，类型为numpy.ndarray或者torch.Tensor。
  - wav.scp文件，需符合如下要求：

```sh
cat wav.scp
vad_example1  data/test/audios/vad_example1.wav
vad_example2  data/test/audios/vad_example2.wav
...
```

- 若输入格式wav文件url，api调用方式可参考如下范例：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
)

segments_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav')
print(segments_result)
```

- 输入音频为pcm格式，调用api时需要传入音频采样率参数audio_fs，例如：

```python
segments_result = inference_pipeline(audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.pcm', audio_fs=16000)
```

- 若输入格式为文件wav.scp(注：文件名需要以.scp结尾)，可添加 output_dir 参数将识别结果写入文件中，参考示例如下：

```python
inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
    output_dir='./output_dir',
)

inference_pipeline(audio_in="wav.scp")
```
识别结果输出路径结构如下：

```sh
tree output_dir/
output_dir/
└── 1best_recog
    └── text

1 directory, 1 files
```
text：VAD检测语音起止时间点结果文件（单位：ms）

- 若输入音频为已解析的audio音频，api调用方式可参考如下范例：

```python
import soundfile

waveform, sample_rate = soundfile.read("vad_example_zh.wav")
segments_result = inference_pipeline(audio_in=waveform)
print(segments_result)
```

- VAD常用参数调整说明（参考：vad.yaml文件）：
  - max_end_silence_time：尾部连续检测到多长时间静音进行尾点判停，参数范围500ms～6000ms，默认值800ms(该值过低容易出现语音提前截断的情况)。
  - speech_noise_thres：speech的得分减去noise的得分大于此值则判断为speech，参数范围：（-1,1）
    - 取值越趋于-1，噪音被误判定为语音的概率越大，FA越高
    - 取值越趋于+1，语音被误判定为噪音的概率越大，Pmiss越高
    - 通常情况下，该值会根据当前模型在长语音测试集上的效果取balance
    
#### 基于ModelScope进行微调

待开发

### 在本地机器中开发

#### 基于ModelScope进行微调和推理

支持基于ModelScope上数据集及私有数据集进行定制微调和推理，使用方式同Notebook中开发。

#### 基于FunASR进行微调和推理

FunASR框架支持魔搭社区开源的工业级的语音识别模型的training & finetuning，使得研究人员和开发者可以更加便捷的进行语音识别模型的研究和生产，目前已在Github开源：https://github.com/alibaba-damo-academy/FunASR

#### FunASR框架安装

- 安装FunASR和ModelScope

```sh
pip install "modelscope[audio_asr]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
git clone https://github.com/alibaba/FunASR.git
cd FunASR
pip install --editable ./
```

#### 基于FunASR进行推理

接下来会以私有数据集为例，介绍如何在FunASR框架中使用VAD上进行推理。

```sh
cd egs_modelscope/vad/speech_fsmn_vad_zh-cn-16k-common/
python infer.py
```

## 使用方式以及适用范围

运行范围
- 现阶段只能在Linux-x86_64运行，不支持Mac和Windows。

使用方式
- 直接推理：可以直接对长语音数据进行计算，有效语音片段的起止时间点信息（单位：ms）。

## 相关论文以及引用信息

```BibTeX
@inproceedings{zhang2018deep,
  title={Deep-FSMN for large vocabulary continuous speech recognition},
  author={Zhang, Shiliang and Lei, Ming and Yan, Zhijie and Dai, Lirong},
  booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5869--5873},
  year={2018},
  organization={IEEE}
}
```
