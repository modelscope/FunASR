<p align="center">
  <h1 align="center">FunASR</h1>
  <p align="center"><b>语音 → 结构化文本，3 行 Python 搞定</b></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr?style=flat&color=2563eb" alt="PyPI"/></a>
  <a href="https://github.com/modelscope/FunASR/stargazers"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=flat&color=b45309" alt="Stars"/></a>
  <a href="https://github.com/modelscope/FunASR/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-059669?style=flat" alt="License"/></a>
  <a href="https://modelscope.github.io/FunASR/zh/"><img src="https://img.shields.io/badge/文档-在线-blue?style=flat" alt="Docs"/></a>
</p>

<p align="center">
  <a href="./README.md">English</a> ·
  <a href="https://modelscope.github.io/FunASR/zh/"><b>完整文档</b></a> ·
  <a href="https://modelscope.github.io/FunASR/zh/tutorial.html">使用教程</a> ·
  <a href="https://modelscope.github.io/FunASR/zh/training.html">训练指南</a> ·
  <a href="https://modelscope.github.io/FunASR/api.html">API</a> ·
  <a href="https://arxiv.org/abs/2305.11013">论文</a>
</p>

---

```python
from funasr import AutoModel

model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad",
                  punc_model="ct-punc", spk_model="cam++")

res = model.generate(input="meeting.wav")
```
```
[说话人 0]  欢迎大家来体验达摩院推出的语音识别模型。
[说话人 1]  非常感谢，今天我们主要讨论三个议题。
[说话人 0]  好的，请开始吧。
```

一次调用，长音频进，带说话人标签的逐句结果出。VAD 分段、语音识别、标点恢复、说话人分离——全部搞定。

---

## 安装

```bash
pip install funasr
```

> 模型默认从 ModelScope 下载（国内快速）。海外用户加 `hub="hf"` 切换 HuggingFace。

## 为什么选 FunASR

| | |
|---|---|
| **50+ 语言** | Fun-ASR-Nano 覆盖 31 种语言含方言；Qwen3-ASR 覆盖 52 种语言自动检测 |
| **比 Whisper 快 15 倍** | 非自回归架构；SenseVoice 处理 10 秒音频仅需 70ms |
| **说话人分离** | "谁说了什么"——Paraformer、Fun-ASR-Nano、SenseVoice 三大模型均支持 |
| **情感与音频事件** | SenseVoice 检测开心/悲伤/愤怒 + 背景音乐、笑声、掌声 |
| **完整链路** | DeepSpeed 微调 → ONNX 导出 → Docker 部署 |

## 模型

| 模型 | 语言 | 速度 | 适用场景 | 链接 |
|:-----|:---:|:---:|:---------|:---:|
| **[Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR)** | 31 | ●●●○ | 多语言、方言、歌词 | [ModelScope](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) · [HF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| **Paraformer** | 2 | ●●●● | 中文生产级识别 | [ModelScope](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) · [HF](https://huggingface.co/funasr/paraformer-zh) |
| **SenseVoice** | 5 | ●●●●● | 识别 + 情感 + 音频事件 | [ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall) · [HF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| **[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)** | 52 | ●●○○ | 最高精度，上下文理解 | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) · [HF](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| **Paraformer-Streaming** | 1 | 实时 | 实时转写 | [ModelScope](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) · [HF](https://huggingface.co/funasr/paraformer-zh-streaming) |
| **cam++** | — | — | 说话人分离 | [ModelScope](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) · [HF](https://huggingface.co/funasr/campplus) |
| **emotion2vec+** | — | — | 语音情感识别 | [ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_large) · [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

速度：●●●●● = 极快（10s 音频 70ms），●●○○ = 较慢（大模型）。完整列表 → [model_zoo/](./model_zoo)

## 最新动态

- **2026.05** — [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)：52 种语言，基于大模型，自动语种检测
- **2026.05** — [GLM-ASR-Nano](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)：17 种语言，方言和低音量优化
- **2026.05** — Fun-ASR-Nano 和 SenseVoice 新增说话人分离支持
- **2025.12** — [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR)：31 种语言，数千万小时数据训练
- **2024.07** — [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)：多任务语音理解（识别 + 情感 + 事件）

<details><summary>更早版本</summary>

- 2024.10 — 新增 Whisper-large-v3-turbo 支持
- 2024.09 — 新增语音唤醒模型
- 2024.03 — Qwen-Audio、Whisper-large-v3 支持
- 2024.01 — FunASR 1.0 发布
- 2023.11 — 离线文件转写服务 3.0（CPU）
- 2023.10 — Paraformer-VAD-SPK 组合流水线
- 2023.08 — 实时转写服务发布

</details>

## 生态

| | |
|:--|:--|
| [**Fun-ASR-Nano**](https://github.com/FunAudioLLM/Fun-ASR) | 多语言语音识别大模型——31 种语言、时间戳、热词 |
| [**SenseVoice**](https://github.com/FunAudioLLM/SenseVoice) | 多任务语音理解——识别、情感、音频事件 |
| [**FunClip**](https://github.com/modelscope/FunClip) | 基于 FunASR + 大模型的 AI 视频剪辑 |
| [**CosyVoice**](https://github.com/FunAudioLLM/CosyVoice) | 自然语音生成，支持音色克隆 |

## 社区

问题反馈 → [GitHub Issues](https://github.com/modelscope/FunASR/issues)  
交流讨论 → 钉钉群（扫码加入）

<img src="docs/images/dingding.png" width="150"/>

## 引用

```bibtex
@inproceedings{gao2023funasr,
  author={Zhifu Gao and Zerui Li and Jiaming Wang and Haoneng Luo and Xian Shi and Mengzhe Chen and Yabin Li and Lingyun Zuo and Zhihao Du and Zhangyu Xiao and Shiliang Zhang},
  title={FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  year={2023},
  booktitle={INTERSPEECH},
}
```

## 许可证

代码：[MIT](./LICENSE) · 模型权重：[FunASR 模型许可](./MODEL_LICENSE)
