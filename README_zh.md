<div align="center">
<img src="docs/images/funasr_logo.jpg" width="400"/>

**基础端到端语音识别工具包**

[![PyPI](https://img.shields.io/pypi/v/funasr)](https://pypi.org/project/funasr/)
[![GitHub Stars](https://img.shields.io/github/stars/modelscope/FunASR?style=social)](https://github.com/modelscope/FunASR/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/modelscope/FunASR/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/funasr?color=blue)](https://pypi.org/project/funasr/)

[English](./README.md) | [简体中文](./README_zh.md) | [完整文档](https://modelscope.github.io/FunASR/zh/) | [论文](https://arxiv.org/abs/2305.11013)

</div>

---

FunASR 是[阿里巴巴通义实验室](https://tongyi.aliyun.com)开发的工业级语音识别工具包，提供统一的 Python 接口，一次调用即可完成从原始音频到带说话人标注、标点恢复的结构化文本全流程。

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",       # 语音识别
    vad_model="fsmn-vad",        # 语音端点检测
    punc_model="ct-punc",        # 标点恢复
    spk_model="cam++",           # 说话人分离
)
res = model.generate(input="meeting.wav", batch_size_s=300)

for s in res[0]["sentence_info"]:
    print(f"[说话人 {s['spk']}] {s['text']}")
```

**输出：**

> \[说话人 0\] 欢迎大家来体验达摩院推出的语音识别模型。  
> \[说话人 1\] 非常感谢，今天我们主要讨论三个议题。  
> \[说话人 0\] 好的，请开始吧。

---

## 核心特性

- **统一流水线** — VAD、ASR、标点、说话人分离、情感检测，一个 `AutoModel()` 调用组合完成
- **50+ 语言** — Fun-ASR-Nano（31 种语言含方言）、Qwen3-ASR（52 种语言自动检测）
- **高性能** — 非自回归推理；SenseVoice 处理 10 秒音频仅需 70ms（比 Whisper 快 15 倍）
- **说话人分离** — 逐句话者标注，兼容 Paraformer、Fun-ASR-Nano、SenseVoice
- **情感与音频事件** — SenseVoice 识别情感（开心/悲伤/愤怒/中性）并检测背景音乐、笑声、掌声
- **生产就绪** — DeepSpeed 微调、ONNX 导出、Docker 运行时或 Python SDK 部署

## 安装

```bash
pip install -U funasr

# 或从源码安装（获取最新模型支持）
git clone https://github.com/modelscope/FunASR.git && pip install -e ./FunASR
```

> 模型默认从 [ModelScope](https://modelscope.cn) 下载（国内快速）。海外用户添加 `hub="hf"` 切换 [HuggingFace](https://huggingface.co/FunASR)。

## 模型

| 模型 | 类型 | 语言 | 参数量 | 推理速度 | 链接 |
|:-----|:-----|:---:|:------:|:--------:|:----:|
| [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) | ASR | 31 | 800M | 中 | [MS](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) · [HF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| [Paraformer-zh](https://huggingface.co/funasr/paraformer-zh) | ASR | 2 | 220M | 快 | [MS](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch) · [HF](https://huggingface.co/funasr/paraformer-zh) |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | ASR+情感+事件 | 5 | 234M | 极快 (70ms/10s) | [MS](https://modelscope.cn/models/iic/SenseVoiceSmall) · [HF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | ASR (大模型) | 52 | 1.7B | 慢 | [MS](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B) · [HF](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| [GLM-ASR-Nano](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | ASR (大模型) | 17 | 1.5B | 慢 | [MS](https://modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512) · [HF](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) |
| Paraformer-Streaming | ASR | 1 | 220M | 实时 | [MS](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) · [HF](https://huggingface.co/funasr/paraformer-zh-streaming) |
| fsmn-vad | VAD | 2 | 0.4M | — | [MS](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch) · [HF](https://huggingface.co/funasr/fsmn-vad) |
| ct-punc | 标点恢复 | 2 | 290M | — | [MS](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large) · [HF](https://huggingface.co/funasr/ct-punc) |
| cam++ | 说话人 | — | 7.2M | — | [MS](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) · [HF](https://huggingface.co/funasr/campplus) |
| emotion2vec+ | 情感 | — | 300M | — | [MS](https://modelscope.cn/models/iic/emotion2vec_plus_large) · [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |

完整模型列表 → [model_zoo/](./model_zoo) &nbsp;|&nbsp; 详细用法 → [使用教程](https://modelscope.github.io/FunASR/zh/tutorial.html)

## 最新动态

| 时间 | 更新 |
|:-----|:-----|
| 2026.05 | [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — 52 种语言，基于大模型，自动语种检测 |
| 2026.05 | [GLM-ASR-Nano](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) — 17 种语言，方言和低音量优化 |
| 2026.05 | Fun-ASR-Nano 和 SenseVoice 新增说话人分离支持 |
| 2025.12 | [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) — 31 种语言，数千万小时数据训练 |
| 2024.07 | [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) — 多任务语音理解（识别 + 情感 + 事件） |

<details><summary>查看完整更新历史</summary>

参见 [CHANGELOG.md](./CHANGELOG.md)。

</details>

## 了解更多

| 资源 | 说明 |
|:-----|:-----|
| [使用教程](https://modelscope.github.io/FunASR/zh/tutorial.html) | 安装、选择模型、运行识别/VAD/说话人分离 |
| [训练指南](https://modelscope.github.io/FunASR/zh/training.html) | 在自定义数据上微调 Paraformer、SenseVoice、Fun-ASR-Nano |
| [开发指南](https://modelscope.github.io/FunASR/zh/model-registration.html) | 添加新模型、理解注册表机制、测试与贡献 |
| [API 文档](https://modelscope.github.io/FunASR/api.html) | 自动生成的类和方法文档，附源码链接 |
| [部署文档](./runtime/readme_cn.md) | 离线文件转写服务、实时流式服务（CPU/GPU） |

## 生态

| 项目 | 说明 |
|:-----|:-----|
| [Fun-ASR-Nano](https://github.com/FunAudioLLM/Fun-ASR) | 多语言语音识别大模型——31 种语言、时间戳、热词、说话人分离 |
| [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) | 多任务语音理解——识别、语种识别、情感、音频事件 |
| [FunClip](https://github.com/modelscope/FunClip) | 基于 FunASR 和大模型辅助编辑的 AI 视频剪辑 |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | 自然语音生成，支持多语言、音色和情感控制 |

## 社区

- 问题反馈 → [GitHub Issues](https://github.com/modelscope/FunASR/issues)
- 加入钉钉交流群：

<img src="docs/images/dingding.png" width="150"/>

## 引用

```bibtex
@inproceedings{gao2023funasr,
  author    = {Zhifu Gao and Zerui Li and Jiaming Wang and Haoneng Luo and Xian Shi and Mengzhe Chen and Yabin Li and Lingyun Zuo and Zhihao Du and Zhangyu Xiao and Shiliang Zhang},
  title     = {FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  booktitle = {INTERSPEECH},
  year      = {2023},
}
```

## 许可证

代码：[MIT 许可证](./LICENSE) &nbsp;·&nbsp; 模型权重：[FunASR 模型许可](./MODEL_LICENSE)（允许商业使用，需注明出处）
