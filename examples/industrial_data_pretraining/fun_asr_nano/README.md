# Fun-ASR

「[简体中文](README_zh.md)」|「English」

Fun-ASR is an end-to-end speech recognition large model launched by Tongyi Lab. It is trained on tens of millions of hours of real speech data, possessing powerful contextual understanding capabilities and industry adaptability. It supports low-latency real-time transcription and covers 31 languages. It excels in vertical domains such as education and finance, accurately recognizing professional terminology and industry expressions, effectively addressing challenges like "hallucination" generation and language confusion, achieving "clear hearing, understanding meaning, and accurate writing."

<div align="center">
<img src="images/funasr-v2.png">
</div>

<div align="center">
<h4>
<a href="https://funaudiollm.github.io/funasr"> Homepage </a>
｜<a href="#core-features"> Core Features </a>
｜<a href="#performance-evaluation"> Performance Evaluation </a>
｜<a href="#environment-setup"> Environment Setup </a>
｜<a href="#usage-tutorial"> Usage Tutorial </a>

</h4>

Model Repository: [modelscope](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512), [huggingface](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)

Online Experience:
[ModelScope Community Space](https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano), [huggingface space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano)

</div>

<a name="What's News"></a>

# What's New 🔥

- 2025/12: [Fun-ASR-Nano-2512](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) is an end-to-end speech recognition large model trained on tens of millions of hours real speech data. It supports low-latency real-time transcription and covers 31 languages.
- 2024/7: [FunASR](https://github.com/modelscope/FunASR) is a fundamental speech recognition toolkit that offers a variety of features, including speech recognition (ASR), Voice Activity Detection (VAD), Punctuation Restoration, Language Models, Speaker Verification, Speaker Diarization and multi-talker ASR.

# Core Features 🎯

**Fun-ASR** focuses on high-precision speech recognition, multi-language support, and industry customization capabilities

- **Far-field High-noise Recognition:** Deeply optimized for far-distance sound pickup and high-noise scenarios (such as conference rooms, in-vehicle environments, industrial sites, etc.), improving recognition accuracy to **93%**.
- **Chinese Dialects and Regional Accents:**
  - Supports **7 major dialects**: Wu, Cantonese, Min, Hakka, Gan, Xiang, Jin
  - Covers **26 regional accents**: including Henan, Shaanxi, Hubei, Sichuan, Chongqing, Yunnan, Guizhou, Guangdong, Guangxi and more than 20 other regions
- **Multi-language Free Speech:** Supports recognition of **31 languages**, with focused optimization on East and Southeast Asian languages, supporting free language switching and mixed recognition.
- **Music Background Lyric Recognition:** Enhanced speech recognition performance under music background interference, supporting accurate recognition of lyric content in songs.

# Environment Setup 🐍

```shell
pip install -r requirements.txt
```

<a name="usage-tutorial"></a>

# TODO

- [ ] Support returning timestamps
- [ ] Support speaker diarization
- [ ] Support model training

# Usage 🛠️

## Inference

### Using funasr for inference

```python
from funasr import AutoModel


def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        remote_code="./model.py",
        device="cuda:0",
    )

    wav_path = f"{model.model_path}/example/zh.mp3"
    res = model.generate(input=[wav_path], cache={}, batch_size=1)
    text = res[0]["text"]
    print(text)

    model = AutoModel(
        model=model_dir,
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        remote_code="./model.py",
        device="cuda:0",
    )
    res = model.generate(input=[wav_path], cache={}, batch_size=1)
    text = res[0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

### Direct Inference

```python
from model import FunASRNano


def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()

    wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    res = m.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

<details><summary> Parameter Description (click to expand) </summary>

- `model_dir`: Model name or local disk model path.
- `trust_remote_code`: Whether to trust remote code for loading custom model implementations.
- `remote_code`: Specify the location of specific model code (e.g., `model.py` in the current directory), supporting both absolute and relative paths.
- `device`: Specify the device to use, such as "cuda:0" or "cpu".

</details>

# Performance Evaluation 📝

We compared the multi-language speech recognition performance of Fun-ASR with other models on open-source benchmark datasets (including AISHELL-1, AISHELL-2, Wenetspeech, Librispeech, and Common Voice).

<div align="center">
<img src="images/compare_en.png" width="800" />
</div>
