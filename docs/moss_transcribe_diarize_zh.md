# 在 FunASR 生态中部署 MOSS-Transcribe-Diarize

[English](./moss_transcribe_diarize.md)

本文把第三方
[OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
模型接入 FunASR 部署生态。模型由 OpenMOSS 以 Apache-2.0 发布，不是 FunASR
自有模型。FunASR 为其公开的 Transformers、vLLM 与 SGLang Omni 接口提供适配器，
并保留 OpenMOSS 模型名称、许可证和上游 revision。

MOSS-Transcribe-Diarize 会联合生成转写、时间戳和 `[S01]` 等说话人标签，应用侧
不必再拼接外部 VAD、ASR 和 diarization 管线。这里描述的是部署形态，不表示模型
内部没有分块或分段。

## 固定上游版本

本文按以下不可变版本验证：

- 源码：`OpenMOSS/MOSS-Transcribe-Diarize@cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3`
- 模型：`OpenMOSS-Team/MOSS-Transcribe-Diarize@e8681d68e7042738ffca8ac8212bc8fcb1131ab8`
- 许可证：固定上游源码和模型元数据中的 Apache-2.0
- vLLM 发布版：`v0.27.1@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`
- vLLM CUDA 12.9 x86_64 wheel SHA-256：
  `bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b`

三者都要固定。模型依赖 `trust_remote_code`，生产服务不要执行浮动的模型 revision。

## FunASR AutoModel 契约

本地后端应使用隔离的 Python 3.10+ 环境，并安装 Transformers 5.6 或更新版本。
MOSS 在一次生成中完成长音频转写与说话人识别，因此不要传 `vad_model` 或
`spk_model`。外部 VAD 会把长音频切开，并破坏跨分块的全局说话人身份。

```python
from funasr import AutoModel

model = AutoModel(
    model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    model_revision="e8681d68e7042738ffca8ac8212bc8fcb1131ab8",
    backend="hf",
    device="cuda:0",
    dtype="bf16",
    attn_implementation="sdpa",
    disable_update=True,
)

result = model.generate("audio.wav", max_new_tokens=5120)[0]
print(result["text"])
for segment in result["sentence_info"]:
    print(segment["start"], segment["end"], segment["spk"], segment["text"])
```

适配器在 `raw_text` 中保留原始标签生成，并返回 FunASR 通用字段：

- `text`：移除 MOSS 控制标签后的可读转写；
- `timestamp`：segment 级 `[start_ms, end_ms]`；
- `sentence_info`：包含 `start`、`end`、`text`、`sentence`、`spk` 和 `timestamp`；
- `raw_text`：用于审计的原始 `[start][Sxx]text[end]` 生成。

当解析器无法确认标签结构时，会保留 `text` 与 `raw_text`，并返回空时间戳和分段，
不会静默编造说话人信息。

同一结果契约也可以包装已经启动的 vLLM 服务，并且不会下载本地权重：

```python
from funasr import AutoModel

model = AutoModel(
    model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    backend="vllm",
    vllm_base_url="http://127.0.0.1:8898/v1",
    vllm_model="moss-transcribe-diarize",
    vllm_response_format="diarized_json",
    disable_update=True,
)
result = model.generate("audio.wav", max_completion_tokens=8192)[0]
```

vLLM 适配器发送官方文档中的 OpenAI-compatible multipart 请求，并把官方返回的
speaker `segments` 直接映射到 `sentence_info`。认证可通过 `vllm_api_key` 提供；
请放在 secret store，不要写入源码。

固定 vLLM revision 的生产结构化路径是
`vllm_response_format="diarized_json"`。为保持已有客户端兼容，默认值仍是 `json`，
此时 `raw_text` 保留原始标签生成；结构化模式下，`raw_text` 是 vLLM 返回的清理后
`text`，权威说话人信息位于 `sentence_info`。

## 选择服务后端

| 路径 | 环境 | 响应契约 | 适用场景 |
|---|---|---|---|
| vLLM | CUDA 12（`cu129`）或 CUDA 13（`cu130`） | `response_format=diarized_json` 返回 OpenAI-compatible 说话人 segments；`response_format=json` 保留原始 `[start][Sxx]text[end]` 文本 | 已运行 vLLM，或需要其调度能力 |
| SGLang Omni | 当前上游指南为 CUDA 13 | `response_format=verbose_json` 返回解析后的 segments | API 需要直接返回结构化说话人分段 |
| Transformers | PyTorch 进程 | Python 对象和原始标签文本 | 评测、排障或自定义预处理 |
| moss-transcribe.cpp / LocalAI | C++17、ggml、GGUF；CPU 或 ggml GPU 后端 | LocalAI 的 OpenAI-compatible 转写接口 | 不希望在推理机安装 Python/PyTorch，或需要量化 GGUF 和边缘部署 |

两个 HTTP 后端都使用 `/v1/audio/transcriptions`，但官方文档中的响应格式不能混为
一谈。固定的 vLLM revision 支持 `diarized_json`；承诺 `segments` 前仍要验证实际
部署的 exact server revision。

## vLLM

创建隔离环境并安装已验证的 vLLM 发布版：

```bash
uv venv --python 3.12 .venv-moss
curl -fL \
  https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl \
  -o vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
echo "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b  vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  | sha256sum -c -
uv pip install --python .venv-moss/bin/python --torch-backend=auto \
  "vllm[audio] @ file://$PWD/vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
```

必须安装 `audio` extra。只安装 plain `vllm` 时服务可以启动，但因为没有音频解码器，
请求会返回 HTTP 400（`Invalid or unsupported audio file`）。

使用不可变模型 revision 启动：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-moss/bin/vllm serve \
  OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision e8681d68e7042738ffca8ac8212bc8fcb1131ab8 \
  --served-model-name moss-transcribe-diarize \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8898
```

提交音频并验证官方结构化说话人响应：

```bash
curl -fsS http://127.0.0.1:8898/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=moss-transcribe-diarize \
  -F response_format=diarized_json \
  -F max_completion_tokens=8192 \
  -F temperature=0 \
  | tee moss-transcription.json

python - <<'PY'
import json

with open("moss-transcription.json", encoding="utf-8") as stream:
    payload = json.load(stream)
text = payload.get("text", "")
segments = payload.get("segments", [])
assert text.strip(), payload
assert segments, payload
assert all(
    isinstance(item.get("speaker"), str)
    and item.get("text")
    and item.get("start") <= item.get("end")
    for item in segments
), payload
print(text, sorted({item["speaker"] for item in segments}))
PY
```

如需审计模型原始紧凑标签生成，请把请求改为 `response_format=json`，并验证返回
`text` 中包含 `[S01]`。

长音频需要按业务最长会议验证 `max_completion_tokens`。FunASR vLLM 适配器同时接受
OpenAI-compatible 名称 `max_completion_tokens` 和兼容名称 `max_new_tokens`；两者同时
出现时以前者为准。增大上限也会影响显存和尾延迟。

### 已复现的 vLLM 契约

结构化响应验证使用 vLLM 0.27.1、Torch `2.13.0+cu129` 和一张 H100 80GB。
15.1685 秒双说话人样例
（`43dccc068506439cb633b382b6b98185baa837363d08cc5f7152ca89b0fdc3c8`）
返回两个 `diarized_json` segments，标签为 `S01`、`S02`；同一请求通过 FunASR
适配器后，返回两个时间单调且说话人一致的 `sentence_info`。

同一环境还验证了 FunASR #3539 的两段真实中文长音频：

- 309.600 秒样例
  （`6561ee553c8f762aac4ebd65439d3414820761b547fa3a2edcea43b86a2abc02`）
  使用默认 5120 上限返回 158 个 segments，最后时间戳为 309.41 秒，7.459 秒完成；
- 379.664 秒样例
  （`779899a3ce937dd7352b4db1ea53e3f6aa2cfef7109de0249082223c936f9372`）
  使用默认 5120 上限时生成在 354.98 秒处截断，`diarized_json` 因标签不完整返回
  HTTP 400；设置 `max_completion_tokens=8192` 后返回 224 个 segments，最后时间戳
  覆盖到 378.55 秒，10.516 秒完成。

这两段样例中，MOSS 修正了报告者指出的多处专有词和近音词，也正确保持了“转述基金会
声明”的语义连续性；但模型输出仍包含较短的独立分段，不能据此宣称字幕切分问题已解决。
这些数字只证明 exact input 的完整性边界，不是精度、吞吐或生产容量基准。

较早的 raw-response 验证使用 vLLM `0.23.1rc1.dev949+g68b4a1d58`、Torch
`2.11.0+cu129` 和一张 H100 80GB，结果如下：

- 仓库内 6.000 秒样例
  （`ea03e1f473ad1618a03da3327a545369cb8f6f06cb0f4115535e5a866167d47e`）
  返回 HTTP 200 和 `[0.96][S01]... [5.94]`；
- A + 0.8 秒静音 + B + 0.8 秒静音 + A 的拼接样例
  （`dbb32bcfed2e8226bedf64248a9f4a44685b293a4696d18fb4cfa701b04db912`）
  返回 HTTP 200、`S01 -> S02 -> S01`，时间戳覆盖到 19.08 秒。

这证明固定版本对这些输入满足 API 和说话人返回契约，不是 diarization 精度、重叠语音、
吞吐或生产容量结论。

旧 nightly commit `68b4a1d582818e67adc903bf1b8fc5a5447da2fa` 早于 vLLM
[`#48543`](https://github.com/vllm-project/vllm/pull/48543)：它支持
`response_format=json`，但 `diarized_json` 会返回 HTTP 400。应升级到已验证发布版，
不要静默重试一次成本较高的转写请求。

FunASR 适配器还分别使用 `backend="hf"` 和 `backend="vllm"`，按模型 revision
`e8681d68e7042738ffca8ac8212bc8fcb1131ab8` 做了真实测试。15.1685 秒的双说话人
样例（`43dccc068506439cb633b382b6b98185baa837363d08cc5f7152ca89b0fdc3c8`）通过
统一 `AutoModel` 结果契约返回两个单调时间分段和 `S01`、`S02`。测试完成后已停止
临时 vLLM worker；这只是契约冒烟测试，不是精度基准。

## moss-transcribe.cpp 与 LocalAI

[`localai-org/moss-transcribe.cpp`](https://github.com/localai-org/moss-transcribe.cpp)
是 LocalAI 团队维护的第三方 C++17/ggml 重写，许可证为 MIT；原始 OpenMOSS 模型
权重仍保持 Apache-2.0。它不是 FunASR `AutoModel` 后端，也不能与上面的 Python
适配器混用。

在需要 GGUF 量化、CPU 或 ggml CUDA/Metal/Vulkan/HIP 后端时，可以选择这条路径。
LocalAI `master@a7cc5873ef5b7c909fc9ff7d349d51738ba9bb05` 已包含
`moss-transcribe-cpp` 后端及 Hugging Face importer；后端固定
`moss-transcribe.cpp@190a569c13b4b247450f2fb3b2a431244e84833e`，importer 可识别
`huggingface://mudler/moss-transcribe.cpp-gguf` 并默认选择 Q5_K。部署前应按 LocalAI
文档验证所选 backend package、GGUF SHA-256、实际硬件后端及
`/v1/audio/transcriptions` 响应，不要把上游 README 的性能数字当作本机基准。

## SGLang Omni

按固定上游 SGLang Omni 安装指南准备 CUDA 13 环境，然后下载不可变模型快照并从本地目录启动：

```bash
git clone https://github.com/sgl-project/sglang-omni.git
git -C sglang-omni checkout 3f819f9cdae3d4eeec22f73306c9067a1ec2542e
```

该源码 pin 已把 `max_new_tokens` 传入 transcription 生成请求。最初的 #914
merge 早于这个请求字段，因此虽然它的 H100 benchmark 仍可作为上游证据，却不能
支撑下方长音频命令。

```bash
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision e8681d68e7042738ffca8ac8212bc8fcb1131ab8 \
  --local-dir .models/moss-transcribe-diarize

sgl-omni serve \
  --model-path .models/moss-transcribe-diarize \
  --port 8898 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80
```

请求结构化分段：

```bash
curl -fsS http://127.0.0.1:8898/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F response_format=verbose_json
```

验证每个 segment 都有起止时间与非空文本。SGLang Omni 当前的
`verbose_json` 合同把说话人编号保留为 `segments[].text` 的 `[Sxx]` 前缀，
并没有单独的 `speaker` 字段。接入字幕、会议纪要或分析系统前，应解析并校验此前缀。

FunASR 适配器会完成该校验，并把 SGLang 官方 segments 映射为与 HF、vLLM
一致的 `sentence_info` 合同：

```python
from funasr import AutoModel

model = AutoModel(
    model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    backend="sglang",
    sglang_base_url="http://127.0.0.1:8898/v1",
    sglang_model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    max_new_tokens=65536,
    disable_update=True,
)
result = model.generate(input="audio.wav", max_new_tokens=65536)[0]
for segment in result["sentence_info"]:
    print(segment["start"], segment["end"], segment["spk"], segment["text"])
```

不要传入 `vad_model` 或 `spk_model`：MOSS 在一次生成中联合完成分段和匿名说话人
归属，外部分段会破坏长轮次中的说话人一致性。适配器把上游带标签原文保存在
`raw_text`，只从标准化 segment 中移除已经校验的 `[Sxx]` 前缀；如果 SGLang
没有返回此前缀，则明确失败，不会伪造说话人身份。

原生 runtime 已通过 SGLang Omni
[#914](https://github.com/sgl-project/sglang-omni/pull/914) 合并。其单张 H100
Seed-TTS EN benchmark 完成 1088/1088 条请求且无请求失败。WER 是在移除
timestamp 与 speaker markup 后针对单说话人英文片段计算的，因此不评估
diarization 或 timestamp 准确率，也不是生产容量承诺。

## 上线前验证

不要只用短单人音频，应使用真实多人长音频：

- 检查长轮次及说话人再次出现时的身份一致性；
- 覆盖重叠语音、串音、音乐、噪声和长静音；
- 验证时间戳单调递增并覆盖完整音频；
- 记录 GPU、CUDA、Torch、后端 commit、模型 revision、音频时长、生成上限、
  墙钟延迟和峰值显存；
- 在 API 网关完成认证、TLS、请求限制和数据留存策略，模型 worker 只绑定内网。

向 FunASR 报告问题时，请明确这是 OpenMOSS 第三方路径，并附上后端与上游 exact
revision。模型架构或权重问题应反馈给上游；部署中心的契约或文档问题可以在 FunASR 提交。
