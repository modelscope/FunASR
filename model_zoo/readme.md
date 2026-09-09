# Model Zoo

[简体中文](./readme_zh.md) | English

Choose the **model**, **checkpoint format**, and **runtime** separately. A model
listed here is not automatically supported by every server or export backend.

## Start by task

| Task | Model family | Important boundary |
| --- | --- | --- |
| Context-aware file transcription | [Fun-ASR-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) | The base checkpoint and native vLLM-converted checkpoint are different artifacts. |
| Broader multilingual file transcription | [Fun-ASR-MLT-Nano](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | Separate checkpoint; do not attribute its language coverage to the base Nano model. |
| Transcription with emotion and audio-event tags | [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | Tags are model outputs, not speaker identity. Speaker-aware pipelines need the documented companion components. |
| Mandarin transcription with timestamps | Paraformer | Offline and streaming checkpoints have different inference contracts. |
| Offline text, timestamps and anonymous speakers together | [MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) | Third-party OpenMOSS model; no external VAD or speaker model is needed for its unified path. Not known-person identification. |

Use the [selection guide](../docs/model_selection.md) for workload choices,
the [SDK contract](../docs/python_api.md) for parameters and return values, and
the [deployment matrix](../docs/deployment_matrix.md) for serving options.

## Model Usage

### Native Transformers checkpoint

Use [FunAudioLLM/Fun-ASR-Nano-2512-hf](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf)
with `AutoProcessor` and `AutoModelForSpeechSeq2Seq`, not the original toolkit,
native vLLM or GGUF loader. [Installation and inference](../docs/transformers_native.md)
pin the merged source and official revision. As checked on 2026-09-09, stable
Transformers 5.16.1 does not contain this native model. Matching torchaudio is required.

### FunASR toolkit

Start with the [installation guide](../docs/installation/installation.md) and
[Python tutorial](../docs/tutorial/README.md). Use an explicit hub and record the
resolved checkpoint/revision, FunASR version, device and inference options.
Aliases are resolved by [the repository mapping](../funasr/download/name_maps_from_hub.py);
an alias is convenient, but is not an immutable model revision.

```python
from funasr import AutoModel

model = AutoModel(model="paraformer-zh", hub="ms", device="cpu")
result = model.generate(input="meeting.wav")
print(result[0]["text"])
```

Replace `meeting.wav` with an existing recording. Download time, warmup and
inference are separate measurements. Preserve the raw result when validating
timestamps, speaker labels or model-specific tags.

## Speech Recognition

### Paraformer

| SDK alias | Intended use | ModelScope (`hub="ms"`) | Hugging Face (`hub="hf"`) |
| --- | --- | --- | --- |
| `paraformer-zh` | Offline Mandarin transcription; ModelScope alias selects SeACo | [SeACo checkpoint](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) | [Checkpoint](https://huggingface.co/funasr/paraformer-zh) |
| `paraformer-zh-streaming` | Chunked streaming with per-session cache | [Checkpoint](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) | [Checkpoint](https://huggingface.co/funasr/paraformer-zh-streaming) |
| `paraformer-en` | Offline English transcription | Resolve through the [hub mapping](../funasr/download/name_maps_from_hub.py). | Resolve through the hub mapping. |

The legacy [Paraformer VAD/punctuation pipeline](https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
and `paraformer-zh-spk` listing describe pipelines, not the `paraformer-zh` alias target. For explicit
composition, use `vad_model`, `punc_model` and `spk_model` as documented in the
[SDK guide](../docs/python_api.md), rather than treating all components as one
interchangeable ASR checkpoint.

## Pipeline components

For speaker vectors and SenseVoice raw/display tags, use the
[speaker and emotion guide](../docs/speaker_emotion.md). For utterance-end KWS,
use the separate [keyword spotting guide](../docs/keyword_spotting.md).

| Component | Alias | Model cards | What it does not do |
| --- | --- | --- | --- |
| Voice activity detection | `fsmn-vad` | [ModelScope](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) / [HF](https://huggingface.co/funasr/fsmn-vad) | Does not transcribe speech or identify a speaker. |
| Punctuation restoration | `ct-punc` | [ModelScope](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary) / [HF](https://huggingface.co/funasr/ct-punc) | Does not create acoustic timestamps. |
| Speaker embeddings | `cam++` | [Hub mapping](../funasr/download/name_maps_from_hub.py) | Does not name known people without a separately designed enrollment/matching system. |
| Timestamp prediction | `fa-zh` | [Hub mapping](../funasr/download/name_maps_from_hub.py) | Must match the documented input/model path; not universal timestamp support for every recognizer. |

See the [full ModelScope inventory](./modelscope_models.md) and
[Hugging Face inventory](./huggingface_models.md) for additional checkpoints.
These inventories include historical variants; verify each model card before
using a checkpoint in a new service.

## Third-party Unified Transcription and Diarization

MOSS-Transcribe-Diarize is published by **OpenMOSS**, not by the FunASR team.
Its unified offline output includes text, timestamps and anonymous speaker
labels scoped to a recording. It is not a realtime streaming or known-person
identification model. Use the [MOSS guide](../docs/moss_transcribe_diarize.md)
for the adapter, native upstream servers, memory requirements and response
boundaries.

## Model License

The [FunASR software license](../LICENSE) does not grant one license for every
model weight. Consult the individual checkpoint's model card/license, publisher,
training-data notes and the applicable [Model License Agreement](../MODEL_LICENSE).
Keep upstream attribution when redistributing models or derived artifacts.

## Validation and next steps

- [Train or fine-tune a supported recipe](../docs/training.md).
- [Register a custom model](../docs/model_registration.md).
- [Choose native vLLM or split-engine](../docs/vllm_guide.md); preserve their distinct artifacts and API contracts.
- [Deploy GGUF models with llama.cpp](../runtime/llama.cpp/README.md); an ONNX export is not a GGUF conversion.
- [Measure quality and runtime separately](../docs/benchmark/rtf_reproducibility.md), using representative audio and exact versions.
