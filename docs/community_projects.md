# FunASR Community Projects

This page collects maintained community integrations and experiments around FunASR models. These projects are useful for users exploring deployment paths beyond the official Python runtime, but they are not official FunASR implementations unless explicitly noted.

## Applications and workflows

| Project | What it provides | Notes |
|---|---|---|
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | Voice-dataset preparation and WebUI transcription with Fun-ASR-Nano, SenseVoice, and classic FunASR models. | The default CLI path uses Fun-ASR-Nano for Chinese, English, Japanese, Korean, and automatic language detection; Cantonese uses classic UniASR. Models download on first use. See the upstream [runtime fallback](https://github.com/RVC-Boss/GPT-SoVITS/pull/2801) and [backend documentation](https://github.com/RVC-Boss/GPT-SoVITS/pull/2803). |

Run dataset transcription from a GPT-SoVITS checkout:

```bash
python tools/asr/funasr_asr.py -i <input> -o <output> -l zh
```

## VAD runtimes

| Project | What it provides | Notes |
|---|---|---|
| [vad-burn](https://github.com/di-osc/vad-burn) | FSMN VAD inference in pure Rust with Python bindings. It supports offline VAD, streaming VAD, CPU-only inference, and automatic download of the default `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` model from ModelScope. | Community project from [#3106](https://github.com/modelscope/FunASR/issues/3106). Useful for Rust services, CLI tools, desktop apps, or Python pipelines that need FSMN VAD without depending on the original Python runtime. |

`vad-burn` reports the following benchmark on `assets/vad_example.wav` (16 kHz mono PCM, 70.47 seconds) on a MacBook Pro M1 with the Burn Flex CPU backend:

| Mode | Avg time | RTF | Speedup |
|---|---:|---:|---:|
| FSMN VAD offline | 73.631 ms | 0.001045 | 957.08x |
| FSMN VAD streaming, 600 ms chunks | 198.425 ms | 0.002816 | 355.15x |

Minimal Python usage from the project:

```python
from vad_burn import FsmnVadModel, VadOptions

vad = FsmnVadModel.from_modelscope()
segments = vad.detect(samples, 16000, VadOptions())

stream = vad.new_stream(VadOptions())
for chunk in chunks:
    segments = stream.push(chunk, 16000)
final_segments = stream.finish()
```

## Add your project

If you maintain a FunASR integration, open a [showcase issue](https://github.com/modelscope/FunASR/issues/new?template=showcase.md) with:

- repository link and maintenance status
- supported FunASR model or runtime path
- install and minimal usage instructions
- benchmark or validation details when available
- a note about whether the project is official, community-maintained, or experimental
