# Fun-ASR-Nano with Transformers

[简体中文](./transformers_native_zh.md) | English

Transcribe Chinese, English and Japanese with `AutoProcessor` and
`AutoModelForSpeechSeq2Seq`. **Transformers 5.17.0 is a released package with
native Fun-ASR-Nano support.** No source checkout, FunASR toolkit or
checkpoint-provided Python code is needed.

[Try the Space](https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano) ·
[Open the notebook](https://colab.research.google.com/github/QwenAudio/Fun-ASR/blob/main/examples/colab/fun_asr_nano_transformers.ipynb) ·
[Runnable examples](https://github.com/QwenAudio/Fun-ASR/tree/main/examples/transformers)

## Get your first transcript

Start in an isolated environment. This reproducible installation is for Linux
x86-64, Python 3.12 and **CPU**; do not upgrade an existing vLLM or FunASR service
environment in place. Torch and torchaudio must match because the native
feature extractor uses `torchaudio.compliance.kaldi.fbank`.

```bash
python3.12 -m venv .venv-funasr-native
. .venv-funasr-native/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu' 'torchaudio==2.10.0+cpu'
python -m pip install 'transformers==5.17.0' 'numpy==1.26.4' \
  'librosa==0.11.0' 'soundfile==0.13.1' \
  'huggingface-hub==1.30.0' 'tokenizers==0.23.2'
python -m pip check
```

Run this Python example. It downloads about 1.66 GB of model weights on first
use and transcribes a short, pinned official English sample. CPU float32 is
explicit; no GPU is needed for this example.

<!-- native-example: transcribe -->
```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

torch.set_num_threads(4)
model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
revision = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
audio = "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/resolve/272c57b82523ada6fd87095e955f8e29100979ab/example/en.mp3"

processor = AutoProcessor.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False
)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False,
    dtype=torch.float32,
).to("cpu").eval()
inputs = processor.apply_transcription_request(
    audio=audio, language="en",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": 16000},
        "text_kwargs": {"padding": True},
    },
)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
new_tokens = generated[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

The model returns the prompt and generated tokens together. Remove the full
input tensor width before decoding. For this sample, the observed raw output
was: “The tribal chieftain called for the boy, and presented him with fifty
pieces of gold.” This is a functional example, not a quality guarantee.

## Your files, keywords and batches

The [command-line example](https://github.com/QwenAudio/Fun-ASR/tree/main/examples/transformers)
adds explicit input validation, sample-rate conversion and completion checks.
Clone the **QwenAudio/Fun-ASR code repository** (not an HF weights repository),
install the [independent native requirements](https://github.com/QwenAudio/Fun-ASR/tree/main/examples/transformers),
and run these commands from its root directory:

```bash
python examples/transformers/transcribe.py recording.wav --language zh --keywords 开放时间
python examples/transformers/transcribe.py chinese.wav english.wav --language zh en
```

It averages multichannel audio to mono and resamples to 16 kHz with `soxr_hq`;
it never overwrites the original file. It accepts 1-4 files with at most
60 seconds of total audio. Longer recordings, empty input and non-finite samples
are rejected, not silently trimmed.

For your own application, pass a waveform, local path or supported URL to
`processor.apply_transcription_request`. A raw waveform must have its true
sample rate; do not label 48 kHz samples as 16 kHz. Use `language="zh"`,
`"en"` or `"ja"`. For a batch, use an audio list, matching language list and
`text_kwargs={"padding": True}`. Preserve input order when decoding.

`keywords` supplies vocabulary hints and `prompt` supplies context. They are
not enforced vocabulary and are not the toolkit's `hotword` parameter. In our
Chinese sample, adding `开放时间` did not correct the raw output `开饭时间`.
Use representative recordings and human references to assess quality.

The short example bounds generation at 128 new tokens. Missing EOS can indicate
truncation; increasing the limit does not prove coverage of a long recording.
The CLI reports `reached_eos` and exits unsuccessfully for missing EOS or empty
text. EOS itself is not proof that all words were recognized.

## From an example to a service

| Need | Checkpoint and interface |
| --- | --- |
| Native Transformers in Python | `FunAudioLLM/Fun-ASR-Nano-2512-hf`, `AutoProcessor` + `AutoModelForSpeechSeq2Seq` |
| FunASR pipelines and HTTP adapters | `FunAudioLLM/Fun-ASR-Nano-2512`, `funasr.AutoModel`; [deployment matrix](./deployment_matrix.md) |
| Native vLLM | `FunAudioLLM/Fun-ASR-Nano-2512-vllm`; [native vLLM guide](./vllm_official_native_validation.md) |
| C++ / edge | Converted GGUF files; [llama.cpp](../runtime/llama.cpp/README.md) |

These are different artifacts, not interchangeable model IDs. Native
Transformers support does not add an HTTP server, queue, streaming protocol,
word timestamps or speaker identities. This native export omits the CTC
timestamp branch. The base Nano checkpoint covers zh/en/ja and Chinese
dialects/accents; the **31-language MLT** checkpoint is separate.

For transcription with anonymous speakers and segment timestamps, see the
third-party OpenMOSS [MOSS guide](./moss_transcribe_diarize.md). For other tasks,
use the [Model Zoo](../model_zoo/readme.md). Validate GPU dtype, attention
backends, memory, concurrency and quality on the actual target hardware.

## Troubleshooting without loading weights

If loading fails, check the active interpreter, `transformers.__version__`,
the `-hf` model ID and model revision first. Version 5.16.1 lacks this model;
upgrade in the new environment, not an unrelated running service. The native
classes live in `transformers.models.fun_asr_nano`.

This optional synthetic-silence check downloads the processor, not model weights:

<!-- native-example: processor -->
```python
import numpy as np
from transformers import AutoProcessor

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
revision = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
processor = AutoProcessor.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False
)
inputs = processor.apply_transcription_request(
    audio=np.zeros(16000, dtype=np.float32), language="en",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": 16000},
    },
)
print({name: tuple(value.shape) for name, value in inputs.items()})
```

Reject empty recordings and batches before calling the processor. Synthetic
preprocessing success is **not an accuracy or capacity result**.

## Reproducibility and sources

Verified on **2026-09-09** using the actual released 5.17.0 wheel, Python 3.12.3,
matching torch/torchaudio 2.10.0+cpu, float32 and four Torch threads on an Intel
Xeon Platinum 8480+. Official English URL, Chinese waveform, Chinese keywords
and a padded Chinese/English batch all returned non-empty text and EOS before
128 tokens. These two short public samples do not establish CER/WER, GPU
compatibility or serving capacity.

The public native revision is
`d93b302ee7fd505e1b3576120fc142fc6f7820e1`; it resolves to
`FunAsrNanoForConditionalGeneration` with `trust_remote_code=False`.
`model.safetensors` is 1,659,773,320 bytes, SHA256
`1bbb6dcc5d8b75084a399d48c4d4b0f3aa1d3f09f2616ae40ed2c4fba03d89c9`.
Audio samples come from the original repository's
[fixed example directory](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/tree/272c57b82523ada6fd87095e955f8e29100979ab/example),
not an invented `-hf/example` path.

[PR #46180](https://github.com/huggingface/transformers/pull/46180) merged as
`fc501343edfccdc840eb8594a6cafa8185c2de53` before the stable package was
published. Earlier source-only instructions described that interval; new
installations should use the released package above.
See the [official 5.17.0 model documentation](https://huggingface.co/docs/transformers/v5.17.0/en/model_doc/fun_asr_nano)
and [native model card](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf).

Keep versions, input hashes and raw results with application evaluations.
Never attach private recordings or credentials to a public issue.
