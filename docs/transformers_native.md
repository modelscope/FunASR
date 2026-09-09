# Fun-ASR-Nano with native Transformers

[简体中文](./transformers_native_zh.md) | English

Use this path when your Python application already works with Hugging Face
processors and generation APIs. It runs Fun-ASR-Nano through native Transformers
classes, without the FunASR toolkit or checkpoint-provided remote Python code.
For a managed HTTP service, start with the [deployment matrix](./deployment_matrix.md);
this guide does not add an OpenAI endpoint, request queue or realtime server.

## What was merged, and what was released?

On **2026-09-09**, [Transformers PR #46180](https://github.com/huggingface/transformers/pull/46180)
was merged as `fc501343edfccdc840eb8594a6cafa8185c2de53`.
At verification time that day, the latest stable **5.16.1 did not contain this
model**: both its release tag and actual wheel were inspected. A merge is not
a stable release. The source build identifies itself as `5.17.0.dev0`; that is
not a promised release version or date. Pin the tested source below, or check
that a later released package really contains `fun_asr_nano` before migrating.

## Choose the checkpoint and the interface together

| Path | Checkpoint | Entry point |
| --- | --- | --- |
| FunASR toolkit | `FunAudioLLM/Fun-ASR-Nano-2512` | `funasr.AutoModel`; separate documented split-engine path |
| Native Transformers | `FunAudioLLM/Fun-ASR-Nano-2512-hf` | `AutoProcessor` + `AutoModelForSpeechSeq2Seq` |
| Native vLLM | `FunAudioLLM/Fun-ASR-Nano-2512-vllm` | [Official native vLLM guide](./vllm_official_native_validation.md) |
| Native C++ | Converted GGUF files for the chosen runtime | [llama.cpp guide](../runtime/llama.cpp/README.md) |

These are different artifacts and API contracts. Renaming a model ID or passing
`backend="vllm"` does not convert weights. Native Transformers support does not
make the `-hf` checkpoint a vLLM, GGUF or FunASR HTTP-server checkpoint.

The official [native HF checkpoint](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf/tree/d93b302ee7fd505e1b3576120fc142fc6f7820e1)
is public and ungated at revision `d93b302ee7fd505e1b3576120fc142fc6f7820e1`.
Its configuration maps to `FunAsrNanoForConditionalGeneration`. Its feature
extractor is embedded in `processor_config.json`; a separate
`preprocessor_config.json` is not required for this snapshot.

## Independent CPU environment

Use a new directory for the virtual environment; do not upgrade an existing
FunASR or vLLM service environment in place. The following is **Linux x86-64,
Python 3.12, CPU-only**, not a CUDA or macOS installation recipe.

```bash
python3.12 -m venv .venv-funasr-native
. .venv-funasr-native/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.10.0+cpu' 'torchaudio==2.10.0+cpu'
python -m pip install 'numpy==1.26.4' 'librosa==0.11.0' 'soundfile==0.13.1' \
  'huggingface-hub==1.30.0' 'tokenizers==0.23.2' \
  'https://github.com/huggingface/transformers/archive/fc501343edfccdc840eb8594a6cafa8185c2de53.zip'
python -m pip check
```

The verification environment installed a wheel built from the exact source
commit above. Matching **torch and torchaudio** are required for this native
feature extractor: it calls `torchaudio.compliance.kaldi.fbank`. The toolkit's
optional audio dependencies do not imply that torchaudio is optional here.
Older prepared environments failed dependency gates; do not disable those gates
or mix incompatible Hub/tokenizers versions to get an import to succeed.
This is a tested environment, not a complete lock of every transitive dependency;
record `python -m pip freeze`, platform details and model revision in your own run.

## Check preprocessing without loading weights

This downloads the public configuration, tokenizer and chat template, **not
model weights**. One second of synthetic silence checks the entry point; it is
not a speech-recognition quality test.

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
    audio=np.zeros(16000, dtype=np.float32),
    language="en",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": 16000},
    },
)
print({name: tuple(value.shape) for name, value in inputs.items()})
```

In the verified environment the text tensors have shape `(1, 42)`, audio features
`(1, 17, 560)`, and feature mask `(1, 17)`. There are 17 audio placeholder tokens
and 17 valid feature frames. These shapes depend on input length and template;
do not hard-code them for real recordings.

## Transcribe one recording

Prepare a non-empty **mono 16 kHz WAV** named `audio.wav` in your working directory.
The code deliberately rejects a different sample rate or channel layout rather
than silently assigning 16 kHz to arbitrary samples. Resample/downmix with an
explicit policy before this step and preserve the original recording for review.

The first model load downloads the weights unless the pinned snapshot is already
cached. It needs substantially more memory and time than preprocessing. This
example uses CPU float32 explicitly; GPU placement, mixed precision, attention
backends and service concurrency need separate validation.

<!-- native-example: transcribe -->
```python
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
revision = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
audio, sample_rate = sf.read("audio.wav", dtype="float32")
if sample_rate != 16000 or audio.ndim != 1 or audio.size == 0:
    raise ValueError("Use a non-empty mono 16 kHz WAV file")

processor = AutoProcessor.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False
)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, revision=revision, trust_remote_code=False, token=False,
    dtype=torch.float32,
).to("cpu").eval()
inputs = processor.apply_transcription_request(
    audio=audio, language="zh",
    processor_kwargs={
        "return_tensors": "pt",
        "audio_kwargs": {"sampling_rate": sample_rate},
    },
)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
new_tokens = generated[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
```

`generated` includes the input prompt. Slice off `inputs.input_ids.shape[1]`
before decoding so prompt framing is not mistaken for recognized speech.
`max_new_tokens=128` bounds this short-file example; hitting the limit can
truncate output. A longer limit is not proof of complete coverage.
Use `language="en"` or `language="ja"` for the corresponding target language;
language selection is a transcription instruction, not a promise of translation.

## Context, hotwords and batches

`apply_transcription_request` accepts `prompt` for context and `keywords` for
hotwords. These are native processor arguments, not the toolkit's `hotword`
parameter or a vLLM HTTP field. The source API accepts one language for the batch
or a language list matching its length; per-recording prompt and nested keyword
lists must also match the batch. Keep the decoded results in input order.

Do not infer quality gains from a template accepting a keyword. Evaluate names,
numbers and unusual terms against authorized, representative recordings.
Batch padding, memory use and generated-text correctness must be tested together
before turning a single-file example into a production batch worker.

## Evidence and limits

The documented single-recording recipe was also executed on the official Chinese
sample, followed by an English single recording, a Chinese keyword request and a
Chinese/English batch. All returned non-empty text and EOS before the 128-token
limit. These are functional checks on two short public samples, not a CER/WER or
capacity evaluation. CPU: Intel Xeon Platinum 8480+, float32, four Torch threads;
matching torch/torchaudio 2.10.0+cpu and librosa 0.11.0. Model weights were loaded
from a previously downloaded official cache whose SHA/size were rechecked.

Samples came from the original model's
[fixed example directory](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/tree/272c57b82523ada6fd87095e955f8e29100979ab/example),
not from an invented -hf example directory. The mono 48 kHz Chinese MP3 was
explicitly resampled with librosa's default soxr_hq and written as float32 16 kHz
WAV before running the unmodified recipe. Its raw text was
`开饭时间早上九点至下午五点。`; adding the candidate keyword `开放时间` did not force that
spelling into the output. A keyword is a hint, not an enforced vocabulary.

Synthetic processor tests exercised language aliases, context/keyword lists and
two-length batches. The empty-list case still raises an upstream `IndexError`
before its intended validation; reject empty batches in your application.
The guide's explicit non-empty waveform check avoids that entry path. Successful
cases do not erase this observed limitation. Initial resampling also exposed the
old librosa 0.10.1 dependency on removed `pkg_resources`; the final environment
uses 0.11.0 and was retested.

The 2026-09-09 isolated CPU verification passed package dependency checks,
native configuration/class resolution and synthetic preprocessing with the
pinned official revision. Installed configuration, modeling, processing and
feature-extraction files were byte-identical to the merged upstream source.
A preprocessing pass is **not an accuracy or capacity result**. It does not
establish word timestamps, diarization, realtime behavior, GPU compatibility,
or performance equivalence with vLLM.

For anonymous speakers and segment timestamps, see the third-party OpenMOSS
[MOSS guide](./moss_transcribe_diarize.md); for other choices see the
[Model Zoo](../model_zoo/readme.md). Do not infer timestamp or identity support
from a model returning text. Preserve input hashes, versions, raw responses and
an authorized human review when assessing an application. Never attach private
recordings or credentials to a public issue.

## Sources and next steps

- [Merged implementation and usage documentation](https://github.com/huggingface/transformers/blob/fc501343edfccdc840eb8594a6cafa8185c2de53/docs/source/en/model_doc/fun_asr_nano.md).
- [Native processor](https://github.com/huggingface/transformers/blob/fc501343edfccdc840eb8594a6cafa8185c2de53/src/transformers/models/fun_asr_nano/processing_fun_asr_nano.py) and [audio feature extractor](https://github.com/huggingface/transformers/blob/fc501343edfccdc840eb8594a6cafa8185c2de53/src/transformers/models/fun_asr_nano/feature_extraction_fun_asr_nano.py).
- [Why the checkpoint suffix matters](https://www.funasr.com/en/blog/fun-asr-nano-transformers.html), an application-oriented walkthrough.
- [Fun-ASR model project](https://github.com/QwenAudio/Fun-ASR) and [FunASR toolkit](https://github.com/modelscope/FunASR).
