([简体中文](./README_zh.md) | English)

# Python SDK tutorial

> **Only need native Fun-ASR-Nano transcription?** Start with [Transformers 5.17.0](../transformers_native.md). It loads the separate `-hf` checkpoint without the FunASR toolkit. This page covers the `funasr.AutoModel` toolkit path; do not mix dependencies, parameters or output contracts.

Start with [installation and environment verification](../installation/installation.md). This path covers one transcript, result inspection, VAD, batching, and model-specific options. For model selection, languages, dependencies, and model cards, use the [Model Zoo](../../model_zoo/readme.md), not a universal capability list. No local setup yet? See the [Colab quickstart](../../examples/colab/README.md).

FunASR software uses the [MIT license](../../LICENSE). Each model weight has its own license: record the full model ID and revision, and follow its model card. The [FunASR Model License Agreement](../../MODEL_LICENSE) applies only when the model card explicitly links to it. Third-party integrations remain third-party models; for example, MOSS-Transcribe-Diarize is from OpenMOSS, not a FunASR-trained checkpoint.

<a id="Inference" name="Inference"></a>
## 1. Produce a first transcript

Run the following Python block after installing the SDK and a matching PyTorch/torchaudio environment. The first run needs network access for the model and this public example WAV, plus sufficient disk and memory. It is based on the repository's [Paraformer example](../../examples/industrial_data_pretraining/paraformer/demo.py); it is not a recorded inference result from this documentation review.

```python
from funasr import AutoModel

audio = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
model = AutoModel(
    model="paraformer-zh", hub="ms", device="cpu", ncpu=4,
    disable_update=True, trust_remote_code=False,
)
results = model.generate(input=audio)
for item in results:
    print(item.get("key"), item.get("text", ""))
print("Model directory:", model.model_path)
```

To transcribe your recording, replace `audio` with an existing local WAV path. Start with a short mono recording matching the selected model's sample rate (16 kHz for this example). File decoding depends on installed audio backends; a readable WAV is a simpler first check than an arbitrary media container. NumPy waveform input has no sample-rate header: supply the actual rate with `fs=sample_rate`, as used by the [audio loader](../../funasr/utils/load_utils.py). Do not merely relabel a waveform's rate.

The remaining Python blocks reuse `AutoModel`, `audio`, and `model` from this first block in the same session. For disconnected use, prepare complete local model directories and local input files using the [offline checklist](../installation/installation.md). `disable_update=True` only skips the SDK's startup version check.

## 2. Understand parameters and results

`AutoModel(...)` constructs the model and optional pipeline components. `model.generate(input=..., **options)` performs inference. Source references: [AutoModel](../../funasr/auto/auto_model.py), [hub aliases](../../funasr/download/name_maps_from_hub.py), and the chosen model's `inference()` implementation.

| Parameter | Scope and meaning |
| --- | --- |
| `model`, `hub` | Model ID/alias or local directory; hub defaults to ModelScope (`ms`), with `hf` for Hugging Face. Aliases can resolve to different hub repositories. |
| `device`, `ncpu` | Explicitly use `cpu` to begin; choose an accelerator only after validating its PyTorch build and model support. The loader has CPU fallback paths; `ncpu` controls PyTorch CPU threads. |
| `vad_model`, `punc_model`, `spk_model` | Optional, separately loaded models. Configure them through `vad_kwargs`, `punc_kwargs`, and `spk_kwargs`. They are not automatically supported by every ASR backend. |
| `batch_size` | Number of inputs per decoding batch on the ordinary non-VAD path; backend restrictions still apply. |
| `batch_size_s` | Duration budget in seconds for VAD-segment batching, based on padded segment length, not a count of files. The current CPU VAD path decodes segments individually. |
| `batch_size_threshold_s` | Segment-duration threshold used by the VAD batching heuristic; not an input duration limit or a universal memory cap. |
| `output_dir` | Optional backend output directory. The returned Python result is still available; files and formats depend on the model. |

`generate()` returns a list of dictionaries, normally one per input recording. Inspect keys before depending on optional fields:

```python
for item in results:
    print(sorted(item.keys()))
    print(item.get("text", ""))
    print(item.get("timestamp", []))
```

For the Paraformer path, `timestamp` contains character/token/word intervals as `[start_ms, end_ms]` pairs when supplied by the checkpoint. Do not assume one pair per displayed character after punctuation or text normalization. Other backends may return `timestamps` with a different schema; follow that backend's guide. Empty speech may produce empty text/timestamps, not a meaningful transcript.

## 3. Add VAD, punctuation, and sentence timestamps

VAD detects speech spans; it does not transcribe them. This standalone call uses the same audio:

```python
vad = AutoModel(model="fsmn-vad", device="cpu", disable_update=True)
vad_results = vad.generate(input=audio)
for item in vad_results:
    print(item["key"], item["value"])
```

`value` is a list of `[start_ms, end_ms]` speech intervals relative to the recording start. An empty list means no detected spans. See the [FSMN VAD example](../../examples/industrial_data_pretraining/fsmn_vad_streaming/demo.py).

For segmented ASR with punctuation:

```python
pipeline = AutoModel(
    model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu", disable_update=True, trust_remote_code=False,
)
segmented_results = pipeline.generate(
    input=audio, batch_size_s=60, batch_size_threshold_s=30,
    sentence_timestamp=True,
)
for item in segmented_results:
    print(item.get("text", ""))
    for sentence in item.get("sentence_info", []):
        print(sentence.get("start"), sentence.get("end"), sentence.get("text", ""))
```

`max_single_segment_time` is in **milliseconds**; the two batching controls are in **seconds**. Segmentation can help with longer files but does not guarantee unlimited duration or bounded memory: the audio-loading stage and each model still consume resources. On memory pressure, use shorter recordings/segments and smaller supported batches, then measure again. Sentence boundaries depend on available timestamp/punctuation alignment and may fall back to VAD segments; inspect the returned data.

For diarization, construct the same pipeline with `spk_model="cam++"` if supported by the chosen components. Iterate each input's `item.get("sentence_info", [])`, then read `sentence.get("spk")`. Do not read `spk` directly from the outer result list. Cluster labels are not verified real-world identities. [Speaker integration example](../../examples/industrial_data_pretraining/sense_voice/demo_spk.py).

## 4. Process multiple recordings

The following deliberately repeats the sample to demonstrate a list input without requiring extra files. Replace the entries with your local WAV paths:

```python
batch_results = model.generate(input=[audio, audio], batch_size=1)
for index, item in enumerate(batch_results):
    print(index, item.get("key"), item.get("text", ""))
```

Increase `batch_size` only if the selected model supports it and memory permits. A file list such as `wav.scp` is also supported: use one `utterance_id path` per line, resolve paths from the process working directory, and give each recording a unique ID. Set `output_dir` if you need model-written artifacts; it is not required simply to receive the returned list. See [data-list examples](../../data/list/train_wav.scp). Do not confuse independent file batching with streaming chunks of one utterance.

## 5. Hotwords and language boundaries

The ModelScope `paraformer-zh` alias in this checkout resolves to a SeACo Paraformer model, whose implementation accepts singular `hotword` as a space-separated string:

```python
biased_results = model.generate(input=audio, hotword="魔搭 达摩院")
print([item.get("text", "") for item in biased_results])
```

This is model-level context biasing, not guaranteed word insertion or deterministic replacement. Verify the resolved model and compare against a no-hotword baseline. [SeACo implementation](../../funasr/models/seaco_paraformer/model.py) and [contextual Paraformer example](../../examples/industrial_data_pretraining/contextual_paraformer/demo.py).

Postprocessing is a different operation in this source checkout:

```python
corrected_results = model.generate(
    input=audio,
    postprocess_hotwords={"科大迅飞": "科大讯飞"},
    return_postprocess_hotword_matches=True,
)
for item in corrected_results:
    print(item.get("text", ""), item.get("postprocess_hotword_matches", []))
```

Explicit mappings replace matching output text. `postprocess_hotword_file` also accepts one target per line or `wrong=>right` mappings. Fuzzy matching additionally needs `pypinyin` and `rapidfuzz`; explicit mappings do not. The [postprocessor](../../funasr/utils/postprocess_hotwords.py) preserves existing timestamps rather than realigning corrected text, so review replacements before making subtitles or alignment claims.

`hotword`, `hotwords`, and `language` are **not interchangeable universal SDK options**. For example, [Fun-ASR-Nano](../../examples/industrial_data_pretraining/fun_asr_nano/README.md) reads plural `hotwords` and a model-specific language hint. Changing a hint does not turn a monolingual checkpoint into a multilingual one. Use the [Model Zoo](../../model_zoo/readme.md) and the exact checkpoint guide for supported languages, accepted hint values, streaming, alignment, and dependency versions. Avoid copying language counts or installation pins between model families.

## 6. Continue with a specific workflow

- **Streaming:** [Paraformer streaming example](../../examples/industrial_data_pretraining/paraformer_streaming/demo.py). Keep a separate `cache={}` per stream and set `is_final=True` for the final chunk. For `[0, 10, 5]` at 16 kHz, 600 ms is **9600 samples**, not 960; chunk duration is not an end-to-end latency guarantee. Streaming VAD may return `[start, -1]`, `[-1, end]`, complete spans, or no spans, in milliseconds.
- **Punctuation and alignment:** [punctuation example](../../examples/industrial_data_pretraining/ct_transformer/demo.py) and [timestamp prediction example](../../examples/industrial_data_pretraining/monotonic_aligner/demo.py). Alignment needs the corresponding text input; it is not equivalent to ASR.
- **Other model families:** [SenseVoice](../../examples/industrial_data_pretraining/sense_voice/README.md), [Fun-ASR-Nano](../../examples/industrial_data_pretraining/fun_asr_nano/README.md), and [third-party OpenMOSS integration](../moss_transcribe_diarize.md). Discover further entries through the Model Zoo.
- **CLI and services:** [CLI reference](../cli.md), [runtime overview](../../runtime/readme.md), and [Docker](../installation/docker.md). Python options do not imply an identical server request schema.

<a id="Training" name="Training"></a>
## Model Training and Testing

Use the [Paraformer recipe](../../examples/industrial_data_pretraining/paraformer/README.md), [finetune.sh](../../examples/industrial_data_pretraining/paraformer/finetune.sh), and [training-data examples](../../data/list/train.jsonl). Inspect dataset paths, label alignment, model license, GPU allocation, and output directories before launching. Training is not an installation smoke test. For trained weights, inspect [infer_from_local.sh](../../examples/industrial_data_pretraining/paraformer/infer_from_local.sh): configuration, tokenizer/frontend assets, and checkpoint paths must agree. Keep [validation data](../../data/list/val.jsonl) separate from training data.

<a id="Export" name="Export"></a>
## Model Export and Testing

Follow the model's [Paraformer export example](../../examples/industrial_data_pretraining/paraformer/export.py) and [ONNX Runtime guide](../../runtime/python/onnxruntime/README.md). Export support and extra dependencies are model/backend-specific. Successful export is not evidence of output equivalence: test the exported artifact on representative inputs and compare with the original model before deployment.

<a id="new-model-registration-tutorial" name="new-model-registration-tutorial"></a>
## Register a Custom Model

Use the [registry tutorial](./Tables.md) and a real implementation such as [SenseVoice](../../funasr/models/sense_voice/model.py). Registration alone does not establish a working `generate()` contract; the model's inference result must match the downstream components it will use.

For failures, return to [troubleshooting](../troubleshooting.md) with interpreter/package versions, resolved model ID/revision, input format, and a minimal reproduction. Do not include private audio or credentials.
