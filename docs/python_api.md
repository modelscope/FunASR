# Python SDK: AutoModel

> **Only need native Fun-ASR-Nano transcription?** Start with [Transformers 5.17.0](./transformers_native.md). It loads the separate `-hf` checkpoint without the FunASR toolkit. This page covers the `funasr.AutoModel` toolkit path; do not mix dependencies, parameters or output contracts.

[简体中文](python_api_zh.md) | [Installation](installation/installation.md) | [Model selection](model_selection.md)

`from funasr import AutoModel` runs models inside your Python process. It is not an HTTP client and does not implement the entire OpenAI API. This guide describes the implementation in this checkout, not every historical FunASR release or upstream checkpoint.

## Construction and Inference Are Different

`AutoModel(**kwargs)` resolves a model configuration, looks up its registered implementation, loads weights, and optionally constructs VAD, punctuation, and speaker models. `model.generate(input, input_len=None, progress_callback=None, **cfg)` runs those already-loaded components. Passing a new `model`, `vad_model`, or `device` to `generate()` is not a supported way to rebuild or move the pipeline; construct a separate instance.

The wrapper accepts `**kwargs`, not a universally validated parameter schema. A keyword being accepted does not mean the selected model uses it. Configuration files can override the fallback defaults below. See [AutoModel source](../funasr/auto/auto_model.py), the [registry](../funasr/register.py), [hub resolution](../funasr/download/download_model_from_hub.py), and [alias maps](../funasr/download/name_maps_from_hub.py).

### Construction Parameters

| Parameter | Source fallback | Meaning |
|---|---|---|
| `model` | Required | Hub alias, full model ID, or complete local model directory. Aliases are hub-specific, not version pins. |
| `hub` | `"ms"` | `"ms"`/`"modelscope"` or `"hf"`/`"huggingface"`. The special `"openai"` loader branch concerns Whisper loading, not HTTP API compatibility. |
| `model_revision` | `"master"` in hub loaders | ModelScope forwards the revision. The generic Hugging Face helper currently ignores it; see reproducibility below. |
| `device` | `"cuda"` | E.g. `"cuda:0"` or `"cpu"`. Unavailable checked accelerator backends fall back to CPU; explicit `ngpu=0` does too. Fallback sets `batch_size=1`. |
| `ngpu` | `1` | Zero selects CPU. This is not a multi-GPU serving or sharding configuration. |
| `ncpu` | `4` | Positive CPU thread count; invalid values use the fallback and values below 1 clamp to 1. Sets process-wide PyTorch threads. |
| `vad_model`, `punc_model`, `spk_model` | `None` | Optional components, built once. Use `vad_model`, not the rejected typo `vda_model`. |
| `vad_kwargs`, `punc_kwargs`, `spk_kwargs` | `{}` | Component configuration dictionaries. Device is inherited; hub and CPU threads are inherited unless provided in the component dictionary. |
| `vad_model_revision`, `punc_model_revision`, `spk_model_revision` | `"master"` each | Separate component revisions, not inherited from the ASR revision. These top-level values replace `model_revision` in the component dictionaries. |
| `spk_mode` | `"punc_segment"` | Use `"punc_segment"` or `"vad_segment"`. The validation also accepts legacy `"default"`, but the sentence-building branches do not implement it; do not select it. |
| `disable_update` | `False` | Disables the FunASR package version check only, not model downloading. |
| `disable_pbar` | `False` | Suppresses wrapper progress bars. |
| `trust_remote_code` | `False` in hub loaders | Opt-in to model-supplied requirements and code paths. Review the exact files and dependencies first. |

### Per-Call Parameters

| Parameter | Source fallback | Scope and limits |
|---|---|---|
| `input` | Required | Audio path, URL, waveform, supported audio bytes, or a list of inputs. Text models such as punctuation accept text. |
| `input_len` | `None` | The direct inference wrapper forwards it as `data_lengths` only for a single `data_type="fbank"` input; it is not a universal waveform-length control. |
| `progress_callback` | `None` | `callback(current, total)` after direct inference batches. VAD processing can forward it to component calls; it is not one global monotonic pipeline progress counter. |
| `batch_size` | `1` | Number of inputs per direct inference call. Actual model batch support varies. |
| `batch_size_s` | `300` | VAD-path batching budget in seconds, based on longest padded segment times segment count, not a hard file-duration limit. CPU VAD processing uses single-segment batches. |
| `batch_size_threshold_s` | `60` | VAD batching threshold for segment duration, not a VAD splitting setting. |
| `merge_vad` | `False` | Enables merging detected VAD regions for this call. |
| `merge_length_s` | `15` | VAD merge target read before call overrides are merged into ASR config. Set this at construction in this checkout. |
| `cache` | No wrapper-managed session | Pass a caller-owned dictionary on every streaming call. It is not a hub/model download cache. |
| `sentence_timestamp` | `False` | Requests sentence records in the VAD pipeline; depends on available timestamps and punctuation, with limited VAD fallback. |
| `return_spk_res` | `True` | Includes clustered speaker results when VAD and a speaker model are configured. `False` does not avoid construction or segment embedding computation. |
| `preset_spk_num` | `None` | Optional speaker-count hint for the clustering backend. |
| `return_spk_center` | `False` | Adds `spk_embedding_center` when clustering runs. Array-valued output may need conversion before JSON serialization. |
| `return_raw_text` | `False` | Preserves pre-punctuation text where punctuation is applied; does not guarantee this field on every path. |

`generate()` restores saved construction configuration before merging call options. Repeat request-specific options on each call; do not rely on the previous call's language, batch settings, or hotwords. This configuration reset does not establish thread safety or reset every model attribute: for example, a speaker-mode fallback mutates `self.spk_mode`. Serialize access to a shared instance unless your own concurrency tests establish otherwise.

## Local Files and Batching

Install the SDK using the [installation guide](installation/installation.md). The three examples below are standalone scripts with command-line arguments and require prepared local model snapshots and real audio. Their syntax and wrapper contracts are checked without downloading weights; they are not reported as full inference tests.

For this example, invoke the script with a local ASR directory followed by one or more audio paths, for example `python transcribe.py /models/paraformer recording.wav recording2.wav`. The checks fail before model construction if those paths do not exist.

```python
import argparse
from pathlib import Path
from funasr import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("audio", nargs="+")
args = parser.parse_args()
model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
if not model_dir.is_dir():
    raise ValueError("model_dir must be a complete local model directory")
audio = [str(Path(path).expanduser().resolve(strict=True)) for path in args.audio]
model = AutoModel(model=str(model_dir), device="cpu", disable_update=True)
results = model.generate(input=audio, batch_size=1)
for result in results:
    print(result.get("key"), result.get("text", ""))
```

`batch_size=1` processes a list serially through direct inference; raise it only for a model that supports batching and after measuring memory use. `batch_size_s` has a different meaning and only controls the wrapper's VAD path. VAD segmentation is not incremental microphone streaming, and it does not guarantee unlimited recording length or bounded total memory.

Use a one-dimensional mono float32 waveform with a known sample rate. For models using the common audio loader, `fs` in `generate()` describes the array's source sample rate (fallback 16000); the model frontend supplies the target rate. Do not pass a stereo matrix or a list of numbers and assume it will be treated as one waveform: a Python list is normally a list of inputs. File decoding depends on the installed audio backends. Top-level bytes are handled by `load_bytes`: recognized containers are decoded/resampled to 16 kHz, otherwise bytes are interpreted as int16 PCM without sample-rate metadata. Prefer a decoded array with an explicit rate for ambiguous inputs. See [audio loading](../funasr/utils/load_utils.py) and [byte-input tests](../tests/test_load_audio_bytes.py).

`prepare_data_iterator` also accepts manifests. A `.scp` line can be `utterance_id /path/to/audio.wav`; `.jsonl` records use `{"source": "/path/to/audio.wav", "key": "utterance_id"}`. `.json` is not parsed as an arbitrary JSON array by this iterator. Do not expose unrestricted paths or URLs directly to untrusted callers.

## VAD, Timestamps, and Speakers

Run this pipeline example with **ASR, VAD, punctuation, speaker directories, then audio**, in that order. It is intended for a compatible local Paraformer/SeACo-Paraformer, FSMN-VAD, CT-Transformer punctuation, and CAM++ setup, not every model combination.

```python
import argparse
from pathlib import Path
from funasr import AutoModel

parser = argparse.ArgumentParser()
for name in ("asr_dir", "vad_dir", "punc_dir", "spk_dir", "audio"):
    parser.add_argument(name)
args = parser.parse_args()
paths = {name: Path(value).expanduser().resolve(strict=True)
         for name, value in vars(args).items()}
if not all(paths[name].is_dir() for name in ("asr_dir", "vad_dir", "punc_dir", "spk_dir")):
    raise ValueError("All model arguments must be complete local directories")
model = AutoModel(
    model=str(paths["asr_dir"]),
    vad_model=str(paths["vad_dir"]),
    punc_model=str(paths["punc_dir"]),
    spk_model=str(paths["spk_dir"]),
    spk_mode="punc_segment",
    device="cpu",
    disable_update=True,
)
for result in model.generate(input=str(paths["audio"]), return_spk_res=True):
    print(result.get("text", ""))
    for sentence in result.get("sentence_info", []):
        print(sentence.get("start"), sentence.get("end"),
              sentence.get("spk"), sentence.get("text", ""))
```

With `vad_model` configured, the wrapper detects regions, sorts them by duration for ASR batching, restores their order, and offsets timestamps to the original recording. Generic speaker clustering runs in this VAD path; setting only `spk_model` does not add diarization to direct inference. `punc_segment` needs usable punctuation and timestamps; missing punctuation or timestamp fields can change the mode to `vad_segment`. VAD boundaries are speech-region boundaries, not necessarily speaker changes. Labels such as `spk=0` are anonymous within a result, not verified identities or stable IDs across recordings.

For `sentence_timestamp=True` without speakers, the current wrapper can return VAD-aligned sentence records when both punctuation and token timestamps are absent, or when punctuation/timestamp lengths fail alignment. Other combinations can return an empty `sentence_info`, including timestamps present but no punctuation result. This flag does not create a forced aligner.

### Result Fields

`generate()` returns a Python `list` of model-result dictionaries, not an OpenAI response object. Do not assume every model produces every field, or index `results[0]` without checking: some VAD empty-text paths can skip a result. No-speech VAD results commonly contain `{"key": ..., "text": "", "timestamp": []}`.

| Field | Meaning |
|---|---|
| `key` | Input identifier, often a file stem or manifest key. Random keys are generated in other cases; they are not guaranteed globally unique. |
| `text` | Decoded text. May contain model-specific rich tags; it is not necessarily plain display text. |
| `timestamp` | When produced by the ASR paths described here: token/word/character interval pairs `[[start_ms, end_ms], ...]`. Granularity and availability depend on model/checkpoint. |
| `timestamps` | Nano's CTC-backed alignment can produce dictionaries with `token`, `start_time`, `end_time` in **seconds**. VAD merging offsets these and can also add millisecond `timestamp` pairs. |
| `sentence_info` | Pipeline sentence dictionaries with `start`/`end` in **milliseconds**, `text`, optional `timestamp`, and `spk` when speaker assignment runs. Some paths also include `sentence`. |
| `raw_text` | Optional pre-punctuation text, not a guaranteed untouched copy after all possible processing. |
| `value` | A standalone VAD model's speech intervals, not ASR text. Online VAD can emit incomplete boundaries; follow that model's streaming protocol. |

Additional fields such as `words`, `ctc_timestamps`, or speaker embeddings are model-specific. Do not relabel SDK milliseconds as service seconds without conversion. See [timestamp regression tests](../tests/test_paraformer_timestamp_contract.py) and the model sources below.

### Consuming SenseVoice Alignment

For this toolkit's SenseVoice path, request `output_timestamp=True` in `generate()`. Pair the returned `words` with `timestamp` in **milliseconds**. These are model alignment units: English subwords can merge into words, while other languages can retain character units. Neither the raw rich-tagged text nor `rich_transcription_postprocess(text)` is a character-by-character index into the timestamps. Display postprocessing and text replacements do not realign audio.

Use this helper on each result that has alignment fields. The length check prevents silent truncation; missing fields require explicit application handling, not invented timestamps.

```python
def sensevoice_word_intervals(result):
    words, timestamps = result.get("words"), result.get("timestamp")
    if words is None or timestamps is None:
        raise ValueError("This result has no word/timestamp alignment")
    if len(words) != len(timestamps):
        raise ValueError("Word/timestamp count mismatch")
    return [
        {"word": word, "start_ms": start_ms, "end_ms": end_ms}
        for word, (start_ms, end_ms) in zip(words, timestamps)
    ]
```

Call `sensevoice_word_intervals(result)` for a dictionary from `model.generate(..., output_timestamp=True)`. Empty aligned lists return an empty list; a no-speech result without `words` raises instead. Keep display-text cleanup separate from the aligned sequence.

A fixed-checkpoint check on FunASR 1.4.15 produced 242 paired units for the public Chinese sample with ITN and 217 without ITN. An English sample had 84 display characters but 16 paired units. These results illustrate the consumption contract, not timestamp precision, transcription accuracy or a fix for every special-symbol case. See the [reproducible inputs and boundaries](https://github.com/QwenAudio/SenseVoice/issues/215#issuecomment-5615363096). This is not native Nano Transformers output and does not add speaker identity.

## Language, Hotwords, and Alignment Are Model-Specific

| Implementation | Runtime controls in this checkout |
|---|---|
| [SeACo-Paraformer](../funasr/models/seaco_paraformer/model.py) | `hotword` (singular), fallback `None`: a whitespace-separated string, local `.txt` path, or supported URL. This parser expects a string, not Nano's list form. The `paraformer-zh` ModelScope alias maps to this implementation's checkpoint. |
| [Paraformer](../funasr/models/paraformer/model.py) | `pred_timestamp` takes precedence when supplied; otherwise `output_timestamp` falls back to `False`. A checkpoint's saved options may enable timestamps. Do not generalize one Paraformer variant's hotword/timestamp behavior to all variants. |
| [SenseVoice](../funasr/models/sense_voice/model.py) | `language="auto"`; language tokens include `zh`, `en`, `yue`, `ja`, `ko`. Unknown hints fall back to the auto token. `use_itn=False`; explicit `text_norm` overrides it. `output_timestamp=False`; enabling it uses this implementation's CTC alignment path. Generic decoder hotwords are not consumed here. |
| [Fun-ASR-Nano](../funasr/models/fun_asr_nano/model.py) | `hotwords=[]` (plural, list of strings), `language=None`, `itn=True`. Language is inserted into a text prompt, not normalized through the SenseVoice token table. CTC timestamps depend on loaded components and complete checkpoint weights, not just an output flag. |
| [Qwen3-ASR adapter](../funasr/models/qwen3_asr/model.py) | `language=None`; `auto` becomes `None`, and implemented ISO aliases such as `zh`/`en` map to full language names. `context=""` is the contextual prompt, not `hotword`. `return_time_stamps=False` or `output_timestamp=False`; either enables a request for timestamps, but a `forced_aligner` must have been configured at construction. Without it, the adapter warns and skips timestamps. |

These are control conventions, not promises of language coverage, accuracy, or identical checkpoint contents across hubs. Nano disables CTC output when required tensors are missing; see [checkpoint validation](../funasr/models/fun_asr_nano/checkpoint_utils.py). MOSS-Transcribe-Diarize is a third-party **OpenMOSS** model with native joint transcription/diarization and a separate schema/dependency path: use the [MOSS guide](moss_transcribe_diarize.md), not the external VAD/CAM++ example above.

### Text-Level Hotword Correction

The wrapper also supports `postprocess_hotwords` (string/list/dict) and `postprocess_hotword_file` in `generate()`. Explicit mappings can use `{"wrong": "right"}` or a file line `wrong=>right`. `postprocess_hotword_threshold=0.85`, `postprocess_hotword_fuzzy=True`, and `return_postprocess_hotword_matches=False` are the fallbacks. Fuzzy matching may require `pypinyin` and `rapidfuzz`; setting `postprocess_hotword_fuzzy=False` selects explicit replacement only.

This runs **after decoding**, updates text and sentence text, and intentionally leaves timestamps aligned to the original recognition. It neither biases the acoustic decoder nor re-aligns replacements. Inspect replacement details before using corrected text for word-level subtitles. Pass these options per call: `generate()` sends its `cfg`, not the construction defaults, to this postprocessor. See [implementation](../funasr/utils/postprocess_hotwords.py) and [tests](../tests/test_postprocess_hotwords.py).

## Streaming Cache Lifecycle

The following example is specifically for the local **Paraformer streaming** checkpoint, not an offline ASR model made streaming by `cache={}`. Invoke it with the model directory and a nonempty mono 16 kHz audio file. It follows the [upstream repository demo](../examples/industrial_data_pretraining/paraformer_streaming/demo.py) and [streaming implementation](../funasr/models/paraformer_streaming/model.py).

```python
import argparse
from pathlib import Path
import soundfile as sf
from funasr import AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("audio")
args = parser.parse_args()
model_dir = Path(args.model_dir).expanduser().resolve(strict=True)
if not model_dir.is_dir():
    raise ValueError("model_dir must be a complete local streaming model directory")
speech, sample_rate = sf.read(args.audio, dtype="float32")
if sample_rate != 16000 or speech.ndim != 1 or len(speech) == 0:
    raise ValueError("Expected nonempty mono 16 kHz audio")
model = AutoModel(model=str(model_dir), device="cpu", disable_update=True)
chunk_size = [0, 10, 5]
stride = chunk_size[1] * 960
cache = {}
for start in range(0, len(speech), stride):
    end = min(start + stride, len(speech))
    result = model.generate(
        input=speech[start:end], cache=cache, is_final=end == len(speech),
        chunk_size=chunk_size, encoder_chunk_look_back=4,
        decoder_chunk_look_back=1, batch_size=1,
    )
    print(result)
```

For this implementation, `chunk_size` falls back to `[0, 10, 5]` and `is_final` to `False`. The middle value times 960 gives the input stride: 9600 samples at 16 kHz is 600 ms. Both `encoder_chunk_look_back` and `decoder_chunk_look_back` default to **0** in source; **4** and **1** above are explicit demo settings, not wrapper defaults.

Create a new `cache={}` per stream, pass the same dictionary on every ordered chunk, keep chunk/look-back settings fixed, and set `is_final=True` on the last chunk to flush buffered state. Paraformer streaming currently reinitializes its cache after finalization; discard it anyway when ending, cancelling, or starting a different stream. Do not share caches across recordings or simultaneous users. Keep `batch_size=1`: this implementation asserts one waveform per call. Returned text is for the decoded chunks in that call, not a guaranteed cumulative transcript; retain outputs in the application. WebSocket session protocols and server-side buffering are separate from this dictionary lifecycle.

## Reproducibility and Offline Use

1. Pin the SDK version/commit and dependency versions separately from model artifacts. Record the hub, full model ID, resolved upstream revision, configuration, and weight hashes for ASR and every auxiliary model. An alias or moving `master` is not a reproducible pin.
2. The generic ModelScope loader forwards `model_revision` to its snapshot downloader. The generic Hugging Face helper accepts it but currently calls **`snapshot_download(model)`**, without `revision`, `local_files_only`, or `check_latest`. Do not claim `hub="hf", model_revision=...` pins that path. Some model adapters have their own loading logic; inspect the selected adapter too.
3. For offline operation, prepare and verify complete local snapshots while online, including tokenizer/frontend files, configuration, weights, nested encoders/LLMs, and optional aligners. Pass existing local directories for all components. Local paths bypass generic hub resolution, but nested model code can still contact services or need missing dependencies; test under an actual network restriction.
4. `disable_update=True` only skips the package update check. `check_latest=False` is not an offline guarantee, and these generic helpers do not provide a universal `local_files_only` switch. URL audio and URL hotword inputs still require network access. Cached hub aliases may still trigger hub requests.
5. Keep `trust_remote_code=False` unless audited model code requires otherwise. ModelScope's generic loader can import `remote_code` (default `"model"`) when trusted; both generic hub loaders can install a model's `requirements.txt` when trusted, while their code-import behavior is not identical. A local snapshot is not automatically safe code.

The FunASR software is [MIT-licensed](../LICENSE); model weights, datasets, third-party adapters, and dependencies can have different licenses. Review the selected model's own license and usage conditions before redistribution or deployment.

## SDK and Service Boundaries

| Interface | Contract to use |
|---|---|
| Python `AutoModel` | In-process model configuration, waveform inputs, streaming dictionaries, and model-specific Python results described here. No `base_url`, API key, or HTTP `response_format` contract. |
| `funasr-server` | Packaged [CLI](../funasr/bin/server.py) creates [the server app](../funasr/bin/_server_app.py). Its `/v1/audio/transcriptions` endpoint is an OpenAI-style speech subset; `/asr` is a separate service-specific endpoint. Form fields, defaults, units, and speaker behavior belong to that server, not `generate()`. |
| Example OpenAI-compatible server | [The example server](../examples/openai_api/server.py) is a different implementation from the packaged app. Use its [OpenAPI reference](../examples/openai_api/OPENAPI.md), [client recipes](../examples/openai_api/CLIENTS.md), and [security/gateway guide](../examples/openai_api/SECURITY.md), then compare the deployed server's live `/openapi.json`. Do not assume their defaults or response fields are identical. |
| vLLM | `AutoModelVLLM`, FunASR's model-specific services, and native `vllm serve` are distinct entry points. Neither the SDK's `cache` nor a transcription HTTP endpoint proves realtime-session compatibility. Follow the [vLLM guide](vllm_guide.md) for the chosen route and checkpoint format. |
| llama.cpp / GGUF | Separate C++ executables and GGUF artifacts, not the Python `AutoModel` kwargs or result schema. Use the [runtime guide](../runtime/llama.cpp/README.md). VAD alone is not speaker diarization. |

The [Hydra inference entry point](../funasr/bin/inference.py) constructs `AutoModel` from configuration and calls `generate()`; it does not turn every CLI/config key into an HTTP form field. Before exposing any service, enforce authentication, TLS, upload limits, timeouts, and access restrictions at the appropriate boundary. A placeholder OpenAI-client API key is not access control.
