# FAQ and Troubleshooting

This page collects the questions that most often block new FunASR users. Start here before opening an issue.

## Which install command should I use?

For most users:

```bash
pip install -U funasr
```

For the newest examples, server CLI, or unreleased fixes:

```bash
git clone https://github.com/modelscope/FunASR.git
cd FunASR
pip install -e ./
```

For the OpenAI-compatible API server, install the web runtime dependencies too:

```bash
pip install funasr fastapi uvicorn python-multipart
```

## Which Python and PyTorch versions are recommended?

Use Python 3.8 or later and install a PyTorch/torchaudio pair that matches your CUDA runtime. If CUDA is not configured, start with CPU to verify the workflow first:

```bash
funasr-server --model sensevoice --device cpu
```

After the CPU smoke test works, switch to CUDA:

```bash
funasr-server --model sensevoice --device cuda
```

## Model download is slow or fails. What should I check?

FunASR models are available from ModelScope and Hugging Face. Choose the hub that is fastest in your network environment, and make sure the machine can reach it before debugging model code.

Common checks:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import funasr; print(funasr.__version__)"
```

If a model is already downloaded on another machine, use the local model path instead of the remote model name.

## `funasr-server` says FastAPI or multipart packages are missing

Install the server dependencies:

```bash
pip install fastapi uvicorn python-multipart
```

Then start again:

```bash
funasr-server --model sensevoice --device cuda
```

## Port 8000 is already in use

Start the service on another port:

```bash
funasr-server --model sensevoice --device cuda --port 9000
```

Then point clients to the new base URL:

```bash
curl http://localhost:9000/health
```

## How do I verify the OpenAI-compatible API quickly?

Start the server:

```bash
funasr-server --model sensevoice --device cuda
```

In another terminal:

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

The response should include `text`. With `verbose_json`, supported models may also return segment-level information.

## How do I run the OpenAI-compatible API with Docker Compose?

Use the example Compose setup when you want a reproducible local smoke test before wiring the API into a product or agent workflow:

```bash
cd examples/openai_api
cp .env.example .env
docker compose up --build
```

Then verify it from another terminal:

```bash
BASE_URL=http://localhost:8000 bash smoke_test.sh
```

The example container defaults to `FUNASR_DEVICE=cpu` so it can start on machines without NVIDIA Container Toolkit. For CUDA, first adapt the image to use CUDA-capable PyTorch/FunASR dependencies, then set `FUNASR_DEVICE=cuda`.

See also:

- [OpenAI API example](../../examples/openai_api/)
- [Client recipes](../../examples/openai_api/CLIENTS.md)
- [Deployment matrix](../deployment_matrix.md)

## Docker starts but `/health` or transcription fails

Check the container logs first:

```bash
docker compose logs -f funasr-api
```

Common causes:

- The first startup is still downloading or loading the model.
- Port 8000 is already in use; set `FUNASR_HOST_PORT=9000` in `.env` and use `BASE_URL=http://localhost:9000`.
- The container is running in CPU mode but `.env` or the command expects CUDA.
- CUDA is requested but the image does not include CUDA-capable PyTorch/FunASR dependencies.
- The model cache volume is empty or corrupted; retry after removing the `funasr-cache` Docker volume.
- The uploaded audio file is too large for the machine; verify with the public sample before testing long recordings.

When opening a Deployment Help issue, include your `.env` values without secrets, the `docker compose` command, container logs, model alias, device, and audio duration.

## Long audio is slow, split incorrectly, or runs out of memory

Use VAD segmentation for long audio and tune segment length for your hardware:

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda",
)
result = model.generate(input="long_meeting.wav", batch_size_s=300)
```

If memory is limited, reduce `batch_size_s`, use CPU for verification, or split very long recordings before batch processing.

## Speaker diarization has no speaker labels

Use a model pipeline that includes both VAD and speaker models:

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    spk_model="cam++",
    device="cuda",
)
result = model.generate(input="meeting.wav")
```

Then inspect `result[0]["sentence_info"]`. Each sentence should include fields such as `text`, `start`, `end`, and `spk` when diarization is available.

## The same command works on CPU but fails on CUDA

This usually points to a CUDA, driver, PyTorch, or GPU memory mismatch. Include these checks in your issue:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torchaudio; print(torchaudio.__version__)"
```

Try a smaller model or lower batch size to rule out GPU memory pressure.

## What information should I include in an issue?

Please include:

- OS and Python version
- FunASR version and install method (`pip`, source, Docker)
- PyTorch, torchaudio, CUDA, and GPU information
- Exact command or minimal Python snippet
- Full traceback or server logs
- Model name and hub (`modelscope`, `hf`, or local path)
- Audio duration, sample rate, format, language, speaker count, and whether the audio can be shared

## Existing ModelScope pipeline examples

- [VAD model with ModelScope pipeline](https://github.com/modelscope/FunASR/discussions/236)
- [Punctuation model with ModelScope pipeline](https://github.com/modelscope/FunASR/discussions/238)
- [Paraformer streaming with ModelScope pipeline](https://github.com/modelscope/FunASR/discussions/241)
- [VAD + ASR + punctuation with ModelScope pipeline](https://github.com/modelscope/FunASR/discussions/278)
- [VAD + ASR + punctuation + NNLM with ModelScope pipeline](https://github.com/modelscope/FunASR/discussions/134)
- [Timestamp prediction with ModelScope pipeline](https://github.com/modelscope/FunASR/discussions/246)
- [Switch online/offline decoding for UniASR](https://github.com/modelscope/FunASR/discussions/151)
