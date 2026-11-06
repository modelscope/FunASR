# Runtime Evaluation Tools

Run these commands from a FunASR checkout with the chosen runtime and its
dependencies available. Importing the scripts or using `--help` does not load a
runtime, audio, or model. ONNX and Torch backends are selected independently.

Use the checkout's runtime modules when evaluating a source change:

```sh
export PYTHONPATH="$PWD/runtime/python/onnxruntime:$PWD/runtime/python/libtorch${PYTHONPATH:+:$PYTHONPATH}"
python runtime/python/utils/test_rtf.py --model_dir /path/to/exported-model --wav_file wav.scp
python runtime/python/utils/test_rtf_gpu.py --model_dir /path/to/exported-model --wav_file wav.scp --device_id 0 --batch_size 8
python runtime/python/utils/test_cer.py --backend torch --model_dir /path/to/exported-model --wav_file wav.scp --output_dir transcripts
```

`wav.scp` contains an utterance ID followed by whitespace and an audio path.
Paths may contain spaces. Blank lines are ignored; empty or malformed lists
are rejected. Supply existing local exported models to avoid runtime download
or export behavior. Torch uses `model.torchscript`; ONNX uses `model.onnx`.
The corresponding quantized names are selected with `--quantize true`.

The serial RTF and transcript tools default to CPU (`--device_id -1`). The
batched GPU tool defaults to CUDA device 0 and rejects negative device IDs.
It requires CUDA-capable Torch or ONNX Runtime, with compatible CUDA/cuDNN
libraries. Missing ONNX CUDA providers cause an error instead of a CPU timing
being reported as a GPU benchmark. ONNX may still place supported graph
operations on CPU; provider availability alone is not proof that every node
runs on CUDA. Use runtime profiling when auditing actual kernel placement.

RTF is elapsed wall time divided by actual input audio duration. There is an
untimed audio-duration prepass and, by default, 30 warmups on the first audio
file (`--warmup 0` disables warmups). Timed model calls include file loading,
frontend, inference and decoding; Torch CUDA is synchronized around timing.
The serial tool invokes each file separately. The GPU tool passes the list to
the backend, whose `--batch_size` controls batching. These numbers are not
model-only kernel timings and depend on cache state and hardware.

The transcript tool writes UTF-8 `text` and space-delimited `token` files. It
does not compute CER/WER or alter references. A missing inference result is an
error, not a successful empty transcript or benchmark.

Offline regression tests use small synthetic graphs and generated audio:

```sh
python -m pytest tests/test_runtime_eval_cli.py tests/test_runtime_eval_native.py
FUNASR_TEST_DEVICE=0 python -m pytest tests/test_runtime_eval_native.py
```

Native tests require Torch, ONNX, ONNX Runtime and the runtimes' audio/tokenizer
dependencies. They test device and CLI mechanics, not pretrained ASR accuracy.
