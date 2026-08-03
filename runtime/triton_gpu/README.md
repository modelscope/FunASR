## Triton Inference Serving Best Practice for SenseVoice

### Quick Start
Directly launch the service using docker compose.
```sh
docker compose up --build
```

### Build Image
Build the docker image from scratch. 
```sh
# build from scratch, cd to the parent dir of Dockerfile.server
docker build . -f Dockerfile/Dockerfile.sensevoice -t soar97/triton-sensevoice:24.05
```

### Create Docker Container
```sh
your_mount_dir=/mnt:/mnt
docker run -it --name "sensevoice-server" --gpus all --net host -v $your_mount_dir --shm-size=2g soar97/triton-sensevoice:24.05
```

### Export SenseVoice Model to Onnx
Please follow the official FunASR guide to export the SenseVoice ONNX file. Also,
download the tokenizer file used by the scoring model.

The default deployment uses Triton's ONNX Runtime backend. Export an unquantized
graph when you plan to build a native TensorRT engine:

```python
from funasr import AutoModel

model = AutoModel(model="iic/SenseVoiceSmall", device="cuda:0")
model.export(
    type="onnx",
    quantize=False,
    device="cuda:0",
    output_dir="./sensevoice_onnx",
    max_seq_len=4096,
)
```

Do not use `model_quant.onnx` for native TensorRT. Dynamic ONNX quantization adds
`DynamicQuantizeLinear` and `MatMulInteger`, which are not supported by this
TensorRT path. The builder detects these operators and exits with an actionable
error before starting an expensive engine build.

### Build a Native TensorRT Engine

Build the plan on the same GPU architecture and TensorRT version used by the
target Triton server. TensorRT plans are not portable across GPU compute
capabilities or arbitrary TensorRT versions.

Inside the target Triton environment, install ONNX if needed and run:

```sh
pip install "onnx>=1.16"

python runtime/triton_gpu/scripts/build_sensevoice_tensorrt.py \
    ./sensevoice_onnx/model.onnx \
    runtime/triton_gpu/model_repo_sense_voice_small/encoder/1/model.plan \
    --precision fp16 \
    --min-batch 1 --opt-batch 8 --max-batch 16 \
    --min-frames 1 --opt-frames 512 --max-frames 4096 \
    --workspace-gb 8

cp runtime/triton_gpu/model_repo_sense_voice_small/encoder/config.pbtxt.tensorrt \
   runtime/triton_gpu/model_repo_sense_voice_small/encoder/config.pbtxt
```

The frame bounds apply after the SenseVoice LFR frontend. With the default
frontend, one feature frame represents approximately 60 ms of audio. Tune the
optimization profile to production traffic; larger maximum batch and frame
bounds increase build time and may require more GPU memory. The script validates
the ONNX checker result, exact SenseVoice tensor contract, profile ordering, GPU
FP16 capability, TensorRT parser result, and atomic plan output.

The maintained baseline was verified with TensorRT 10.0.1 on an NVIDIA H100:

| Check | Result |
|---|---|
| FP32 ONNX parser | 0 TensorRT errors |
| FP16 plan, batch 1-16, frames 1-4096 | 527,504,916 bytes; 113.9 s build |
| Random features, 30 and 64 frames | 100% CTC top-1 agreement with PyTorch |
| Bundled Chinese example | Exact transcript: `开饭时间早上九点至下午五点` |

Keep `config.pbtxt` unchanged to continue using ONNX Runtime, or replace it with
the provided `config.pbtxt.tensorrt` after placing `model.plan` in `encoder/1`.

### Launch Server
Log of directory tree:
```sh
model_repo_sense_voice_small
|-- encoder
|   |-- 1
|   |   `-- model.onnx -> /your/path/model.onnx
|   `-- config.pbtxt
|-- feature_extractor
|   |-- 1
|   |   `-- model.py
|   |-- am.mvn
|   |-- config.pbtxt
|   `-- config.yaml
|-- scoring
|   |-- 1
|   |   `-- model.py
|   |-- chn_jpn_yue_eng_ko_spectok.bpe.model -> /your/path/chn_jpn_yue_eng_ko_spectok.bpe.model
|   `-- config.pbtxt
`-- sensevoice
    |-- 1
    `-- config.pbtxt

8 directories, 10 files


# launch the service 
tritonserver --model-repository /workspace/model_repo_sensevoice_small \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000
```


### Benchmark using Dataset
```sh
git clone https://github.com/yuekaizhang/Triton-ASR-Client.git
cd Triton-ASR-Client
num_task=32
python3 client.py \
    --server-addr localhost \
    --server-port 10086 \
    --model-name sensevoice \
    --compute-cer \
    --num-tasks $num_task \
    --batch-size 16 \
    --manifest-dir ./datasets/aishell1_test
```

Benchmark results below were based on Aishell1 test set with a single V100, the total audio duration is 36108.919 seconds.
|concurrent-tasks | batch-size-per-task | processing time(s) | RTF |
|----------|--------------------|------------|---------------------|
| 32 (onnx fp32)                | 16 | 67.09 | 0.0019|
| 32 (onnx fp32)                | 1 | 82.04  | 0.0023|

(Note: for batch-size-per-task=1 cases, tritonserver could use dynamic batching to improve throughput.)

## Acknowledge
This part originates from NVIDIA CISI project. We also have TTS and NLP solutions deployed on triton inference server. If you are interested, please contact us.
