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
Please follow the official guide of FunASR to export the sensevoice onnx file. Also, you need to download the tokenizer file by yourself. 
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
