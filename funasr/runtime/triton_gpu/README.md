## Inference with Triton 

### Steps:
1. Refer here to [get model.onnx](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/export/README.md)

2. Follow below instructions to using triton
```sh
# using docker image Dockerfile/Dockerfile.server
docker build . -f Dockerfile/Dockerfile.server -t triton-paraformer:23.01 
docker run -it --rm --name "paraformer_triton_server" --gpus all -v <path_host/funasr/runtime/>:/workspace --shm-size 1g --net host triton-paraformer:23.01 
# inside the docker container, prepare previous exported model.onnx
mv <path_model.onnx> /workspace/triton_gpu/model_repo_paraformer_large_offline/encoder/1/

model_repo_paraformer_large_offline/
|-- encoder
|   |-- 1
|   |   `-- model.onnx
|   `-- config.pbtxt
|-- feature_extractor
|   |-- 1
|   |   `-- model.py
|   |-- config.pbtxt
|   `-- config.yaml
|-- infer_pipeline
|   |-- 1
|   `-- config.pbtxt
`-- scoring
    |-- 1
    |   `-- model.py
    |-- config.pbtxt
    `-- token_list.pkl

8 directories, 9 files

# launch the service 
tritonserver --model-repository ./model_repo_paraformer_large_offline \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000

```

### Performance benchmark

Benchmark [speech_paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) based on Aishell1 test set with a single V100, the total audio duration is 36108.919 seconds.

(Note: The service has been fully warm up.)
|concurrent-tasks | processing time(s) | RTF |
|----------|--------------------|------------|
| 60 (onnx fp32)                | 116.0 | 0.0032|

## Acknowledge
This part originates from NVIDIA CISI project. We also have TTS and NLP solutions deployed on triton inference server. If you are interested, please contact us.
