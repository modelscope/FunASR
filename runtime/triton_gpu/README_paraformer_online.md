### Steps:
1. Prepare model repo files
* git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx.git
* Convert lfr_cmvn_pe.onnx model. For example: python export_lfr_cmvn_pe_onnx.py
* If you export to onnx, you should have several model files in `${MODEL_DIR}`:
```
├── README.md
└── model_repo_paraformer_large_online
    ├── cif_search
    │   ├── 1
    │   │   └── model.py
    │   └── config.pbtxt
    ├── decoder
    │   ├── 1
    │   │   └── decoder.onnx
    │   └── config.pbtxt
    ├── encoder
    │   ├── 1
    │   │   └── model.onnx
    │   └── config.pbtxt
    ├── feature_extractor
    │   ├── 1
    │   │   └── model.py
    │   ├── config.pbtxt
    │   └── config.yaml
    ├── lfr_cmvn_pe
    │   ├── 1
    │   │   └── lfr_cmvn_pe.onnx
    │   ├── am.mvn
    │   ├── config.pbtxt
    │   └── export_lfr_cmvn_pe_onnx.py
    └── streaming_paraformer
        ├── 1
        └── config.pbtxt
```

2. Follow below instructions to launch triton server
```sh
# using docker image Dockerfile/Dockerfile.server
docker build . -f Dockerfile/Dockerfile.server -t triton-paraformer:23.01 
docker run -it --rm --name "paraformer_triton_server" --gpus all -v <path_host/model_repo_paraformer_large_online>:/workspace/ --shm-size 1g --net host triton-paraformer:23.01 

# launch the service 
cd /workspace
tritonserver --model-repository model_repo_paraformer_large_online \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000

```

### Performance benchmark with a single A10

* FP32, onnx, [paraformer larger online](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/summary
),Our chunksize is 10 * 960 / 16000 = 0.6 s, so we should care about the perf of latency less than 0.6s so that it can be a realtime application.


| Concurrency | Throughput | Latency_p50 (ms) | Latency_p90 (ms) | Latency_p95 (ms) | Latency_p99 (ms) |
|-------------|------------|------------------|------------------|------------------|------------------|
| 20          | 309.252    | 56.913          | 76.267          | 85.598          | 138.462          |
| 40          | 391.058    | 97.911           | 145.509          | 150.545          | 185.399          |
| 60          | 426.269    | 138.244          | 185.855          | 201.016          | 236.528          |
| 80          | 431.781    | 170.991          | 227.983          | 252.453          | 412.273          |
| 100         | 473.351    | 206.205          | 262.612          | 288.964          | 463.337          |

