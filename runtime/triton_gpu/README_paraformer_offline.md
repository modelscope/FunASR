## Inference with Triton 

### Steps:
1. Prepare model repo files
```sh
git-lfs install
git clone https://www.modelscope.cn/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git

pretrained_model_dir=$(pwd)/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

cp $pretrained_model_dir/am.mvn ./model_repo_paraformer_large_offline/feature_extractor/
cp $pretrained_model_dir/config.yaml ./model_repo_paraformer_large_offline/feature_extractor/
cp $pretrained_model_dir/tokens.json ./model_repo_paraformer_large_offline/scoring/1/

# Refer here to get model.onnx (https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/export/README.md)
cp <exported_onnx_dir>/model.onnx ./model_repo_paraformer_large_offline/encoder/1/
```
Log of directory tree:
```sh
model_repo_paraformer_large_offline/
|-- encoder
|   |-- 1
|   |   `-- model.onnx
|   `-- config.pbtxt
|-- feature_extractor
|   |-- 1
|   |   `-- model.py
|   |-- config.pbtxt
|   |-- am.mvn
|   `-- config.yaml
|-- infer_pipeline
|   |-- 1
|   `-- config.pbtxt
`-- scoring
    |-- 1
    |   `-- model.py
    |    -- tokens.json
    `-- config.pbtxt

8 directories, 9 files
```

2. Follow below instructions to launch triton server
```sh
# using docker image Dockerfile/Dockerfile.server
docker build . -f Dockerfile/Dockerfile.server -t triton-paraformer:23.01 
docker run -it --rm --name "paraformer_triton_server" --gpus all -v <path_host/model_repo_paraformer_large_offline>:/workspace/ --shm-size 1g --net host triton-paraformer:23.01 

# launch the service 
tritonserver --model-repository /workspace/model_repo_paraformer_large_offline \
             --pinned-memory-pool-byte-size=512000000 \
             --cuda-memory-pool-byte-size=0:1024000000

```

### Performance benchmark

Benchmark [speech_paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) based on Aishell1 test set with a single V100, the total audio duration is 36108.919 seconds.

```sh
# For client container:
docker run -it --rm --name "client_test" --net host --gpus all -v <path_host/triton_gpu/client>:/workpace/ soar97/triton-k2:22.12.1 # noqa
# For aishell manifests:
apt-get install git-lfs
git-lfs install
git clone https://huggingface.co/csukuangfj/aishell-test-dev-manifests
sudo mkdir -p /root/fangjun/open-source/icefall-aishell/egs/aishell/ASR/download/aishell
tar xf ./aishell-test-dev-manifests/data_aishell.tar.gz -C /root/fangjun/open-source/icefall-aishell/egs/aishell/ASR/download/aishell/ # noqa

serveraddr=localhost
manifest_path=/workspace/aishell-test-dev-manifests/data/fbank/aishell_cuts_test.jsonl.gz
num_task=60
python3 client/decode_manifest_triton.py \
    --server-addr $serveraddr \
    --compute-cer \
    --model-name infer_pipeline \
    --num-tasks $num_task \
    --manifest-filename $manifest_path
```

(Note: The service has been fully warm up.)
|concurrent-tasks | processing time(s) | RTF |
|----------|--------------------|------------|
| 60 (onnx fp32)                | 116.0 | 0.0032|

## Acknowledge
This part originates from NVIDIA CISI project. We also have TTS and NLP solutions deployed on triton inference server. If you are interested, please contact us.
