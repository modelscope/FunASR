# Libtorch-python

## Export the model

### Install [modelscope and funasr](https://github.com/alibaba-damo-academy/FunASR#installation)

```shell
# pip3 install torch torchaudio
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
pip install torch-quant # Optional, for torchscript quantization
pip install onnx onnxruntime # Optional, for onnx quantization
```

### Export [onnx model](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)

```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type torch --quantize True
```

## Install the `funasr_torch`

install from pip

```shell
pip install -U funasr_torch
# For the users in China, you could install with the command:
# pip install -U funasr_torch -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

or install from source code

```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/libtorch
pip install -e ./
# For the users in China, you could install with the command:
# pip install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

## Run the demo

- Model_dir: the model path, which contains `model.torchscript`, `config.yaml`, `am.mvn`.
- Input: wav formt file, support formats: `str, np.ndarray, List[str]`
- Output: `List[str]`: recognition result.
- Example:

     ```python
     from funasr_torch import Paraformer

     model_dir = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
     model = Paraformer(model_dir, batch_size=1)

     wav_path = ['/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav']

     result = model(wav_path)
     print(result)
     ```

## Performance benchmark

Please ref to [benchmark](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/benchmark_libtorch.md)

## Speed

Environment：Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz

Test [wav, 5.53s, 100 times avg.](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav)

| Backend  | RTF (FP32) |
|:--------:|:----------:|
| Pytorch  |   0.110    |
| Libtorch |   0.048    |
|   Onnx   |   0.038    |

## Acknowledge

This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
