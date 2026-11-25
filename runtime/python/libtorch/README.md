# Libtorch-python

## Version 0.1.7

This standalone Torch runtime release includes the already-merged fixes for
mixed-length Paraformer timestamp batches, long/empty contextual hotwords,
terminal abbreviations and rich-transcription word boundaries. Contextual and
Seaco hotwords keep their first ten characters, matching the exported embedding
width; empty whitespace entries are omitted. These limits are not configurable
recognition-quality guarantees.

Install `funasr-torch==0.1.7` in a dedicated environment. The root `funasr`
distribution and `funasr-onnx` have separate versions and do not replace this
package. Python WebSocket server-script changes are also separate.

Wheel and source-distribution installation regressions exercise native wrapper
and postprocessing behavior with synthetic acoustic outputs. Exact runtime
source is bound to previously accepted real CUDA fixture tests; this release
does not claim new pretrained ASR accuracy, performance or a full model suite.

## Export the model

### Install [modelscope and funasr](https://github.com/modelscope/FunASR#installation)

```shell
# pip3 install torch torchaudio
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
pip install torch-quant # Optional, for torchscript quantization
pip install onnx onnxruntime # Optional, for onnx quantization
```

### Export [onnx model](https://github.com/modelscope/FunASR/tree/main/funasr/export)

```shell
python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type torch --quantize True
```

## Install the `funasr_torch`

`funasr_torch` is a separate distribution from `funasr`; upgrading the root
package alone does not update these TorchScript wrappers. Version 0.1.5
includes the Contextual/Seaco Paraformer hotword batching and empty
hotword-file handling fix (#3838). This is a runtime wrapper fix, not a
model-weight update or an ASR accuracy claim.

Version 0.1.6 makes the standalone CMVN reader accept blank lines and
raise a file-specific error for missing or truncated statistics (#3842).
Valid float64 statistics and shipped feature output are unchanged. This is
a CPU parser/package fix, not a model-weight or ASR accuracy update.

install from pip

```shell
pip install -U "funasr_torch>=0.1.6"
# For the users in China, you could install with the command:
# pip install -U "funasr_torch>=0.1.6" -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

or install from source code

```shell
git clone https://github.com/modelscope/FunASR.git && cd FunASR
cd runtime/python/libtorch
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

Please ref to [benchmark](https://github.com/modelscope/FunASR/blob/main/runtime/docs/benchmark_libtorch.md)

## Speed

Environment：Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz

Test [wav, 5.53s, 100 times avg.](https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav)

| Backend  | RTF (FP32) |
|:--------:|:----------:|
| Pytorch  |   0.110    |
| Libtorch |   0.048    |
|   Onnx   |   0.038    |

## Acknowledge

This project is maintained by [FunASR community](https://github.com/modelscope/FunASR).
