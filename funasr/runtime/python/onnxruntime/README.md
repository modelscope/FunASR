## Using funasr with ONNXRuntime


### Introduction
- Model comes from [speech_paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary).


### Steps:
1. Export the model.
   - Command: (`Tips`: torch >= 1.11.0 is required.)

       More details ref to ([export docs](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export))

       - `e.g.`, Export model from modelscope
         ```shell
         python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize False
         ```
       - `e.g.`, Export model from local path, the model'name must be `model.pb`.
         ```shell
         python -m funasr.export.export_model --model-name ./damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type onnx --quantize False
         ```


2. Install the `funasr_onnx`

install from pip
```shell
pip install --upgrade funasr_onnx -i https://pypi.Python.org/simple
```

or install from source code

```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/funasr_onnx
python setup.py build
python setup.py install
```

3. Run the demo.
   - Model_dir: the model path, which contains `model.onnx`, `config.yaml`, `am.mvn`.
   - Input: wav formt file, support formats: `str, np.ndarray, List[str]`
   - Output: `List[str]`: recognition result.
   - Example:
        ```python
        from funasr_onnx import Paraformer

        model_dir = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        model = Paraformer(model_dir, batch_size=1)

        wav_path = ['/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav']

        result = model(wav_path)
        print(result)
        ```

## Performance benchmark

Please ref to [benchmark](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/python/benchmark_onnx.md)

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [SWHL](https://github.com/RapidAI/RapidASR) for contributing the onnxruntime (for paraformer model).
