## Using funasr with libtorch

[FunASR](https://github.com/alibaba-damo-academy/FunASR) hopes to build a bridge between academic research and industrial applications on speech recognition. By supporting the training & finetuning of the industrial-grade speech recognition model released on ModelScope, researchers and developers can conduct research and production of speech recognition models more conveniently, and promote the development of speech recognition ecology. ASR for Fun！

### Introduction
- Model comes from [speech_paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary).

### Steps:
1. Export the model.
   - Command: (`Tips`: torch >= 1.11.0 is required.)

       More details ref to ([export docs](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export))

       - `e.g.`, Export model from modelscope
         ```shell
         python -m funasr.export.export_model --model-name damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type torch --quantize False
         ```
       - `e.g.`, Export model from local path, the model'name must be `model.pb`.
         ```shell
         python -m funasr.export.export_model --model-name ./damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --export-dir ./export --type torch --quantize False
         ```


2. Install the `funasr_torch`.
    
    install from pip
    ```shell
    pip install --upgrade funasr_torch -i https://pypi.Python.org/simple
    ```
    or install from source code

    ```shell
    git clone https://github.com/alibaba/FunASR.git && cd FunASR
    cd funasr/runtime/python/libtorch
    python setup.py build
    python setup.py install
    ```

3. Run the demo.
   - Model_dir: the model path, which contains `model.torchscripts`, `config.yaml`, `am.mvn`.
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

Please ref to [benchmark](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/python/benchmark_libtorch.md)

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
