## Using paraformer with ONNXRuntime

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg"></a>
</p>

### Introduction
- Model comes from [speech_paraformer](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary).


### Steps:
1. Download the whole directory (`funasr/runtime/python/onnxruntime`) to the local.
2. Install the related packages.
   ```bash
   pip install requirements.txt
   ```
3.  Export the model.
    - Export your model([docs](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/export)), or [Download Link](https://swap.oss-cn-hangzhou.aliyuncs.com/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.onnx?OSSAccessKeyId=LTAI4FxMqzhBUx5XD4mKs296&Expires=2036094510&Signature=agmtMkxLEviGg3Rt3gOO4PvfrJY%3D)
    - Put the model into the `resources/models`.
        ```text
        .
        ├── demo.py
        ├── rapid_paraformer
        │   ├── __init__.py
        │   ├── kaldifeat
        │   ├── __pycache__
        │   ├── rapid_paraformer.py
        │   └── utils.py
        ├── README.md
        ├── requirements.txt
        ├── test_onnx.py
        ├── tests
        │   ├── __pycache__
        │   └── test_infer.py
        └── test_wavs
            ├── 0478_00017.wav
            └── asr_example_zh.wav
        ```
4. Run the demo.
   - Input: wav formt file, support formats: `str, np.ndarray, List[str]`
   - Output: `List[str]`: recognition result.
   - Example:
        ```python
        from paraformer_onnx import Paraformer


        config_path = 'resources/config.yaml'
        model = Paraformer(config_path)

        wav_path = ['example/asr_example.wav']

        result = model(wav_path)
        print(result)
        ```

## Acknowledge
1. We acknowledge [SWHL](https://github.com/RapidAI/RapidASR) for contributing the onnxruntime(python api).
