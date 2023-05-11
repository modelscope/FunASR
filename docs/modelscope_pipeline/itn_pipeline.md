# Inverse Text Normalization (ITN)

> **Note**: 
> The modelscope pipeline supports all the models in [model zoo](https://modelscope.cn/models?page=1&tasks=inverse-text-processing&type=audio) to inference. Here we take the model of the Japanese ITN model as example to demonstrate the usage.

## Inference

### Quick start
#### [Japanese ITN model](https://modelscope.cn/models/damo/speech_inverse_text_processing_fun-text-processing-itn-ja/summary)
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

itn_inference_pipline = pipeline(
    task=Tasks.inverse_text_processing,
    model='damo/speech_inverse_text_processing_fun-text-processing-itn-ja',
    model_revision=None)

itn_result = itn_inference_pipline(text_in='百二十三')
print(itn_result)
# 123
```
- read text data directly.
```python
rec_result = inference_pipeline(text_in='一九九九年に誕生した同商品にちなみ、約三十年前、二十四歳の頃の幸四郎の写真を公開。')
# 1999年に誕生した同商品にちなみ、約30年前、24歳の頃の幸四郎の写真を公開。
```
- text stored via url，example：https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/ja_itn_example.txt
```python
rec_result = inference_pipeline(text_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/ja_itn_example.txt')
```

Full code of demo, please ref to [demo](https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing/inverse_text_normalization)

### API-reference
#### Define pipeline
- `task`: `Tasks.inverse_text_processing`
- `model`: model name in [model zoo](https://modelscope.cn/models?page=1&tasks=inverse-text-processing&type=audio), or model path in local disk
- `output_dir`: `None` (Default), the output path of results if set
- `model_revision`: `None` (Default), setting the model version

#### Infer pipeline
- `text_in`: the input to decode, which could be:
  - text bytes, `e.g.`: "一九九九年に誕生した同商品にちなみ、約三十年前、二十四歳の頃の幸四郎の写真を公開。"
  - text file, `e.g.`: https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_text/ja_itn_example.txt
  In this case of `text file` input, `output_dir` must be set to save the output results

## Modify Your Own ITN Model
The rule-based ITN code is open-sourced in [FunTextProcessing](https://github.com/alibaba-damo-academy/FunASR/tree/main/fun_text_processing), users can modify by their own grammar rules for different languages. Let's take Japanese as an example, users can add their own whitelist in ```FunASR/fun_text_processing/inverse_text_normalization/ja/data/whitelist.tsv```. After modified the grammar rules, the users can export and evaluate their own ITN models in local directory.

### Export ITN Model
Export ITN model via ```FunASR/fun_text_processing/inverse_text_normalization/export_models.py```. An example to export ITN model to local folder is shown as below.
```shell
cd FunASR/fun_text_processing/inverse_text_normalization/
python export_models.py --language ja --export_dir ./itn_models/
```

### Evaluate ITN Model
Users can evaluate their own ITN model in local directory via ```FunASR/fun_text_processing/inverse_text_normalization/inverse_normalize.py```. Here is an example:
```shell
cd FunASR/fun_text_processing/inverse_text_normalization/
python inverse_normalize.py --input_file ja_itn_example.txt --cache_dir ./itn_models/ --output_file output.txt --language=ja
```