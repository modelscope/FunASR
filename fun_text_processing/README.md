**Fundamental Text Processing (FunTextProcessing)**
==========================

### Introduction

FunTextProcessing is a Python toolkit for fundamental text processing in ASR including text processing , inverse text processing, which is included in the `FunASR`.

### Highlights

- FunTextProcessing supports inverse text processing (ITN), text processing (TN).
- FunTextProcessing supports multilingual, 10+ languages for ITN, 5 languages for TN.


### Example
#### Inverse Text Processing (ITN)
Given text inputs, such as speech recognition results, use `fun_text_processing/inverse_text_normalization/inverse_normalize.py` to output ITN results. You may refer to the following example scripts.

```
test_file=fun_text_processing/inverse_text_normalization/id/id_itn_test_input.txt

python fun_text_processing/inverse_text_normalization/inverse_normalize.py --input_file $test_file --cache_dir ./itn_model/ --output_file output.txt --language=id
```


### Acknowledge
1. We borrowed a lot of codes from [NeMo](https://github.com/NVIDIA/NeMo).
2. We refered the codes from [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) for Chinese inverse text normalization. 

### License

This project is licensed under the Apache-2.0 license. FunTextProcessing also contains various third-party components and some code modified from other repos under other open source licenses. 
