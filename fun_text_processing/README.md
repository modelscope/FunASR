**Fundamental Text Processing (FunTextProcessing)**
==========================

### Introduction

FunTextProcessing is a Python toolkit for fundamental text processing in ASR including text processing , inverse text processing, which is included in the `FunASR`.

### Highlights

- FunTextProcessing supports inverse text processing (ITN), text processing (TN).
- FunTextProcessing supports multilingual, 10+ languages for ITN, 5 languages for TN.

#### Installation

Fun Text Processing, specifically (Inverse) Text Normalization, requires Pynini to be installed.
```
bash fun_text_processing/install_pynini.sh
```

### Example
#### Inverse Text Processing (ITN)
Given text inputs, such as speech recognition results, use `fun_text_processing/inverse_text_normalization/inverse_normalize.py` to output ITN results. You may refer to the following example scripts.

```
python fun_text_processing/inverse_text_normalization/inverse_normalize.py --text="ratus dua puluh tiga" --language=id
```

Arguments:
- text - Input text. Should not exceed 500 words.
- input_file - Input file with lines of input text. Only one of text or input_file is accepted.
- output_file - Output file to save normalizations. Needed if input_file is specified.
- language - language id.
- input_case - Only for text normalization. lower_cased or cased.
- verbose - Outputs intermediate information.
- cache_dir - Specifies a cache directory for compiled grammars. If grammars exist, this significantly improves speed.
- overwrite_cache - Updates grammars in cache.
- whitelist - TSV file with custom mappings of written text to spoken form.


### Acknowledge
1. We borrowed a lot of codes from [NeMo](https://github.com/NVIDIA/NeMo).
2. We refered the codes from [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) for Chinese inverse text normalization. 

### License

This project is licensed under the Apache-2.0 license. FunTextProcessing also contains various third-party components and some code modified from other repos under other open source licenses. 
