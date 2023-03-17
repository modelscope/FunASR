# Benchmark 

### Data set:
Aishell1 [test set](https://www.openslr.org/33/) , the total audio duration is 36108.919 seconds.

### Tools
- Install 
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR
pip install --editable ./
cd funasr/runtime/python/utils
pip install -r requirements.txt
```

- recipe
set the model, data path and output_dir

```shell
nohup bash test_rtf.sh &> log.txt &
```


## [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) 

 ### Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz   16core-32processor    with avx512_vnni


### Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz   16core-32processor    with avx512_vnni


### Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz    32core-64processor   without avx512_vnni


## [Paraformer](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)
