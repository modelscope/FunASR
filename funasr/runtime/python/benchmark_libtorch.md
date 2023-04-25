# CPU Benchmark (Libtorch)

## Configuration
### Data set:
Aishell1 [test set](https://www.openslr.org/33/) , the total audio duration is 36108.919 seconds.

### Tools
#### Install Requirements
Install ModelScope and FunASR
```shell
pip install -U modelscope funasr
# For the users in China, you could install with the command:
#pip install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

Install requirements
```shell
git clone https://github.com/alibaba-damo-academy/FunASR.git && cd FunASR
cd funasr/runtime/python/utils
pip install -r requirements.txt
```

#### Recipe

##### test_rtf
set the model, data path and output_dir
```shell
nohup bash test_rtf.sh &> log.txt &
```

##### test_cer
set the model, data path and output_dir
```shell
nohup bash test_cer.sh &> log.txt &
```


## [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) 


### Intel(R) Xeon(R) Platinum 8269CY CPU @ 2.50GHz   16core-32processor    with avx512_vnni

| concurrent-tasks | processing time(s) |  RTF   | Speedup Rate |
|:----------------:|:------------------:|:------:|:------------:|
| 1 (torch fp32)   |        3522        | 0.0976 |     10.3     |
|  1 (torch int8)  |        1746        | 0.0484 |     20.7     |
| 32 (torch fp32)  |        236         | 0.0066 |    152.7     |
| 32 (torch int8)  |        114         | 0.0032 |    317.4     |
| 64 (torch fp32)  |        235         | 0.0065 |    153.7     |
| 64 (torch int8)  |        113         | 0.0031 |    319.2     |


[//]: # (### Intel&#40;R&#41; Xeon&#40;R&#41; Platinum 8163 CPU @ 2.50GHz    32core-64processor   without avx512_vnni)


## [Paraformer](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary)
