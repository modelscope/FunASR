# Benchmark (ModeScope Pipeline)


## Configuration
### Data set:
[Aishell1](https://www.openslr.org/33/): dev, test

[Aishell2](https://www.aishelltech.com/aishell_2): dev_ios, test_ios, test_android, test_mic

[WenetSpeech](https://github.com/wenet-e2e/WenetSpeech): dev, test_meeting, test_net


### Tools
#### [Install Requirements](https://alibaba-damo-academy.github.io/FunASR/en/installation/installation.html#installation)

Install ModelScope and FunASR from pip
```shell
pip install -U modelscope funasr
# For the users in China, you could install with the command:
#pip install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

Or install FunASR from source code
```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install -e ./
# For the users in China, you could install with the command:
# pip install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple
```


#### Recipe


##### [Test CER](https://alibaba-damo-academy.github.io/FunASR/en/modelscope_pipeline/asr_pipeline.html#inference-with-multi-thread-cpus-or-multi-gpus)
set the `model`, `data_dir` and `output_dir` in `infer.sh`.
```shell
cd egs_modelscope/asr/TEMPLATE
bash infer.sh
```

## Benchmark CER


### Chinese Dataset


<table>
    <tr align="center">
        <td>Model</td>
        <td>Offline/Online</td>
        <td colspan="2">Aishell1</td>
        <td colspan="4">Aishell2</td>
        <td colspan="3">WenetSpeech</td>
    </tr>
    <tr align="center">
        <td></td>
        <td></td>
        <td>dev</td> 
        <td>test</td>
        <td>dev_ios</td>
        <td>test_ios</td>
        <td>test_android</td>
        <td>test_mic</td>
        <td>dev</td>
        <td>test_meeting</td>
        <td>test_net</td>
    </tr>
    <tr align="center">
        <td> <a href="https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary">Paraformer-large</a> </td>
        <td>Offline</td>
        <td>1.76</td>
        <td>1.94</td>
        <td>2.79</td>
        <td>2.84</td>
        <td>3.08</td>
        <td>3.03</td>
        <td>3.43</td>
        <td>7.01</td>
        <td>6.66</td>
    </tr>
    <tr align="center">
        <td> <a href="https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary">Paraformer-large-long</a> </td> 
        <td>Offline</td>      
        <td>1.80</td>
        <td>2.10</td>
        <td>2.78</td>
        <td>2.87</td>
        <td>3.12</td>
        <td>3.11</td>
        <td>3.44</td>
        <td>13.28</td>
        <td>7.08</td>
    </tr>
    <tr align="center">
        <td> <a href="https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary">Paraformer-large-contextual</a> </td>
        <td>Offline</td>
        <td>1.76</td>
        <td>2.02</td>
        <td>2.73</td>
        <td>2.85</td>
        <td>2.98</td>
        <td>2.95</td>
        <td>3.42</td>
        <td>7.16</td>
        <td>6.72</td>
    </tr>
    <tr align="center">
        <td> <a href="https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary">Paraformer</a> </td> 
        <td>Offline</td>
        <td>3.24</td>
        <td>3.69</td>
        <td>4.58</td>
        <td>4.63</td>
        <td>4.83</td>
        <td>4.71</td>
        <td>4.19</td>
        <td>8.32</td>
        <td>9.19</td>
    </tr>
   <tr align="center">
        <td> <a href="https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online/summary">UniASR</a> </td> 
        <td>Online</td>
        <td>3.34</td>
        <td>3.99</td>
        <td>4.62</td>
        <td>4.52</td>
        <td>4.77</td>
        <td>4.73</td>
        <td>4.51</td>
        <td>10.63</td>
        <td>9.70</td>
    </tr>
   <tr align="center">
        <td> <a href="https://modelscope.cn/models/damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline/summary">UniASR-large</a> </td> 
        <td>Offline</td>      
        <td>2.93</td>
        <td>3.48</td>
        <td>3.95</td>
        <td>3.87</td>
        <td>4.11</td>
        <td>4.11</td>
        <td>4.16</td>
        <td></td>
        <td></td>
    </tr>
    <tr align="center">
        <td> <a href="https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-aishell1-pytorch/summary">Paraformer-aishell</a> </td>
        <td>Offline</td>
        <td>4.88</td>
        <td>5.43</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
   <tr align="center">
        <td> <a href="https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary">ParaformerBert-aishell</a> </td>
        <td>Offline</td>
        <td>6.14</td>
        <td>7.01</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
   <tr align="center">
        <td> <a href="https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary">Paraformer-aishell2</a> </td> 
        <td>Offline</td>
        <td>-</td>
        <td>-</td>
        <td>5.82</td>
        <td>6.30</td>
        <td>6.60</td>
        <td>5.83</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
   <tr align="center">
        <td> <a href="https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary">ParaformerBert-aishell2</a> </td> 
        <td>Offline</td>
        <td>-</td>
        <td>-</td>
        <td>4.95</td>
        <td>5.45</td>
        <td>5.59</td>
        <td>5.83</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
</table>


### English Dataset

