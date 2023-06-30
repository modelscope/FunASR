# Leaderboard IO


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


<table border="1">
    <tr align="center">
        <td style="border: 1px solid">Model</td>
        <td style="border: 1px solid">Offline/Online</td>
        <td colspan="2" style="border: 1px solid">Aishell1</td>
        <td colspan="4" style="border: 1px solid">Aishell2</td>
        <td colspan="3" style="border: 1px solid">WenetSpeech</td>
    </tr>
    <tr align="center">
        <td style="border: 1px solid"></td>
        <td style="border: 1px solid"></td>
        <td style="border: 1px solid">dev</td> 
        <td style="border: 1px solid">test</td>
        <td style="border: 1px solid">dev_ios</td>
        <td style="border: 1px solid">test_ios</td>
        <td style="border: 1px solid">test_android</td>
        <td style="border: 1px solid">test_mic</td>
        <td style="border: 1px solid">dev</td>
        <td style="border: 1px solid">test_meeting</td>
        <td style="border: 1px solid">test_net</td>
    </tr>
    <tr align="center">
        <td style="border: 1px solid"> <a href="https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary">Paraformer-large</a> </td>
        <td style="border: 1px solid">Offline</td>
        <td style="border: 1px solid">1.76</td>
        <td style="border: 1px solid">1.94</td>
        <td style="border: 1px solid">2.79</td>
        <td style="border: 1px solid">2.84</td>
        <td style="border: 1px solid">3.08</td>
        <td style="border: 1px solid">3.03</td>
        <td style="border: 1px solid">3.43</td>
        <td style="border: 1px solid">7.01</td>
        <td style="border: 1px solid">6.66</td>
    </tr>
    <tr align="center">
        <td style="border: 1px solid"> <a href="https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary">Paraformer-large-long</a> </td> 
        <td style="border: 1px solid">Offline</td>      
        <td style="border: 1px solid">1.80</td>
        <td style="border: 1px solid">2.10</td>
        <td style="border: 1px solid">2.78</td>
        <td style="border: 1px solid">2.87</td>
        <td style="border: 1px solid">3.12</td>
        <td style="border: 1px solid">3.11</td>
        <td style="border: 1px solid">3.44</td>
        <td style="border: 1px solid">13.28</td>
        <td style="border: 1px solid">7.08</td>
    </tr>
    <tr align="center">
        <td style="border: 1px solid"> <a href="https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary">Paraformer-large-contextual</a> </td>
        <td style="border: 1px solid">Offline</td>
        <td style="border: 1px solid">1.76</td>
        <td style="border: 1px solid">2.02</td>
        <td style="border: 1px solid">2.73</td>
        <td style="border: 1px solid">2.85</td>
        <td style="border: 1px solid">2.98</td>
        <td style="border: 1px solid">2.95</td>
        <td style="border: 1px solid">3.42</td>
        <td style="border: 1px solid">7.16</td>
        <td style="border: 1px solid">6.72</td>
    </tr>
    <tr align="center">
        <td style="border: 1px solid"> <a href="https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary">Paraformer</a> </td> 
        <td style="border: 1px solid">Offline</td>
        <td style="border: 1px solid">3.24</td>
        <td style="border: 1px solid">3.69</td>
        <td style="border: 1px solid">4.58</td>
        <td style="border: 1px solid">4.63</td>
        <td style="border: 1px solid">4.83</td>
        <td style="border: 1px solid">4.71</td>
        <td style="border: 1px solid">4.19</td>
        <td style="border: 1px solid">8.32</td>
        <td style="border: 1px solid">9.19</td>
    </tr>
   <tr align="center">
        <td style="border: 1px solid"> <a href="https://modelscope.cn/models/damo/speech_UniASR_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-online/summary">UniASR</a> </td> 
        <td style="border: 1px solid">Online</td>
        <td style="border: 1px solid">3.34</td>
        <td style="border: 1px solid">3.99</td>
        <td style="border: 1px solid">4.62</td>
        <td style="border: 1px solid">4.52</td>
        <td style="border: 1px solid">4.77</td>
        <td style="border: 1px solid">4.73</td>
        <td style="border: 1px solid">4.51</td>
        <td style="border: 1px solid">10.63</td>
        <td style="border: 1px solid">9.70</td>
    </tr>
   <tr align="center">
        <td style="border: 1px solid"> <a href="https://modelscope.cn/models/damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline/summary">UniASR-large</a> </td> 
        <td style="border: 1px solid">Offline</td>      
        <td style="border: 1px solid">2.93</td>
        <td style="border: 1px solid">3.48</td>
        <td style="border: 1px solid">3.95</td>
        <td style="border: 1px solid">3.87</td>
        <td style="border: 1px solid">4.11</td>
        <td style="border: 1px solid">4.11</td>
        <td style="border: 1px solid">4.16</td>
        <td style="border: 1px solid">10.09</td>
        <td style="border: 1px solid">8.69</td>
    </tr>
    <tr align="center">
        <td style="border: 1px solid"> <a href="https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-aishell1-pytorch/summary">Paraformer-aishell</a> </td>
        <td style="border: 1px solid">Offline</td>
        <td style="border: 1px solid">4.88</td>
        <td style="border: 1px solid">5.43</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
    </tr>
   <tr align="center">
        <td style="border: 1px solid"> <a href="https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary">ParaformerBert-aishell</a> </td>
        <td style="border: 1px solid">Offline</td>
        <td style="border: 1px solid">6.14</td>
        <td style="border: 1px solid">7.01</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
    </tr>
   <tr align="center">
        <td style="border: 1px solid"> <a href="https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary">Paraformer-aishell2</a> </td> 
        <td style="border: 1px solid">Offline</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">5.82</td>
        <td style="border: 1px solid">6.30</td>
        <td style="border: 1px solid">6.60</td>
        <td style="border: 1px solid">5.83</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
    </tr>
   <tr align="center">
        <td style="border: 1px solid"> <a href="https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary">ParaformerBert-aishell2</a> </td> 
        <td style="border: 1px solid">Offline</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">4.95</td>
        <td style="border: 1px solid">5.45</td>
        <td style="border: 1px solid">5.59</td>
        <td style="border: 1px solid">5.83</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
        <td style="border: 1px solid">-</td>
    </tr>
</table>


### English Dataset

