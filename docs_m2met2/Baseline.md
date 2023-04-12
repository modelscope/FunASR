# Baseline
## Overview
We provide an end-to-end sa-asr baseline conducted on [FunASR](https://github.com/alibaba-damo-academy/FunASR) as a receipe. The model architecture is shown in Figure 3. The SpeakerEncoder is initialized with a pre-trained [speaker verification model](https://modelscope.cn/models/damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch/summary) from [ModelScope](https://modelscope.cn/home). This speaker verification model is also be used to extract the speaker embedding in the speaker profile. 

![model archietecture](images/sa_asr_arch.png)

## Quick start
#TODO: fill with the README.md of the baseline

## Baseline results
The results of the baseline system are shown in Table 3. The speaker profile adopts the oracle speaker embedding during training. However, due to the lack of oracle speaker label during evaluation, the speaker profile provided by an additional spectral clustering is used. Meanwhile, the results of using the oracle speaker profile on Eval and Test Set are also provided to show the impact of speaker profile accuracy. 
![baseline result](images/baseline_result.png)