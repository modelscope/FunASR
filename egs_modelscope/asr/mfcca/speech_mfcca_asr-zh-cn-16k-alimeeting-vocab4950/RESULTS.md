# Paraformer-Large
- Model link: <https://www.modelscope.cn/models/NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950/summary>
- Model size: 45M

# Environments
- date: `Tue Feb 13 20:13:22 CST 2023`
- python version: `3.7.12`
- FunASR version: `0.1.0`
- pytorch version: `pytorch 1.7.0`
- Git hash: ``
- Commit date: ``

# Beachmark Results

## result (paper)
beam=20，CER tool：https://github.com/yufan-aslp/AliMeeting 

|        model        | Para (M) | Data (hrs) | Eval (CER%) | Test (CER%) |
|:-------------------:|:---------:|:---------:|:---------:| :---------:|
| MFCCA | 45   |   917  |   16.1   | 17.5   |

## result（modelscope）

beam=10

with separating character (src)

|        model        | Para (M) | Data (hrs) | Eval_sp (CER%) | Test_sp (CER%) | 
|:-------------------:|:---------:|:---------:|:---------:| :---------:|
| MFCCA | 45   |   917  |   17.1   | 18.6   |

without separating character (src)

|        model        | Para (M) | Data (hrs) | Eval_nosp (CER%) | Test_nosp (CER%) | 
|:-------------------:|:---------:|:---------:|:---------:| :---------:|
| MFCCA | 45   |   917  |   16.4   | 18.0   |

## 偏差

Considering the differences of the CER calculation tool and decoding beam size, the results of CER are biased (<0.5%).