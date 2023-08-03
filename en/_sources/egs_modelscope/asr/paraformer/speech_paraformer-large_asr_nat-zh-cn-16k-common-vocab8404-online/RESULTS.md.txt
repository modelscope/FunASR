# Paraformer-Large
- Model link: <https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary>
- Model size: 220M

# Beachmark Results

## AISHELL-1
- Decode config: 
  - Decode without CTC
  - Decode without LM

| CER(%)    | Pretrain model|
|:---------:|:-------------:|
| dev       | 2.37          |
| test      | 3.34          |

## AISHELL-2
- Decode config: 
  - Decode without CTC
  - Decode without LM

| CER(%)       | Pretrain model|
|:------------:|:-------------:|
| dev_ios      | 4.04          |
| test_android | 3.86          |
| test_ios     | 4.38          |
| test_mic     | 4.21          |

## Wenetspeech
- Decode config: 
  - Decode without CTC
  - Decode without LM

| testset   | CER(%)|
|:---------:|:-----:|
| dev       | 4.55  |
| test      | 10.64  |
| test_net  | 7.78  |
