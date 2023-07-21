# Paraformer-Large
- Model link: <https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary>
- Model size: 220M

# Beachmark Results

## AISHELL-1
- Decode config: 
  - Decode without CTC
  - Decode without LM

| CER(%)    | Pretrain model|[Finetune model](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell1-vocab8404-pytorch/summary) |
|:---------:|:-------------:|:-------------:|
| dev       | 1.75          |1.62           |
| test      | 1.95          |1.78           |

## AISHELL-2
- Decode config: 
  - Decode without CTC
  - Decode without LM

| CER(%)       | Pretrain model|[Finetune model](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell2-vocab8404-pytorch/summary)|
|:------------:|:-------------:|:------------:|
| dev_ios      | 2.80          |2.60          |
| test_android | 3.13          |2.84          |
| test_ios     | 2.85          |2.82          |
| test_mic     | 3.06          |2.88          |

## Wenetspeech
- Decode config: 
  - Decode without CTC
  - Decode without LM

| testset   | CER(%)|
|:---------:|:-----:|
| dev       | 3.57  |
| test      | 6.97  |
| test_net  | 6.74  |

## SpeechIO TIOBE
- Decode config 1:
  - Decode without CTC
  - Decode without LM
  - With text norm
- Decode config 2:
  - Decode without CTC
  - Decode with Transformer-LM
  - LM weight: 0.15
  - With text norm

| testset | w/o LM | w/ LM |
|:------------------:|:----:|:----:|
|SPEECHIO_ASR_ZH00001| 0.49 | 0.35 |
|SPEECHIO_ASR_ZH00002| 3.23 | 2.86 |
|SPEECHIO_ASR_ZH00003| 1.13 | 0.80 |
|SPEECHIO_ASR_ZH00004| 1.33 | 1.10 |
|SPEECHIO_ASR_ZH00005| 1.41 | 1.18 |
|SPEECHIO_ASR_ZH00006| 5.25 | 4.85 |
|SPEECHIO_ASR_ZH00007| 5.51 | 4.97 |
|SPEECHIO_ASR_ZH00008| 3.69 | 3.18 |
|SPEECHIO_ASR_ZH00009| 3.02 | 2.78 |
|SPEECHIO_ASR_ZH000010| 3.35 | 2.99 |
|SPEECHIO_ASR_ZH000011| 1.54 | 1.25 |
|SPEECHIO_ASR_ZH000012| 2.06 | 1.68 |
|SPEECHIO_ASR_ZH000013| 2.57 | 2.25 |
|SPEECHIO_ASR_ZH000014| 3.86 | 3.08 |
|SPEECHIO_ASR_ZH000015| 3.34 | 2.67 |


# Fine-tuning Results

## Fine-tuning
- Train config: 
  - Training data: aishell-1
  - Training info: lr 0.0002, dataset_type: small, batch bins 2000, 2 gpu, acc_grad 1, 20 epochs
  - Decoding info: beam_size 1, average_num 10

| model    | dev cer(%) | test cer(%) |
|:---------:|:-------------:|:-------------:|
| Pretrain       | 1.75          |1.95           |
| Full-tuning      | 1.62          |1.78           |

- Train config: 
  - Training data: 16k sichuan dialect
  - Training info: lr 0.0002, dataset_type: small, batch bins 2000, 2 gpu, acc_grad 1, 20 epochs
  - Decoding info: beam_size 1, average_num 10
  
  
|   model  | Training Data(h) | common cer(%) | sichuan cer(%) |
|:--------:|:-------------:|:-------:|:------------:|
| Pretrain |               |   8.57  |     19.81    |
| Full-tuning |      50      |   8.8   |      12      |
|          |      100     |   9.24  |     11.63    |
|          |      200     |   9.82  |     10.47    |
|          |      300     |   9.95  |     10.44    |
|          |     1000     |   9.99  |     9.78     |


## Lora Fine-tuning
- Train config: 
  - Training data: 16k sichuan dialect
  - Training info: lr 0.0002, dataset_type: small, batch bins 2000, 2 gpu, acc_grad 1, 20 epochs
  - Lora info: lora_bias: "all", lora_list ['q','v'], lora_rank:8, lora_alpha:16, lora_dropout:0.1
  - Decoding info: beam_size 1, average_num 10
  
| model         | Training Data(h) | Trainable Parameters(M) | Memory Usage(G) | common cer(%) | sichuan cer(%) |
|:---------------:|:------------------:|:-------------------------:|:-----------------:|:---------------:|:----------------:|
| Pretrain      |                  |                         |                 | 8.57          | 19.81          |
| Full-tuning   | 50               | 220.9                   | 15              | 8.8           | 12             |
| Lora Finetune | 50               | 2.29                    | 7               | 9.13          | 12.13          |
| Full-tuning   | 200              | 220.9                   | 15              | 9.82          | 10.47          |
| Lora Finetune | 200              | 2.29                    | 7               | 9.21          | 11.28          |
