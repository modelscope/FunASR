# Paraformer
pretrained model in [ModelScope](https://www.modelscope.cn/home)：[speech_paraformer_asr_nat-aishell1-pytorch](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-aishell1-pytorch/summary)

## Training Config
- Feature info: using 80 dims fbank, global cmvn, speed perturb(0.9, 1.0, 1.1), specaugment
- Train info: lr 5e-4, batch_size 25000, 2 gpu(Tesla V100), acc_grad 1, 50 epochs
- Train config: conf/train_asr_paraformer_conformer_12e_6d_2048_256.yaml
- LM config: LM was not used

## Results (CER)

- Decode config: conf/decode_asr_transformer_noctc_1best.yaml (ctc weight:0.0)

|   testset   | CER(%)  |
|:-----------:|:-------:|
|     dev     |  4.66   |
|    test     |  5.11   |

- Decode config: conf/decode_asr_transformer.yaml (ctc weight:0.5)

|   testset   | CER(%)  |
|:-----------:|:-------:|
|     dev     |  4.52   |
|    test     |  4.94   |