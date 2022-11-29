# ParaformerBert + specaug + speed perturbation + specaugmentation
## Environments
- date: `Mon Nov 21 13:25:30 CST 2022`
- python version: `3.7.12`
- FunASR version: `0.1.0`
- pytorch version: `pytorch 1.7.0`

## Config files
- train config: conf/train_asr_paraformerbert_conformer_12e_6d_2048_256.yaml
- model size: 46M
- lm config: LM was not used
- decode config: conf/decode_asr_transformer_noctc_1best.yaml (CTC was not used)

## Results (CER)
|   testset   | CER(%)  |
|:-----------:|:-------:|
|     dev     |  4.30   |
|    test     |  4.80   |