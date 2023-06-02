
# Conformer Transducer Result

## Training Config
- Feature info: using 80 dims fbank, global cmvn, speed perturb(0.9, 1.0, 1.1), specaugment
- Train config: conf/train_conformer_rnnt.yaml
- LM config: LM was not used
- Model size: 30.54M

## Results (CER)
- Decode config: conf/decode_rnnt_transformer.yaml

|      testset   | WER(%)  |
|:--------------:|:-------:|
|    test_clean  |  6.64   |
|    test_other  |  17.12  |
