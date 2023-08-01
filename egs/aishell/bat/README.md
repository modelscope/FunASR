# Boundary Aware Transducer (BAT) Result

## Training Config
- 8 gpu(Tesla V100)
- Feature info: using 80 dims fbank, global cmvn, speed perturb(0.9, 1.0, 1.1), specaugment
- Train config: conf/train_conformer_bat.yaml
- LM config: LM was not used
- Model size: 90M

## Results (CER)
- Decode config: conf/decode_bat_conformer.yaml

|   testset   |  CER(%) |
|:-----------:|:-------:|
|     dev     |  4.56   |
|    test     |  4.97   |
