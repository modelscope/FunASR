
# Streaming RNN-T Result

## Training Config
- 8 gpu(Tesla V100)
- Feature info: using 80 dims fbank, global cmvn, speed perturb(0.9, 1.0, 1.1), specaugment
- Train config: conf/train_conformer_rnnt_unified.yaml
- chunk config: chunk size 16, full left chunk
- LM config: LM was not used
- Model size: 90M

## Results (CER)
- Decode config: conf/train_conformer_rnnt_unified.yaml

|   testset   | CER(%)  |
|:-----------:|:-------:|
|     dev     |  5.53   |
|    test     |  6.24   |
