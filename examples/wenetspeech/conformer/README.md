
# Conformer Result

## Training Config
- Feature info: using 80 dims fbank, global cmvn, speed perturb(0.9, 1.0, 1.1), specaugment
- Train info: lr 5e-4, batch_size 25000, 2 gpu(Tesla V100), acc_grad 1, 50 epochs
- Train config: conf/train_asr_transformer.yaml
- LM config: LM was not used
- Model size: 46M

## Results (CER)

|   testset   | CER(%)  |
|:-----------:|:-------:|
|     dev     |  4.42   |
|    test     |  4.87   |