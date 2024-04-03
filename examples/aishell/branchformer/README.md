# Branchformer Result

## Training Config
- Feature info: using raw speech, extracting 80 dims fbank online, global cmvn, speed perturb(0.9, 1.0, 1.1), specaugment
- Train info: lr 0.001, batch_size 10000, 4 gpu(Tesla V100), acc_grad 1, 180 epochs
- Train config: conf/train_asr_branchformer.yaml
- LM config: LM was not used

## Results (CER)

|   testset   | CER(%)  |
|:-----------:|:-------:|
|     dev     |  4.15   |
|    test     |  4.51   |