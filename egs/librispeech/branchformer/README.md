# Branchformer Result

## Training Config
- Feature info: using raw speech, extracting 80 dims fbank online, global cmvn, speed perturb(0.9, 1.0, 1.1), specaugment
- Train config: conf/train_asr_branchformer.yaml
- LM config: LM was not used

## Results (CER)
- Decode config: conf/decode_asr_transformer_beam10_ctc0.3.yaml (ctc weight:0.3)

|   testset   | CER(%)  |
|:-----------:|:-------:|
|    dev_clean     |  2.12   |
|    dev_other     |  5.25   |
|    test_clean     |  2.34   |
|    test_other     |  5.39   |