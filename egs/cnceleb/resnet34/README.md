
# ResNet34 Result

## Training Config
- Feature info: using 80 dims fbank, no cmvn, speed perturb(0.9, 1.0, 1.1)
- Train info: lr 1e-4, batch_size 64, 1 gpu(Tesla V100), acc_grad 1, 300000 steps, clip_gradient_norm 3.0, weight_l2_regularizer 0.01
- Loss info: additive angular margin softmax, feature_scaling_factor=8, margin 0.25
- Model info: ResNet34, global statistics pooling, Dense
- Train config: conf/train_sv_resnet34.yaml
- Model size: 5.60 M parameters

## Results (EER & minDCF)
- Test set: Alimeeting-test, CN-Celeb-eval-speech

|       testset         | EER(%)  |  minDCF | Threshold |
|:---------------------:|:-------:|:-------:| :--------:| 
|    Alimeeting-test    |  1.45   | 0.0849  | 0.9666    |
|  CN-Celeb-eval-speech |  9.00   | 0.2936  | 0.9465    |