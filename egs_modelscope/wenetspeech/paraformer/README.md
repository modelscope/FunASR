# ModelScope: Paraformer-large Model

## Highlight

### ModelScope: Paraformer-Large Model
- <strong>Fast</strong>: Non-autoregressive (NAR) model, the Paraformer can achieve comparable performance to the state-of-the-art AR transformer, with more than 10x speedup.
- <strong>Accurate</strong>: SOTA in a lot of public ASR tasks, with a very significant relative improvement, capable of industrial implementation.
- <strong>Convenient</strong>: Quickly and easily download Paraformer-large from Modelscope for finetuning and inference.
    - Support finetuning and inference on AISHELL-1 and AISHELL-2.
    - Support inference on AISHELL-1, AISHELL-2, Wenetspeech, SpeechIO and other audio.

## How to infer using a pretrained ModelScope Paraformer-large Model

### Inference
- Setting parameters in `paraformer_large_infer.sh`
    - <strong>ori_data:</strong> please set the wenetspeech raw data path
    - <strong>data_dir:</strong> data output dictionary
    - <strong>exp_dir:</strong> the result path
    - <strong>model_name:</strong> speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch # base model, download from modelscope
    - <strong>test_sets:</strong> please set the testsets name
- Then you can run the pipeline to infer with: 
```sh
    sh ./paraformer_large_infer.sh
```
