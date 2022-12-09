# ModelScope Model

## How to finetune and infer using a pretrained ModelScope Model

### Finetune
- Modify finetune training related parameters in `conf/train_asr_uniasr_40e1_12d1_20e2_12d2_1280_320_lfr6.yaml`
- Setting parameters in `modelscope_common_finetune.sh`
    - <strong>dataset:</strong> the dataset dir needs to include files: train/wav.scp, train/text; optional dev/wav.scp, dev/text, test/wav.scp test/text
    - <strong>tag:</strong> exp tag
    - <strong>init_model_name:</strong> speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online # pre-trained model, download from modelscope during fine-tuning
- Then you can run the pipeline to finetune with our model download from modelscope:
```sh
    sh ./modelscope_common_finetune.sh
``` 

### Inference

Or you can use the finetuned model for inference directly.

- Setting parameters in `modelscope_common_infer.sh`
    - <strong>data_dir:</strong> # wav list, ${data_dir}/wav.scp
    - <strong>exp_dir:</strong> the result path
    - <strong>model_name:</strong> speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-online # pre-trained model, download from modelscope
- Then you can run the pipeline to infer with: 
```sh
    sh ./modelscope_common_infer.sh
```
