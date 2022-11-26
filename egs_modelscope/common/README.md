# ModelScope Model

## How to finetune and infer using a pretrained ModelScope Model

### Finetune
- Modify finetune training related parameters in `conf/train_asr_paraformer_sanm_50e_16d_2048_512_lfr6.yaml`
- Setting parameters in `modelscope_common_finetune.sh`
    - <strong>dataset:</strong> the dataset dir needs to include files: train/wav.scp, train/text; optional dev/wav.scp, dev/text, test/wav.scp test/text
    - <strong>tag:</strong> exp tag
    - <strong>init_model_name:</strong> speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch # pre-trained model, download from modelscope during fine-tuning
- Then you can run the pipeline to finetune with our model download from modelscope:
```sh
    sh ./modelscope_common_finetune.sh
``` 

### Inference

Or you can use the finetuned model for inference directly.

- Setting parameters in `modelscope_common_infer.sh`
    - <strong>data_dir:</strong> # wav list, ${data_dir}/wav.scp
    - <strong>exp_dir:</strong> the result path
    - <strong>model_name:</strong> speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch # pre-trained model, download from modelscope
- Then you can run the pipeline to infer with: 
```sh
    sh ./modelscope_common_infer.sh
```
