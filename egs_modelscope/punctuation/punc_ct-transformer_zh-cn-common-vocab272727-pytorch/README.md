# ModelScope Model

## How to finetune and infer using a pretrained ModelScope Model

### Inference

Or you can use the finetuned model for inference directly.

task=Tasks.punctuation,
    model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',

- Setting parameters in `modelscope_common_infer.sh`
    - <strong>model:</strong> damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch  # pre-trained model, download from modelscope
    - <strong>text_in:</strong> input path, text or url
    - <strong>output_dir:</strong> the result dir
- Then you can run the pipeline to infer with: 
```sh
    python ./infer.py
```
