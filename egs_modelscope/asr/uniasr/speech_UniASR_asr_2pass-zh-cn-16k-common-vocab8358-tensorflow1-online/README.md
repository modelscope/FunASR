# ModelScope Model

## How to finetune and infer using a pretrained Paraformer-large Model

### Finetune

- Modify finetune training related parameters in `finetune.py`
    - <strong>output_dir:</strong> # result dir
    - <strong>data_dir:</strong> # the dataset dir needs to include files: train/wav.scp, train/text; validation/wav.scp, validation/text.
    - <strong>batch_bins:</strong> # batch size
    - <strong>max_epoch:</strong> # number of training epoch
    - <strong>lr:</strong> # learning rate

- Then you can run the pipeline to finetune with:
```python
    python finetune.py
```

### Inference

Or you can use the finetuned model for inference directly.

- Setting parameters in `infer.py`
    - <strong>audio_in:</strong> # support wav, url, bytes, and parsed audio format.
    - <strong>output_dir:</strong> # If the input format is wav.scp, it needs to be set.

- Then you can run the pipeline to infer with:
```python
    python infer.py
```
