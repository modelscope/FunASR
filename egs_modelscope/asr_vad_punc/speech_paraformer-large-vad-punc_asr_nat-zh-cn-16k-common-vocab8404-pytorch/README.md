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

### Inference using local finetuned model

- Modify inference related parameters in `infer_after_finetune.py`
    - <strong>output_dir:</strong> # result dir
    - <strong>data_dir:</strong> # the dataset dir needs to include `test/wav.scp`. If `test/text` is also exists, CER will be computed
    - <strong>decoding_model_name:</strong> # set the checkpoint name for decoding, e.g., `valid.cer_ctc.ave.pb`

- Then you can run the pipeline to finetune with:
```python
    python infer_after_finetune.py
```

- Results

The decoding results can be found in `$output_dir/decoding_results/text.cer`, which includes recognition results of each sample and the CER metric of the whole test set.
