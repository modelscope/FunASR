# ModelScope Model

## How to finetune and infer using a pretrained UniASR Model

### Finetune

- Modify finetune training related parameters in `finetune.py`
    - <strong>output_dir:</strong> # result dir
    - <strong>data_dir:</strong> # the dataset dir needs to include files: `train/wav.scp`, `train/text`; `validation/wav.scp`, `validation/text`
    - <strong>dataset_type:</strong> # for dataset larger than 1000 hours, set as `large`, otherwise set as `small`
    - <strong>batch_bins:</strong> # batch size. For dataset_type is `small`, `batch_bins` indicates the feature frames. For dataset_type is `large`, `batch_bins` indicates the duration in ms
    - <strong>max_epoch:</strong> # number of training epoch
    - <strong>lr:</strong> # learning rate

- Then you can run the pipeline to finetune with:
```python
    python finetune.py
```

### Inference

Or you can use the finetuned model for inference directly.

- Setting parameters in `infer.py`
    - <strong>data_dir:</strong> # the dataset dir needs to include `test/wav.scp`. If `test/text` is also exists, CER will be computed
    - <strong>output_dir:</strong> # result dir
    - <strong>ngpu:</strong> # the number of GPUs for decoding
    - <strong>njob:</strong> # the number of jobs for each GPU

- Then you can run the pipeline to infer with:
```python
    python infer.py
```

- Results

The decoding results can be found in `$output_dir/1best_recog/text.cer`, which includes recognition results of each sample and the CER metric of the whole test set.

### Inference using local finetuned model

- Modify inference related parameters in `infer_after_finetune.py`
    - <strong>output_dir:</strong> # result dir
    - <strong>data_dir:</strong> # the dataset dir needs to include `test/wav.scp`. If `test/text` is also exists, CER will be computed
    - <strong>decoding_model_name:</strong> # set the checkpoint name for decoding, e.g., `valid.cer_ctc.ave.pth`

- Then you can run the pipeline to finetune with:
```python
    python infer_after_finetune.py
```

- Results

The decoding results can be found in `$output_dir/decoding_results/text.cer`, which includes recognition results of each sample and the CER metric of the whole test set.
