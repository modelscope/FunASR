# ModelScope Model

## How to infer using a pretrained Paraformer-large Model

### Inference

You can use the pretrain model for inference directly.

- Setting parameters in `infer.py`
    - <strong>audio_in:</strong> # Support wav, url, bytes, and parsed audio format.
    - <strong>output_dir:</strong> # If the input format is wav.scp, it needs to be set.
    - <strong>batch_size:</strong> # Set batch size in inference.
    - <strong>param_dict:</strong> # Set the hotword list in inference.

- Then you can run the pipeline to infer with:
```python
    python infer.py
```

