# ModelScope Model

## How to finetune and infer using a pretrained ModelScope Model

### Inference

Or you can use the finetuned model for inference directly.

- Setting parameters in `infer.py`
    - <strong>audio_in:</strong> # support wav, url, bytes, and parsed audio format.
    - <strong>output_dir:</strong> # If the input format is wav.scp, it needs to be set.

- Then you can run the pipeline to infer with:
```python
    python infer.py
```


Modify inference related parameters in vad.yaml.

- max_end_silence_time: The end-point silence duration  to judge the end of sentence, the parameter range is 500ms~6000ms, and the default value is 800ms
- speech_noise_thres:  The balance of speech and silence scores, the parameter range is (-1,1)
    - The value tends to -1, the greater probability of noise being judged as speech
    - The value tends to 1, the greater probability of speech being judged as noise
