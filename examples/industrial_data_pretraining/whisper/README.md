# Whisper Fine-tuning with FunASR

Fine-tune OpenAI Whisper models on your own data using FunASR's training framework.

## Supported Models

- whisper-tiny / whisper-tiny.en
- whisper-base / whisper-base.en
- whisper-small / whisper-small.en
- whisper-medium / whisper-medium.en
- whisper-large-v1 / whisper-large-v2 / whisper-large-v3 / whisper-large-v3-turbo

## Data Preparation

Prepare data in JSONL format:

```json
{"key": "utt001", "source": "/path/to/audio1.wav", "target": "the transcription text"}
{"key": "utt002", "source": "/path/to/audio2.wav", "target": "another transcription"}
```

## Fine-tuning

```bash
bash finetune.sh
```

Or customize directly:

```python
from funasr import AutoModel

model = AutoModel(model="Whisper-large-v3", model_conf={"hub": "openai"})

# Training uses the forward() method which computes cross-entropy loss
# on (mel-spectrogram, token_ids) pairs
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| model | Whisper-large-v3 | Model size |
| lr | 1e-5 | Learning rate (lower for larger models) |
| max_epoch | 10 | Training epochs |
| batch_size | 4 | Per-GPU batch size |
| warmup_steps | 500 | LR warmup |

## Tips

- For Chinese fine-tuning, use `whisper-large-v3` (best multilingual base)
- Freeze encoder for faster training: add `++train_conf.freeze_param="model.encoder"`
- Use smaller learning rates (1e-5 ~ 5e-6) to avoid catastrophic forgetting
- Recommended: 100+ hours of target-domain audio for meaningful improvement

## After Fine-tuning

```python
from funasr import AutoModel

# Load fine-tuned model
model = AutoModel(model="/path/to/exp/whisper_finetune")
result = model.generate(input="test.wav")
print(result[0]["text"])
```
