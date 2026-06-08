# Fine-tuning FSMN-VAD

This guide shows how to fine-tune the FSMN-VAD (Voice Activity Detection) model on your own data. Fine-tuning is recommended when the default model does not perform well on your specific audio domain (e.g., noisy environments, walkie-talkie audio, call center recordings).

## When to Fine-tune

- High miss rate (speech segments not detected)
- High false alarm rate (non-speech detected as speech)
- Domain-specific audio that differs significantly from the training data (e.g., 8kHz telephony, high-noise industrial environments)

If you only need minor adjustments, try [parameter tuning](#parameter-tuning-without-fine-tuning) first.

## Data Preparation

### Required files

1. **wav.scp** — Audio file list
2. **vad.txt** — VAD labels (speech segment timestamps)

### wav.scp format

```
utt001 /path/to/audio1.wav
utt002 /path/to/audio2.wav
utt003 /path/to/audio3.wav
```

### vad.txt format

Each line contains the utterance ID followed by a list of speech segments `[start_ms, end_ms]`:

```
utt001 [[320, 2150], [3800, 6420], [8100, 12500]]
utt002 [[0, 5000], [6200, 9800]]
utt003 [[100, 3000]]
```

Prepare separate `train_wav.scp` / `train_vad.txt` and `val_wav.scp` / `val_vad.txt` for training and validation.

### Recommended data size

- **Minimum**: 50-100 labeled audio files
- **Recommended**: 500+ for robust fine-tuning
- Include both positive (speech) and negative (silence/noise) examples representative of your target domain

## Fine-tuning

### Step 1: Place data

```
data/list/
├── train_wav.scp
├── train_vad.txt
├── val_wav.scp
└── val_vad.txt
```

### Step 2: Run fine-tuning

```bash
cd examples/industrial_data_pretraining/fsmn_vad_streaming
bash finetune.sh
```

Key parameters in `finetune.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name_or_model_dir` | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` | Base model (use `*-8k-*` for 8kHz audio) |
| `train_conf.max_epoch` | 20 | Training epochs |
| `optim_conf.lr` | 0.00005 | Learning rate (lower = more conservative) |
| `train_conf.validate_interval` | 1000 | Validation frequency (steps) |

### Step 3: Use the fine-tuned model

```python
from funasr import AutoModel

model = AutoModel(model="/path/to/outputs/model_dir")
result = model.generate(input="test.wav")
```

## Parameter Tuning (without fine-tuning)

For minor adjustments, pass VAD parameters directly:

```python
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={
        "max_end_silence_time": 500,     # Max silence within a sentence (ms)
        "speech_noise_thres": 0.5,       # Speech/noise threshold (lower = more sensitive)
        "max_single_segment_time": 60000, # Max segment duration (ms)
    }
)
```

### Key parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_end_silence_time` | 800 | Increase to tolerate longer pauses within sentences |
| `speech_noise_thres` | 0.6 | Decrease to detect more speech (may increase false alarms) |
| `max_single_segment_time` | 60000 | Max duration of a single speech segment |
| `speech_2_noise_ratio` | 1.0 | Decrease to be more sensitive to speech |
| `max_start_silence_time` | 3000 | Max leading silence before speech starts |
