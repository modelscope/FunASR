# Finetune

「[简体中文](finetune_zh.md)」|「English」

## Requirements

```
pip install "funasr>=1.3.26"
```

## Data Prepare

Data examples

```
head -n1 data/train_example.jsonl | jq

{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "语音转写：<|startofspeech|>!https://modelscope.cn/datasets/FunAudioLLM/funasr-demo/resolve/master/audios/IT0011W0002.wav<|endofspeech|>"
    },
    {
      "role": "assistant",
      "content": "几点了？"
    }
  ],
  "speech_length": 145,
  "text_length": 3
}
```

Full ref to `data/train_example.jsonl`

Description：

- The content of systemis fixed as `You are a helpful assistant.`
- The content of userincludes the prompt and the path to the audio file (enclosed between `<|startofspeech|>!`and `<|endofspeech|>`).
  - The default prompts are `语音转写：`and `Speech transcription: `.
  - For corresponding languages, prompts can be combined, such as `语音转写成英文：`and `Transcribe speech into Chinese: `.
  - When the text annotation corresponding to the audio file contains no Arabic numerals or punctuation marks, you can use `语音转写，不进行文本规整：`and `Speech transcription without text normalization: `.
- The content of assistant corresponds to the text annotation of the audio file.
- speech_length: The number of fbank frames of the audio file (10ms per frame).
- text_length: The number of tokens in the annotation text of the audio file (encoded using `Qwen/Qwen3-0.6B`).

- `messages[2]["content"]`: transcription
- `speech_length`: number of fbank frames of the audio file
- `text_length`: number of tokens of the transcription (tokenized by `Qwen3-0.6B`)

We provide a data format conversion tool `scp2jsonl.py`, which can convert common speech recognition training data formats such as wav scp and transcription into the ChatML format.

`train_wav.scp`

```
BAC009S0764W0121 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav
BAC009S0916W0489 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0916W0489.wav
```

`train_text.txt`

```
BAC009S0764W0121 甚至出现交易几乎停滞的情况
BAC009S0916W0489 湖北一公司以员工名义贷款数十员工负债千万
```

```
python tools/scp2jsonl.py \
  ++scp_file=data/train_wav.scp \
  ++transcript_file=data/train_text.txt \
  ++jsonl_file=data/train_example.jsonl
```

## Finetune

Modify the `audio_encoder_conf.freeze`, `audio_adaptor_conf.freeze`, and `llm_conf.freeze` in `finetune.sh`.

Set the `freeze` parameter of the modules to be fine-tuned to false(by default, only the LLM is fine-tuned).

For more detailed parameters, refer to: [SenseVoice Model Training and Testing](https://github.com/modelscope/FunASR/blob/main/docs/tutorial/README.md#Model%20Training%20and%20Testing)

```
bash finetune.sh
```

### Recommended Configuration

- For training data less than 1000 hours, it is recommended to fine-tune the audio_adaptor.
- For training data less than 5000 hours, it is recommended to fine-tune the audio_encoder and audio_adaptor.
- For training data greater than 10000 hours, it is recommended to perform full-parameter fine-tuning.

## LoRA Finetune

As an alternative to full-parameter / partial fine-tuning, you can LoRA-tune the
Qwen3-0.6B LLM (attaching adapters to its `q_proj` / `v_proj` linear layers):

```
bash lora_finetune.sh
```

What it sets:

- `llm_conf.use_lora=true` — injects `LoRALinear` adapters into the target layers
  of the LLM (base weights shared and frozen, trainable `lora_A` / `lora_B` added).
- `lora_only=true` — freezes every non-LoRA parameter (audio encoder, adaptor and
  CTC decoder), so only the adapters train. Checkpoints still save the full state
  dict (base weights unchanged + adapter params), so you can resume or decode with
  the same `use_lora=true` config.
- `llm_conf.freeze=true` — keeps the LLM base weights frozen (redundant with
  `lora_only`, but explicit).

LoRA hyper-parameters live under `llm_conf.lora_conf` (defaults ship with the
model config): `r`, `lora_alpha`, `lora_dropout`, `target_modules`. Override them
on the command line, e.g.:

```
++llm_conf.lora_conf.r=32
```

To keep the audio encoder / adaptor **trainable** while LoRA-tweaking only the LLM
(a good middle ground for a few hundred hours of domain data), set
`lora_only=false`, `audio_encoder_conf.freeze=false` and
`audio_adaptor_conf.freeze=false`; the LLM base weights stay frozen via
`llm_conf.freeze=true`.

**Decoding / deployment:** the fine-tuned checkpoint works with `decode.py` below
without any merge step — the forward pass adds the adapter output on top of the
base weights. To fold the adapters into the base weights for a standalone
deployment checkpoint, compute `W' = W + (lora_alpha / r) * lora_B @ lora_A` for
each target module and drop the `lora_A` / `lora_B` keys.

## Model Evaluation

After model fine-tuning is completed, you can decode the model using the decode.py script:

```
python decode.py \
  ++model_dir=/path/to/finetuned \
  ++scp_file=data/val_wav.scp \
  ++output_file=output.txt
```

After decoding is completed, text inverse normalization needs to be applied to the annotations and recognition results, and then the WER should be calculated:

```
python tools/whisper_mix_normalize.py data/val_text.txt data/val_norm.txt
python tools/whisper_mix_normalize.py output.txt output_norm.txt
compute-wer data/val_norm.txt output_norm.txt cer.txt
tail -n8 cer.txt
```
