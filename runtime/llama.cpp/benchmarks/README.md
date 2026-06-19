# Reproducing the FunASR-vs-whisper.cpp benchmark

Scripts and method behind [`../BENCHMARKS.md`](../BENCHMARKS.md).

## Metric (authoritative FunASR口径)
- **micro-CER** = `Σ edit_distance / Σ reference_chars` over all files (not a per-file mean).
- **normalize_zh**: `re.sub(r'[^\w一-鿿]', '', text).upper()` — drop punctuation/whitespace,
  keep word chars + CJK, upper-case. (SenseVoice meta tags `<|...|>` are stripped first.)
- **RTF** = `Σ compute_time / Σ audio_duration`, model-load time excluded.

`compute_cer.py` implements exactly this:
```bash
python compute_cer.py --refs testset.json --hyp_dir <hyps>/ [--time_file <times>.txt]
```
`testset.json` is a list of `{"id"/"key", "ref", "duration"}`; `<hyps>/{key}.txt` are
the transcripts; `<times>.txt` has `key compute_seconds` per line.

## Producing hypotheses

FunASR (this runtime), per clip:
```bash
# SenseVoice / Paraformer: ids -> detok
build/bin/llama-funasr-sensevoice -m sensevoice-small.gguf -a $k.wav > $k.ids
python ../sensevoice/detok.py <model>/chn_jpn_yue_eng_ko_spectok.bpe.model $k.ids > $k.txt
build/bin/llama-funasr-paraformer -m paraformer.gguf -a $k.wav > $k.ids
python ../paraformer/detok_paraformer.py <model>/tokens.json $k.ids > $k.txt
# Fun-ASR-Nano: text directly
build/bin/llama-funasr-cli --enc funasr-encoder.gguf -m qwen3-0.6b-q8_0.gguf -a $k.wav --chunk 15 > $k.txt
```
Compute time is on each tool's stderr (`encode … s` / `enc … dec … s`).

whisper.cpp, per clip (forced Chinese, no timestamps):
```bash
whisper-cli -m models/ggml-<size>.bin -l zh -nt -t 8 $k.wav > $k.txt   # 2>stderr has "total time"/"load time"
```
RTF compute time = `(total − load) ms`.

## Notes
- Run all systems with the **same thread count** (here `-t 8` / 8 threads) for a fair RTF.
- Whisper does its own internal 30 s windowing; the FunASR segmentation口径 is documented
  in `BENCHMARKS.md` (see the methodology/caveats sections).
