#!/usr/bin/env python3
"""Test Fun-ASR-Nano + VAD + SPK + PUNC: full pipeline with speaker diarization"""
import sys
import time

def main():
    from funasr import AutoModel

    print("[Fun-ASR-Nano-SPK] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        trust_remote_code=True,
        remote_code="./model.py",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
        punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        device="cpu",
        disable_update=True,
        hub="hf",
    )
    print("[Fun-ASR-Nano-SPK] Model loaded in %.1fs" % (time.time() - t0))

    wav_path = model.model_path + "/example/zh.mp3"

    print("[Fun-ASR-Nano-SPK] Running inference...")
    t0 = time.time()
    res = model.generate(input=[wav_path], cache={}, batch_size=1, language="中文")
    print("[Fun-ASR-Nano-SPK] Inference done in %.1fs" % (time.time() - t0))

    if not res or len(res) == 0:
        print("[Fun-ASR-Nano-SPK] FAILED - empty result")
        return 1

    result = res[0]
    keys = list(result.keys())
    print("[Fun-ASR-Nano-SPK] Result keys: %s" % keys)
    print("[Fun-ASR-Nano-SPK] Text: %s" % result.get("text", ""))

    # Verify timestamp
    ts = result.get("timestamp", None)
    if ts is None or len(ts) == 0:
        print("[Fun-ASR-Nano-SPK] FAILED - no timestamp")
        return 1
    print("[Fun-ASR-Nano-SPK] Timestamp count: %d, first: %s" % (len(ts), ts[0]))

    # Verify sentence_info with speaker labels
    si = result.get("sentence_info", None)
    if si is None or len(si) == 0:
        print("[Fun-ASR-Nano-SPK] FAILED - no sentence_info")
        return 1

    print("[Fun-ASR-Nano-SPK] sentence_info:")
    for s in si:
        print("  spk=%d | [%d-%d] %s" % (s["spk"], s["start"], s["end"], s["text"]))

    has_spk = all("spk" in s for s in si)
    if not has_spk:
        print("[Fun-ASR-Nano-SPK] FAILED - missing spk label in sentence_info")
        return 1

    print("[Fun-ASR-Nano-SPK] PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
