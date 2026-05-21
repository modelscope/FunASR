#!/usr/bin/env python3
"""Test SenseVoice + VAD + SPK + PUNC: speaker diarization with SenseVoice"""
import sys
import time

def main():
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    print("[SenseVoice-SPK] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
        device="cpu",
        disable_update=True,
    )
    print("[SenseVoice-SPK] Model loaded in %.1fs" % (time.time() - t0))

    wav_path = model.model_path + "/example/zh.mp3"

    print("[SenseVoice-SPK] Running inference...")
    t0 = time.time()
    res = model.generate(
        input=wav_path, cache={}, language="auto", use_itn=True,
        batch_size_s=60, merge_vad=True, merge_length_s=15,
    )
    print("[SenseVoice-SPK] Inference done in %.1fs" % (time.time() - t0))

    if not res or len(res) == 0:
        print("[SenseVoice-SPK] FAILED - empty result")
        return 1

    result = res[0]
    keys = list(result.keys())
    print("[SenseVoice-SPK] Result keys: %s" % keys)

    # Verify text
    text = rich_transcription_postprocess(result.get("text", ""))
    print("[SenseVoice-SPK] Text: %s" % text)
    if not text:
        print("[SenseVoice-SPK] FAILED - empty text")
        return 1

    # Verify timestamp exists
    ts = result.get("timestamp", None)
    if ts is None or len(ts) == 0:
        print("[SenseVoice-SPK] FAILED - no timestamp")
        return 1
    print("[SenseVoice-SPK] Timestamp count: %d" % len(ts))

    # Verify sentence_info with speaker labels
    si = result.get("sentence_info", None)
    if si is None or len(si) == 0:
        print("[SenseVoice-SPK] FAILED - no sentence_info")
        return 1

    print("[SenseVoice-SPK] sentence_info:")
    for s in si:
        print("  spk=%d | [%d-%d] %s" % (s["spk"], s["start"], s["end"], rich_transcription_postprocess(s.get("text", s.get("sentence", "")))))

    has_spk = all("spk" in s for s in si)
    if not has_spk:
        print("[SenseVoice-SPK] FAILED - missing spk label")
        return 1

    print("[SenseVoice-SPK] PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
