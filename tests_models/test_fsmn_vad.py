#!/usr/bin/env python3
"""Test FSMN-VAD: Voice Activity Detection"""
import sys
import time

def main():
    from funasr import AutoModel

    print("[FSMN-VAD] Loading model...")
    t0 = time.time()
    model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", device="cpu", disable_update=True)
    print("[FSMN-VAD] Model loaded in %.1fs" % (time.time()-t0))

    print("[FSMN-VAD] Running inference...")
    t0 = time.time()
    res = model.generate(
        input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    )
    print("[FSMN-VAD] Inference done in %.1fs" % (time.time()-t0))
    print("[FSMN-VAD] Result: %s" % res)

    if res and len(res) > 0 and "value" in res[0]:
        print("[FSMN-VAD] Detected %d speech segments" % len(res[0]["value"]))
        print("[FSMN-VAD] PASSED")
        return 0
    else:
        print("[FSMN-VAD] FAILED - no VAD segments detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
