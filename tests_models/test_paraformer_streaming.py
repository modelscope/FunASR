#!/usr/bin/env python3
"""Test Paraformer-Streaming: online/streaming ASR"""
import sys
import time

def main():
    from funasr import AutoModel

    print("[Paraformer-Streaming] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        device="cpu",
        disable_update=True,
    )
    print("[Paraformer-Streaming] Model loaded in %.1fs" % (time.time()-t0))

    print("[Paraformer-Streaming] Running inference...")
    t0 = time.time()
    test_wav = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
    res = model.generate(input=test_wav, cache={}, is_final=True)
    print("[Paraformer-Streaming] Inference done in %.1fs" % (time.time()-t0))
    print("[Paraformer-Streaming] Result: %s" % res)

    if res and len(res) > 0 and "text" in res[0]:
        print("[Paraformer-Streaming] PASSED")
        return 0
    else:
        print("[Paraformer-Streaming] FAILED - no text in result")
        return 1

if __name__ == "__main__":
    sys.exit(main())
