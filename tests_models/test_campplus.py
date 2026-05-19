#!/usr/bin/env python3
"""Test CAM++: Speaker Verification"""
import sys
import time

def main():
    from funasr import AutoModel

    print("[CAM++] Loading model...")
    t0 = time.time()
    model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", device="cpu", disable_update=True)
    print("[CAM++] Model loaded in %.1fs" % (time.time()-t0))

    print("[CAM++] Extracting speaker embedding...")
    t0 = time.time()
    res = model.generate(
        input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    )
    print("[CAM++] Inference done in %.1fs" % (time.time()-t0))

    if res and len(res) > 0:
        spk_embedding = res[0].get("spk_embedding", None)
        if spk_embedding is not None:
            print("[CAM++] Embedding shape: %s" % str(spk_embedding.shape))
            print("[CAM++] PASSED")
            return 0
    print("[CAM++] FAILED - no speaker embedding")
    return 1

if __name__ == "__main__":
    sys.exit(main())
