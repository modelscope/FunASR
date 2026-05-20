#!/usr/bin/env python3
"""Test GLM-ASR: robust multi-language speech recognition"""
import sys
import time


def main():
    from funasr import AutoModel

    url_zh = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
    url_en = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav"

    # Standard FunASR usage
    # Default hub="ms" (ModelScope), use hub="hf" for HuggingFace
    print("[GLM-ASR] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="zai-org/GLM-ASR-Nano-2512",
        hub="hf",
        device="cuda:0",
        disable_update=True,
    )
    print("[GLM-ASR] Model loaded in %.1fs" % (time.time() - t0))

    # Test 1: Chinese audio
    print("[GLM-ASR] Test 1: Chinese inference...")
    t0 = time.time()
    res = model.generate(input=url_zh)
    print("[GLM-ASR] Inference done in %.1fs" % (time.time() - t0))

    if res and len(res) > 0 and "text" in res[0]:
        print("[GLM-ASR] Result (zh): %s" % res[0]["text"])
        print("[GLM-ASR] Test 1 PASSED")
    else:
        print("[GLM-ASR] Test 1 FAILED - no text in result")
        return 1

    # Test 2: English audio
    print("[GLM-ASR] Test 2: English inference...")
    t0 = time.time()
    res = model.generate(input=url_en)
    print("[GLM-ASR] Inference done in %.1fs" % (time.time() - t0))

    if res and len(res) > 0 and "text" in res[0]:
        print("[GLM-ASR] Result (en): %s" % res[0]["text"])
        print("[GLM-ASR] Test 2 PASSED")
    else:
        print("[GLM-ASR] Test 2 FAILED - no text in result")
        return 1

    print("[GLM-ASR] All tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
