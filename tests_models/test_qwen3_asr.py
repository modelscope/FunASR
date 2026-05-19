#!/usr/bin/env python3
"""Test Qwen3-ASR: multi-language speech recognition via AutoModel"""
import sys
import time


def main():
    from funasr import AutoModel

    model_path = "Qwen/Qwen3-ASR-1.7B"
    url_zh = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
    url_en = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"

    print("[Qwen3-ASR] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="Qwen3ASR",
        model_conf={
            "model_path": model_path,
            "dtype": "bf16",
            "max_new_tokens": 256,
            "max_inference_batch_size": 32,
        },
        device="cuda:0",
        disable_update=True,
    )
    print("[Qwen3-ASR] Model loaded in %.1fs" % (time.time() - t0))

    # Test 1: Chinese audio with forced language
    print("[Qwen3-ASR] Test 1: Chinese inference...")
    t0 = time.time()
    res = model.generate(
        input=url_zh,
        language="Chinese",
    )
    print("[Qwen3-ASR] Inference done in %.1fs" % (time.time() - t0))

    if res and len(res) > 0 and "text" in res[0]:
        print("[Qwen3-ASR] Result (zh): %s" % res[0]["text"])
        print("[Qwen3-ASR] Test 1 PASSED")
    else:
        print("[Qwen3-ASR] Test 1 FAILED - no text in result")
        return 1

    # Test 2: English audio with forced language
    print("[Qwen3-ASR] Test 2: English inference...")
    t0 = time.time()
    res = model.generate(
        input=url_en,
        language="English",
    )
    print("[Qwen3-ASR] Inference done in %.1fs" % (time.time() - t0))

    if res and len(res) > 0 and "text" in res[0]:
        print("[Qwen3-ASR] Result (en): %s" % res[0]["text"])
        print("[Qwen3-ASR] Test 2 PASSED")
    else:
        print("[Qwen3-ASR] Test 2 FAILED - no text in result")
        return 1

    # Test 3: Auto language detection (no forced language)
    print("[Qwen3-ASR] Test 3: Auto language detection...")
    t0 = time.time()
    res = model.generate(
        input=url_zh,
    )
    print("[Qwen3-ASR] Inference done in %.1fs" % (time.time() - t0))

    if res and len(res) > 0 and "text" in res[0]:
        print("[Qwen3-ASR] Result (auto): %s" % res[0]["text"])
        if "language" in res[0]:
            print("[Qwen3-ASR] Detected language: %s" % res[0]["language"])
        print("[Qwen3-ASR] Test 3 PASSED")
    else:
        print("[Qwen3-ASR] Test 3 FAILED - no text in result")
        return 1

    print("[Qwen3-ASR] All tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
