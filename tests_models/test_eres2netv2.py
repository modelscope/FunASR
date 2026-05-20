#!/usr/bin/env python3
"""Test ERes2NetV2: speaker verification/diarization with ASR"""
import sys
import time


def main():
    from funasr import AutoModel

    url_zh = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"

    # Test 1: ERes2NetV2 as spk_model combined with VAD + ASR
    print("[ERes2NetV2] Loading model with VAD + ASR + SPK...")
    t0 = time.time()
    model = AutoModel(
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 60000},
        spk_model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
        device="cuda:0",
        disable_update=True,
    )
    print("[ERes2NetV2] Model loaded in %.1fs" % (time.time() - t0))

    # Test with speaker diarization
    print("[ERes2NetV2] Running inference with speaker diarization...")
    t0 = time.time()
    res = model.generate(
        input=url_zh,
        batch_size_s=300,
    )
    print("[ERes2NetV2] Inference done in %.1fs" % (time.time() - t0))

    if res and len(res) > 0 and "text" in res[0]:
        print("[ERes2NetV2] Result: %s" % res[0]["text"])
        if "spk" in res[0]:
            print("[ERes2NetV2] Speaker: %s" % res[0]["spk"])
        print("[ERes2NetV2] Test 1 PASSED")
    else:
        print("[ERes2NetV2] Test 1 FAILED - no text in result")
        return 1

    # Test 2: ERes2NetV2 standalone speaker embedding
    print("[ERes2NetV2] Test 2: Standalone speaker embedding...")
    t0 = time.time()
    spk_model = AutoModel(
        model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
        device="cuda:0",
        disable_update=True,
    )
    res = spk_model.generate(input=url_zh)
    print("[ERes2NetV2] Embedding done in %.1fs" % (time.time() - t0))

    if res and len(res) > 0 and "spk_embedding" in res[0]:
        emb = res[0]["spk_embedding"]
        print("[ERes2NetV2] Embedding shape: %s" % str(emb.shape))
        print("[ERes2NetV2] Test 2 PASSED")
    else:
        print("[ERes2NetV2] Test 2 FAILED - no spk_embedding in result")
        return 1

    print("[ERes2NetV2] All tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
