#!/usr/bin/env python3
"""Test Paraformer-large: offline ASR (Chinese)"""
import sys
import time

def main():
    from funasr import AutoModel

    print("[Paraformer] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        vad_kwargs={"max_single_segment_time": 60000},
        punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
        device="cpu",
        disable_update=True,
    )
    print("[Paraformer] Model loaded in %.1fs" % (time.time()-t0))

    print("[Paraformer] Running inference...")
    t0 = time.time()
    res = model.generate(
        input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
        cache={},
    )
    print("[Paraformer] Inference done in %.1fs" % (time.time()-t0))
    print("[Paraformer] Result: %s" % res)

    if res and len(res) > 0 and "text" in res[0]:
        print("[Paraformer] PASSED")
        return 0
    else:
        print("[Paraformer] FAILED - no text in result")
        return 1

if __name__ == "__main__":
    sys.exit(main())
