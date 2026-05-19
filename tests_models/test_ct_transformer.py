#!/usr/bin/env python3
"""Test CT-Transformer: Punctuation Restoration"""
import sys
import time

def main():
    from funasr import AutoModel

    print("[CT-Transformer] Loading model...")
    t0 = time.time()
    model = AutoModel(model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", device="cpu", disable_update=True)
    print("[CT-Transformer] Model loaded in %.1fs" % (time.time()-t0))

    print("[CT-Transformer] Running inference...")
    t0 = time.time()
    res = model.generate(input="那今天的天气呢也是蛮好的啊你觉得怎么样呢我觉得还不错")
    print("[CT-Transformer] Inference done in %.1fs" % (time.time()-t0))
    print("[CT-Transformer] Result: %s" % res)

    if res and len(res) > 0 and "text" in res[0]:
        print("[CT-Transformer] PASSED")
        return 0
    else:
        print("[CT-Transformer] FAILED - no punctuation result")
        return 1

if __name__ == "__main__":
    sys.exit(main())
