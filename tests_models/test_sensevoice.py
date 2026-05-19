#!/usr/bin/env python3
"""Test SenseVoice: multi-task speech understanding"""
import sys
import time

def main():
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    print("[SenseVoice] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cpu",
        disable_update=True,
    )
    print("[SenseVoice] Model loaded in %.1fs" % (time.time()-t0))

    print("[SenseVoice] Running inference (Chinese)...")
    t0 = time.time()
    res = model.generate(
        input=model.model_path + "/example/zh.mp3",
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    print("[SenseVoice] Inference done in %.1fs" % (time.time()-t0))

    if res and len(res) > 0 and "text" in res[0]:
        text = rich_transcription_postprocess(res[0]["text"])
        print("[SenseVoice] Result: %s" % text)
        print("[SenseVoice] PASSED")
        return 0
    else:
        print("[SenseVoice] FAILED - no text in result")
        return 1

if __name__ == "__main__":
    sys.exit(main())
