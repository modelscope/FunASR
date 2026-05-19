#!/usr/bin/env python3
"""Test Paraformer-Streaming: chunk-by-chunk streaming ASR"""
import sys
import time
import os

def main():
    import soundfile
    from funasr import AutoModel

    print("[Paraformer-Streaming] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        device="cpu",
        disable_update=True,
        disable_pbar=True,
    )
    print("[Paraformer-Streaming] Model loaded in %.1fs" % (time.time() - t0))

    wav_file = os.path.join(model.model_path, "example/asr_example.wav")
    speech, sample_rate = soundfile.read(wav_file)

    chunk_size = [0, 10, 5]
    encoder_chunk_look_back = 4
    decoder_chunk_look_back = 1
    chunk_stride = chunk_size[1] * 960  # 600ms

    print("[Paraformer-Streaming] Running streaming inference (%.2fs audio, %d chunks)..." % (
        len(speech) / sample_rate, int((len(speech) - 1) / chunk_stride + 1)))
    t0 = time.time()

    cache = {}
    total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)
    all_text = ""

    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
        is_final = i == total_chunk_num - 1
        res = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
        )
        txt = res[0].get("text", "") if res else ""
        all_text += txt

    print("[Paraformer-Streaming] Inference done in %.1fs" % (time.time() - t0))
    print("[Paraformer-Streaming] Result: '%s'" % all_text)

    expected = "欢迎大家来体验达摩院推出的语音识别模型"
    if expected in all_text:
        print("[Paraformer-Streaming] PASSED")
        return 0
    else:
        print("[Paraformer-Streaming] FAILED - expected text not found")
        return 1

if __name__ == "__main__":
    sys.exit(main())
