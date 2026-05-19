#!/usr/bin/env python3
"""Test FSMN-VAD Streaming: chunk-by-chunk voice activity detection"""
import sys
import time
import os

def main():
    import soundfile
    from funasr import AutoModel

    print("[FSMN-VAD-Streaming] Loading model...")
    t0 = time.time()
    model = AutoModel(
        model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        device="cpu",
        disable_update=True,
        disable_pbar=True,
    )
    print("[FSMN-VAD-Streaming] Model loaded in %.1fs" % (time.time() - t0))

    wav_file = os.path.join(model.model_path, "example/vad_example.wav")
    speech, sample_rate = soundfile.read(wav_file)
    chunk_size = 200  # ms
    chunk_stride = int(chunk_size * sample_rate / 1000)

    total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)
    print("[FSMN-VAD-Streaming] Audio: %.2fs, %d chunks of %dms" % (
        len(speech) / sample_rate, total_chunk_num, chunk_size))

    print("[FSMN-VAD-Streaming] Running streaming inference...")
    t0 = time.time()

    cache = {}
    all_events = []
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
        is_final = i == total_chunk_num - 1
        res = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
        )
        if res[0]["value"]:
            all_events.extend(res[0]["value"])

    print("[FSMN-VAD-Streaming] Inference done in %.1fs" % (time.time() - t0))

    # Parse streaming VAD events into complete segments
    # Streaming output: [beg, -1] = speech start, [-1, end] = speech end, [beg, end] = complete
    complete_segments = []
    pending_start = None
    for event in all_events:
        if event[0] >= 0 and event[1] == -1:
            pending_start = event[0]
        elif event[0] == -1 and event[1] >= 0:
            if pending_start is not None:
                complete_segments.append([pending_start, event[1]])
                pending_start = None
        elif event[0] >= 0 and event[1] >= 0:
            complete_segments.append(event)

    print("[FSMN-VAD-Streaming] Raw events: %d, Complete segments: %d" % (
        len(all_events), len(complete_segments)))
    print("[FSMN-VAD-Streaming] Segments: %s" % complete_segments)

    if not complete_segments:
        print("[FSMN-VAD-Streaming] FAILED - no complete segments")
        return 1

    # Verify segments have valid ranges
    for seg in complete_segments:
        if seg[1] <= seg[0]:
            print("[FSMN-VAD-Streaming] FAILED - invalid segment: %s" % seg)
            return 1

    # Verify consistency: run again with fresh cache
    cache2 = {}
    all_events2 = []
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride:(i + 1) * chunk_stride]
        is_final = i == total_chunk_num - 1
        res = model.generate(
            input=speech_chunk, cache=cache2, is_final=is_final, chunk_size=chunk_size,
        )
        if res[0]["value"]:
            all_events2.extend(res[0]["value"])

    if all_events != all_events2:
        print("[FSMN-VAD-Streaming] FAILED - inconsistent across sessions")
        return 1

    print("[FSMN-VAD-Streaming] Consistency: 2 sessions identical")
    print("[FSMN-VAD-Streaming] PASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
