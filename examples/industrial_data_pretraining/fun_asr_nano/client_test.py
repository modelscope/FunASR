#!/usr/bin/env python3
"""Fun-ASR-Nano Streaming ASR - Automated Test Script.

Tests the WebSocket server with audio files and validates the response format.

Usage:
    python client_test.py --server ws://localhost:10095 --file test_audio.wav
    python client_test.py --server ws://localhost:10095 --file test_audio.wav --hotwords "热词1,热词2"
"""

import asyncio
import argparse
import json
import time
import sys
import numpy as np

try:
    import websockets
except ImportError:
    print("ERROR: pip install websockets")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    print("ERROR: pip install soundfile")
    sys.exit(1)


SAMPLE_RATE = 16000


def load_audio(file_path):
    audio, sr = sf.read(file_path)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            print(f"ERROR: Audio is {sr}Hz. Install librosa for resampling.")
            sys.exit(1)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32)


async def test_basic(server, audio, hotwords=None):
    """Test basic streaming ASR flow."""
    print("=" * 60)
    print("TEST: Basic Streaming ASR")
    print("=" * 60)

    t_start = time.perf_counter()

    async with websockets.connect(server, ping_interval=None) as ws:
        await ws.send("START")
        resp = json.loads(await ws.recv())
        assert resp["event"] == "started", f"Expected 'started', got: {resp}"
        print("  [PASS] START -> event:started")

        if hotwords:
            await ws.send(f"HOTWORDS:{hotwords}")
            resp = json.loads(await ws.recv())
            assert resp["event"] == "hotwords_set", f"Expected 'hotwords_set', got: {resp}"
            print(f"  [PASS] HOTWORDS -> {len(resp['hotwords'])} words set")

        int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)
        chunk_size = 4096
        partial_count = 0

        for i in range(0, len(int16), chunk_size):
            chunk = int16[i:i+chunk_size]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(0.02)

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                data = json.loads(msg)
                if "sentences" in data:
                    partial_count += 1
                    assert "partial" in data
                    assert "duration_ms" in data
                    assert "is_final" in data
                    assert isinstance(data["sentences"], list)
            except asyncio.TimeoutError:
                pass

        await ws.send("STOP")

        final_result = None
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=30)
            data = json.loads(msg)
            if data.get("is_final"):
                final_result = data
            if data.get("event") == "stopped":
                break

        t_end = time.perf_counter()

    assert final_result is not None, "No final result received"
    assert final_result["is_final"] is True
    sentences = final_result["sentences"]
    assert len(sentences) > 0, "No sentences in final result"

    print(f"  [PASS] Received final result: {len(sentences)} sentences")
    print(f"  [PASS] Partial updates received: {partial_count}")

    for s in sentences:
        assert "text" in s, f"Missing 'text' in sentence: {s}"
        assert "start" in s, f"Missing 'start' in sentence: {s}"
        assert "end" in s, f"Missing 'end' in sentence: {s}"
        assert isinstance(s["text"], str) and len(s["text"]) > 0
        assert s["end"] > s["start"]

    print("  [PASS] All sentences have valid format (text, start, end)")

    has_spk = all("spk" in s for s in sentences)
    if has_spk:
        spk_ids = set(s["spk"] for s in sentences)
        print(f"  [PASS] Speaker diarization: {len(spk_ids)} speakers detected")
    else:
        print("  [INFO] No speaker IDs in result")

    elapsed = t_end - t_start
    audio_duration = len(audio) / SAMPLE_RATE
    rtf = elapsed / audio_duration
    print(f"\n  Audio: {audio_duration:.1f}s | Time: {elapsed:.2f}s | RTF: {rtf:.3f}")

    print("\n  --- Results ---")
    for s in sentences:
        spk = f" SPK{s['spk']}" if "spk" in s else ""
        print(f"  [{s['start']/1000:.1f}-{s['end']/1000:.1f}s]{spk}: {s['text']}")

    return True


async def test_empty_audio(server):
    """Test with very short/empty audio."""
    print("\n" + "=" * 60)
    print("TEST: Empty/Short Audio Handling")
    print("=" * 60)

    async with websockets.connect(server, ping_interval=None) as ws:
        await ws.send("START")
        await ws.recv()

        short_audio = np.zeros(800, dtype=np.int16)
        await ws.send(short_audio.tobytes())
        await asyncio.sleep(0.1)

        await ws.send("STOP")
        final = None
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(msg)
            if data.get("is_final"):
                final = data
            if data.get("event") == "stopped":
                break

        assert final is not None
        assert final["sentences"] == [] or all(s["text"].strip() for s in final["sentences"])
        print("  [PASS] Short audio handled gracefully")

    return True


async def test_multiple_sessions(server, audio):
    """Test multiple consecutive sessions on same connection."""
    print("\n" + "=" * 60)
    print("TEST: Multiple Sessions")
    print("=" * 60)

    audio_short = audio[:SAMPLE_RATE * 5]
    int16 = (audio_short * 32768).clip(-32768, 32767).astype(np.int16)

    async with websockets.connect(server, ping_interval=None) as ws:
        for session_num in range(2):
            await ws.send("START")
            resp = json.loads(await ws.recv())
            assert resp["event"] == "started"

            for i in range(0, len(int16), 4096):
                await ws.send(int16[i:i+4096].tobytes())
                await asyncio.sleep(0.01)

            await ws.send("STOP")
            got_final = False
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=15)
                data = json.loads(msg)
                if data.get("is_final"):
                    got_final = True
                if data.get("event") == "stopped":
                    break
            assert got_final, f"Session {session_num+1}: no final result"
            print(f"  [PASS] Session {session_num+1} completed")

    return True


async def run_tests(args):
    audio = load_audio(args.file)
    print(f"Loaded: {args.file} ({len(audio)/SAMPLE_RATE:.1f}s)")
    print(f"Server: {args.server}\n")

    passed = 0
    failed = 0

    tests = [
        ("Basic Streaming", lambda: test_basic(args.server, audio, args.hotwords)),
        ("Empty Audio", lambda: test_empty_audio(args.server)),
        ("Multiple Sessions", lambda: test_multiple_sessions(args.server, audio)),
    ]

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano Streaming Test")
    parser.add_argument("--server", type=str, default="ws://localhost:10095")
    parser.add_argument("--file", type=str, required=True, help="Test audio file")
    parser.add_argument("--hotwords", type=str, default="", help="Hotwords (comma-separated)")
    args = parser.parse_args()

    success = asyncio.run(run_tests(args))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
