#!/usr/bin/env python3
"""Fun-ASR-Nano Python WebSocket Client.

Supports real-time microphone recording and audio file streaming.

Usage:
    # Microphone mode
    python client_python.py --server ws://localhost:10095 --mic

    # File mode
    python client_python.py --server ws://localhost:10095 --file audio.wav

    # With hotwords
    python client_python.py --server ws://localhost:10095 --file audio.wav --hotwords "张三,李四,北京"

    # Disable speaker diarization display
    python client_python.py --server ws://localhost:10095 --mic --no-spk
"""

import asyncio
import argparse
import json
import sys
import numpy as np

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)


SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

SPK_COLORS = [
    "\033[36m", "\033[35m", "\033[33m", "\033[32m",
    "\033[34m", "\033[91m", "\033[96m", "\033[95m",
]
RESET = "\033[0m"
GRAY = "\033[90m"
GREEN = "\033[92m"


def format_time(ms):
    s = ms / 1000
    return f"{s:.1f}s"


def print_result(data, show_spk=True):
    """Print ASR result to terminal."""
    sentences = data.get("sentences", [])
    partial = data.get("partial", "")
    partial_start = data.get("partial_start_ms", 0)
    is_final = data.get("is_final", False)

    sys.stdout.write("\033[2J\033[H")

    print(f"{GREEN}Fun-ASR-Nano Streaming ASR{RESET}")
    print(f"{GRAY}{'─' * 60}{RESET}")

    for s in sentences:
        start = s.get("start", s.get("start_ms", 0))
        end = s.get("end", s.get("end_ms", 0))
        spk = s.get("spk", -1)
        text = s["text"]

        time_str = f"{GRAY}[{format_time(start)}-{format_time(end)}]{RESET}"
        spk_str = ""
        if show_spk and spk >= 0:
            color = SPK_COLORS[spk % len(SPK_COLORS)]
            spk_str = f" {color}SPK{spk}{RESET}"

        print(f"  {time_str}{spk_str} {text}")

    if partial:
        print(f"  {GRAY}[{format_time(partial_start)}-...] {partial}{RESET}")

    if is_final:
        print(f"\n{GRAY}{'─' * 60}{RESET}")
        print(f"{GREEN}Done.{RESET} {len(sentences)} sentences")
    else:
        print(f"\n{GRAY}Recording... Press Ctrl+C to stop{RESET}")

    sys.stdout.flush()


async def run_mic(args):
    """Stream from microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("Please install sounddevice: pip install sounddevice")
        sys.exit(1)

    print(f"Connecting to {args.server}...")
    async with websockets.connect(args.server, ping_interval=None) as ws:
        await ws.send("START")
        resp = await ws.recv()
        event = json.loads(resp)
        if event.get("event") != "started":
            print(f"Unexpected response: {resp}")
            return

        if args.hotwords:
            await ws.send(f"HOTWORDS:{args.hotwords}")
            await ws.recv()

        print("Recording... Press Ctrl+C to stop\n")

        audio_queue = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            audio_queue.put_nowait(indata.copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='int16',
            blocksize=CHUNK_SAMPLES, callback=audio_callback,
        )

        async def send_audio():
            with stream:
                while True:
                    chunk = await audio_queue.get()
                    await ws.send(chunk.tobytes())

        async def recv_results():
            async for msg in ws:
                data = json.loads(msg)
                if "sentences" in data:
                    print_result(data, show_spk=args.spk)
                if data.get("is_final") or data.get("event") == "stopped":
                    break

        send_task = asyncio.create_task(send_audio())
        recv_task = asyncio.create_task(recv_results())

        try:
            await asyncio.gather(send_task, recv_task)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            send_task.cancel()
            if ws.open:
                await ws.send("STOP")
                async for msg in ws:
                    data = json.loads(msg)
                    if "sentences" in data:
                        print_result(data, show_spk=args.spk)
                    if data.get("is_final") or data.get("event") == "stopped":
                        break


async def run_file(args):
    """Stream an audio file."""
    try:
        import soundfile as sf
    except ImportError:
        print("Please install soundfile: pip install soundfile")
        sys.exit(1)

    audio, sr = sf.read(args.file)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            print(f"Audio is {sr}Hz, need 16kHz. Install librosa: pip install librosa")
            sys.exit(1)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    duration = len(audio) / SAMPLE_RATE
    print(f"File: {args.file} ({duration:.1f}s)")
    print(f"Connecting to {args.server}...")

    async with websockets.connect(args.server, ping_interval=None) as ws:
        await ws.send("START")
        await ws.recv()

        if args.hotwords:
            await ws.send(f"HOTWORDS:{args.hotwords}")
            await ws.recv()

        int16 = (audio * 32768).clip(-32768, 32767).astype(np.int16)

        chunk_size = CHUNK_SAMPLES
        total_chunks = (len(int16) + chunk_size - 1) // chunk_size

        async def send_audio():
            for i in range(0, len(int16), chunk_size):
                chunk = int16[i:i+chunk_size]
                await ws.send(chunk.tobytes())
                await asyncio.sleep(CHUNK_DURATION_MS / 1000 * 0.5)
            await ws.send("STOP")

        async def recv_results():
            async for msg in ws:
                data = json.loads(msg)
                if "sentences" in data:
                    print_result(data, show_spk=args.spk)
                if data.get("is_final") or data.get("event") == "stopped":
                    break

        await asyncio.gather(send_audio(), recv_results())


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano Python Client")
    parser.add_argument("--server", type=str, default="ws://localhost:10095")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument("--file", type=str, help="Audio file to transcribe")
    parser.add_argument("--hotwords", type=str, default="", help="Hotwords (comma-separated)")
    parser.add_argument("--spk", action="store_true", default=True, help="Show speaker IDs")
    parser.add_argument("--no-spk", dest="spk", action="store_false")
    args = parser.parse_args()

    if not args.mic and not args.file:
        parser.error("Specify --mic or --file")

    try:
        if args.mic:
            asyncio.run(run_mic(args))
        else:
            asyncio.run(run_file(args))
    except KeyboardInterrupt:
        print(f"\n{RESET}Interrupted.")


if __name__ == "__main__":
    main()
