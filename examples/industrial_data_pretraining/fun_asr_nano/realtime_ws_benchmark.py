#!/usr/bin/env python3
"""Benchmark client for serve_realtime_ws.py.

The script replays a 16 kHz mono PCM16 WAV file to one or more realtime
WebSocket sessions and reports client-observable latency metrics. It does not
change the service and it does not require soundfile/librosa.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import wave
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets", file=sys.stderr)
    sys.exit(1)


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2


def load_pcm16_wav(path):
    """Read a mono 16 kHz PCM16 WAV file and return raw bytes plus duration."""
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        if channels != 1 or sample_width != SAMPLE_WIDTH_BYTES or sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{wav_path} must be 16 kHz mono PCM16 WAV; got "
                f"{sample_rate} Hz, {channels} channel(s), {sample_width * 8}-bit samples"
            )
        audio = wf.readframes(frames)
    return audio, frames / SAMPLE_RATE


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def round_or_none(value, digits=3):
    return None if value is None else round(value, digits)


async def receive_message(ws, timeout):
    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
    if isinstance(message, bytes):
        return {"_binary_bytes": len(message)}
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return {"_raw": message}


async def send_command(ws, command, expected_event, timeout):
    await ws.send(command)
    response = await receive_message(ws, timeout)
    if expected_event and response.get("event") != expected_event:
        raise RuntimeError(f"{command} expected event={expected_event!r}, got {response}")
    return response


async def recv_results(ws, metrics, audio_started_at, stop_sent_at_ref, timeout):
    while True:
        try:
            data = await receive_message(ws, timeout)
        except asyncio.TimeoutError:
            if metrics["final_messages"] == 0 and not metrics["stopped"]:
                metrics["errors"].append(f"timed out waiting for server message after {timeout}s")
            return
        except websockets.exceptions.ConnectionClosed as exc:
            if metrics["final_messages"] == 0 and not metrics["stopped"]:
                metrics["errors"].append(f"connection closed while receiving: {exc}")
            return

        now = time.perf_counter()
        metrics["messages"] += 1

        event = data.get("event")
        if event:
            metrics["events"][event] = metrics["events"].get(event, 0) + 1

        has_result = "sentences" in data or "partial" in data or data.get("is_final") is True
        if has_result:
            metrics["result_messages"] += 1
            if metrics["first_update_ms"] is None:
                metrics["first_update_ms"] = (now - audio_started_at) * 1000.0
            if data.get("partial"):
                metrics["partial_messages"] += 1
            duration_ms = data.get("duration_ms")
            if data.get("is_final") is not True and isinstance(duration_ms, (int, float)):
                metrics["response_lag_ms"].append((now - audio_started_at) * 1000.0 - duration_ms)

        if data.get("is_final") is True:
            metrics["final_messages"] += 1
            metrics["final_update_ms"] = (now - audio_started_at) * 1000.0
            if stop_sent_at_ref["value"] is not None:
                metrics["final_after_stop_ms"] = (now - stop_sent_at_ref["value"]) * 1000.0

        if event == "stopped":
            metrics["stopped"] = True
            return


async def run_client(client_id, args, audio_bytes, audio_seconds):
    chunk_bytes = max(1, int(SAMPLE_RATE * args.chunk_ms / 1000.0)) * SAMPLE_WIDTH_BYTES
    total_audio_seconds = audio_seconds * args.loops
    metrics = {
        "client_id": client_id,
        "audio_seconds": total_audio_seconds,
        "chunk_ms": args.chunk_ms,
        "messages": 0,
        "result_messages": 0,
        "partial_messages": 0,
        "final_messages": 0,
        "events": {},
        "first_update_ms": None,
        "final_update_ms": None,
        "final_after_stop_ms": None,
        "response_lag_ms": [],
        "stopped": False,
        "errors": [],
    }

    wall_started_at = time.perf_counter()
    stop_sent_at_ref = {"value": None}

    try:
        async with websockets.connect(
            args.server,
            ping_interval=None,
            open_timeout=args.connect_timeout,
            max_size=args.max_message_size,
        ) as ws:
            await send_command(ws, "START", "started", args.recv_timeout)
            if args.hotwords:
                await send_command(ws, f"HOTWORDS:{args.hotwords}", "hotwords_set", args.recv_timeout)
            if args.language:
                await send_command(ws, f"LANGUAGE:{args.language}", "language_set", args.recv_timeout)

            audio_started_at = time.perf_counter()
            recv_task = asyncio.create_task(
                recv_results(ws, metrics, audio_started_at, stop_sent_at_ref, args.recv_timeout)
            )

            sent_audio_seconds = 0.0
            for _ in range(args.loops):
                for offset in range(0, len(audio_bytes), chunk_bytes):
                    chunk = audio_bytes[offset : offset + chunk_bytes]
                    await ws.send(chunk)
                    sent_audio_seconds += len(chunk) / (SAMPLE_RATE * SAMPLE_WIDTH_BYTES)
                    if args.pace:
                        target_elapsed = sent_audio_seconds
                        elapsed = time.perf_counter() - audio_started_at
                        delay = target_elapsed - elapsed
                        if delay > 0:
                            await asyncio.sleep(delay)

            stop_sent_at_ref["value"] = time.perf_counter()
            metrics["send_seconds"] = stop_sent_at_ref["value"] - audio_started_at
            await ws.send("STOP")
            await recv_task
    except Exception as exc:
        metrics["errors"].append(str(exc))

    wall_seconds = time.perf_counter() - wall_started_at
    lags = metrics.pop("response_lag_ms")
    send_seconds = metrics.get("send_seconds")
    audio_per_wall = total_audio_seconds / wall_seconds if wall_seconds else None
    send_audio_per_wall = total_audio_seconds / send_seconds if send_seconds else None
    response_lag_ms_max = max(lags) if lags else None
    response_lag_ms_p95 = percentile(lags, 95)
    metrics.update(
        {
            "first_update_ms": round_or_none(metrics["first_update_ms"], 1),
            "final_update_ms": round_or_none(metrics["final_update_ms"], 1),
            "final_after_stop_ms": round_or_none(metrics["final_after_stop_ms"], 1),
            "wall_seconds": round(wall_seconds, 3),
            "send_seconds": round_or_none(send_seconds, 3),
            "audio_per_wall": round_or_none(audio_per_wall),
            "send_audio_per_wall": round_or_none(send_audio_per_wall),
            "response_lag_ms_max": round_or_none(response_lag_ms_max, 1),
            "response_lag_ms_p95": round_or_none(response_lag_ms_p95, 1),
        }
    )
    return metrics


def summarize(results, elapsed_seconds):
    total_audio = sum(item["audio_seconds"] for item in results)
    first_updates = [item["first_update_ms"] for item in results if item["first_update_ms"] is not None]
    final_after_stop = [
        item["final_after_stop_ms"] for item in results if item["final_after_stop_ms"] is not None
    ]
    lag_p95 = [item["response_lag_ms_p95"] for item in results if item["response_lag_ms_p95"] is not None]
    errors = sum(len(item["errors"]) for item in results)
    return {
        "clients": len(results),
        "total_audio_seconds": round(total_audio, 3),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "aggregate_audio_per_wall": round_or_none(total_audio / elapsed_seconds if elapsed_seconds else None),
        "first_update_ms_p50": round_or_none(statistics.median(first_updates), 1)
        if first_updates
        else None,
        "first_update_ms_p95": round_or_none(percentile(first_updates, 95), 1),
        "final_after_stop_ms_p50": round_or_none(statistics.median(final_after_stop), 1)
        if final_after_stop
        else None,
        "final_after_stop_ms_p95": round_or_none(percentile(final_after_stop, 95), 1),
        "client_response_lag_ms_p95_max": round_or_none(max(lag_p95), 1) if lag_p95 else None,
        "partial_messages": sum(item["partial_messages"] for item in results),
        "final_messages": sum(item["final_messages"] for item in results),
        "errors": errors,
    }


def print_summary(summary, results):
    print("FunASR realtime WebSocket benchmark")
    print(f"clients: {summary['clients']}")
    print(f"total audio seconds: {summary['total_audio_seconds']}")
    print(f"elapsed seconds: {summary['elapsed_seconds']}")
    print(f"aggregate audio/wall: {summary['aggregate_audio_per_wall']}x")
    print(f"first update p50/p95 ms: {summary['first_update_ms_p50']} / {summary['first_update_ms_p95']}")
    print(
        "final after STOP p50/p95 ms: "
        f"{summary['final_after_stop_ms_p50']} / {summary['final_after_stop_ms_p95']}"
    )
    print(f"max client response-lag p95 ms: {summary['client_response_lag_ms_p95_max']}")
    print(f"partial/final messages: {summary['partial_messages']} / {summary['final_messages']}")
    print(f"errors: {summary['errors']}")
    if summary["errors"]:
        for item in results:
            for error in item["errors"]:
                print(f"client {item['client_id']} error: {error}")


async def async_main(args):
    audio_bytes, audio_seconds = load_pcm16_wav(args.wav)
    started_at = time.perf_counter()
    results = await asyncio.gather(
        *(run_client(client_id, args, audio_bytes, audio_seconds) for client_id in range(args.clients))
    )
    elapsed = time.perf_counter() - started_at
    summary = summarize(results, elapsed)
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        with output_path.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps({"type": "client", **result}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
    print_summary(summary, results)
    return 1 if summary["errors"] else 0


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark serve_realtime_ws.py with one or more clients")
    parser.add_argument("wav", help="16 kHz mono PCM16 WAV file")
    parser.add_argument("--server", default="ws://localhost:10095", help="WebSocket server URL")
    parser.add_argument("--clients", type=int, default=1, help="Concurrent clients")
    parser.add_argument("--loops", type=int, default=1, help="Times each client replays the WAV before STOP")
    parser.add_argument("--chunk-ms", type=int, default=100, help="PCM frame duration per WebSocket send")
    parser.add_argument("--language", default="", help="Optional LANGUAGE command value")
    parser.add_argument("--hotwords", default="", help="Optional HOTWORDS command value, comma separated")
    parser.add_argument("--no-pace", dest="pace", action="store_false", help="Send as fast as possible")
    parser.add_argument("--output-jsonl", default="", help="Write per-client metrics and summary JSONL")
    parser.add_argument("--connect-timeout", type=float, default=10.0, help="Connection timeout seconds")
    parser.add_argument("--recv-timeout", type=float, default=30.0, help="Timeout waiting for server messages")
    parser.add_argument("--max-message-size", type=int, default=16 * 1024 * 1024, help="WebSocket max message size")
    parser.set_defaults(pace=True)
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients must be >= 1")
    if args.loops < 1:
        parser.error("--loops must be >= 1")
    if args.chunk_ms < 10:
        parser.error("--chunk-ms must be >= 10")
    return args


def main():
    args = parse_args()
    try:
        raise SystemExit(asyncio.run(async_main(args)))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
