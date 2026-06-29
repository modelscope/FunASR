#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-ASR streaming WebSocket service.

把官方 example_qwen3_asr_vllm_streaming.py 的原生流式 API
(init_streaming_state / streaming_transcribe / finish_streaming_transcribe)
包成一个 WebSocket 服务，协议与 Fun-ASR-Nano 的 serve_realtime_ws.py 一致，
因此可以直接用同一个 bench_streaming_ws.py 压测,对比。

协议：
  1. 客户端连接 ws://host:port
  2. 客户端发文本 "START"            → 服务端回 {"event": "started"}
  3. 客户端发二进制 int16 PCM 块（16kHz 单声道）
  4. 服务端随转写增长发 {"partial": "<当前文本>"}
  5. 客户端发文本 "STOP"             → 服务端回
        {"is_final": true, "sentences": [{"text": "<最终文本>"}]}
        然后 {"event": "stopped"}

架构对齐 serve_realtime_ws.py：单 asyncio 事件循环，streaming_transcribe /
finish_streaming_transcribe 同步调用、阻塞整个循环——这样压测出来的并发特性
才和 Fun-ASR-Nano 那条同口径可比。生产扩展同样靠 多进程 + CUDA MPS + nginx
（见 vllm_guide §6.7）。

关于 VAD 参见配套说明文档


依赖：
  pip install -U "qwen-asr[vllm]==0.0.6" "transformers==4.57.6" websockets numpy
启动：
  python serve_qwen3_asr_ws.py --port 10095 --gpu-memory-utilization 0.8
  # 可选：--chunk-size-sec 控制流式块大小（默认 2.0）。值越小出字越快/越勤，
  #       但并发开销越大（实测 1.0 比 2.0 明显更吃并发）。
"""
import asyncio
import argparse
import json
import logging

import numpy as np
import websockets

from qwen_asr import Qwen3ASRModel

# websockets 默认会对每个连接打 INFO 级 "connection open/closed"，压测时刷屏；
# 提到 WARNING 关掉这条噪音（不影响连接行为，纯日志）。
logging.getLogger("websockets").setLevel(logging.WARNING)

SAMPLE_RATE = 16000

# 全局只加载一次；所有连接共用模型，各自持有独立的 streaming state。
asr = None

# 流式块大小（秒），由 --chunk-size-sec 设置，handle_client 里 init_streaming_state 用。
# 默认 2.0（官方 example 值）；值越小出字越勤、并发开销越大。
CHUNK_SIZE_SEC = 2.0


def int16_pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """bench 发来的是 int16 小端 PCM；Qwen3-ASR 的 streaming_transcribe 吃 float32 [-1,1)。"""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


async def handle_client(ws, path=None):  # path 兼容老版本 websockets 的两参回调
    state = None
    last_partial = None
    try:
        async for msg in ws:
            # ---- 文本控制消息 ----
            if isinstance(msg, str):
                if msg == "START":
                    # 每个连接一份独立 state；参数同官方 example
                    state = asr.init_streaming_state(
                        unfixed_chunk_num=2,
                        unfixed_token_num=5,
                        chunk_size_sec=CHUNK_SIZE_SEC,
                    )
                    last_partial = None
                    await ws.send(json.dumps({"event": "started"}))

                elif msg == "STOP":
                    if state is not None:
                        # 同步收尾，阻塞循环（与 serve_realtime_ws.py 句尾 finalize 同口径）
                        asr.finish_streaming_transcribe(state)
                        final_text = (state.text or "").strip()
                        await ws.send(json.dumps({
                            "is_final": True,
                            "sentences": [{"text": final_text}] if final_text else [],
                        }))
                    await ws.send(json.dumps({"event": "stopped"}))
                    break
                # 其它文本忽略

            # ---- 二进制音频块 ----
            else:
                if state is None:
                    continue  # 还没 START，丢弃
                seg = int16_pcm_to_float32(msg)
                # 同步调用，阻塞整个事件循环 —— 这正是要复刻的单循环架构
                asr.streaming_transcribe(seg, state)
                text = state.text or ""
                # 只在文本变化时发 partial，避免刷屏（不影响 bench 的首词延迟统计）
                if text != last_partial:
                    last_partial = text
                    await ws.send(json.dumps({"partial": text}))

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        logging.exception("Unexpected error in Qwen3-ASR WebSocket handler")


async def amain(args):
    global asr, CHUNK_SIZE_SEC
    CHUNK_SIZE_SEC = args.chunk_size_sec
    print(f"Loading {args.model} (gpu_memory_utilization={args.gpu_memory_utilization}, chunk_size_sec={CHUNK_SIZE_SEC}) ...")
    # Streaming is vLLM-only and no forced aligner supported.（官方 example 注释）
    asr = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=32,  # 流式用小值，同官方 example
    )
    print(f"Serving on ws://{args.host}:{args.port}  (Ctrl-C to stop)")
    async with websockets.serve(
        handle_client, args.host, args.port, max_size=10 * 1024 * 1024
    ):
        await asyncio.Future()  # run forever


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=10095)
    p.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                   dest="gpu_memory_utilization")
    p.add_argument("--chunk-size-sec", type=float, default=2.0,
                   dest="chunk_size_sec",
                   help="流式块大小(秒)，传给 init_streaming_state。默认 2.0；越小出字越勤但并发开销越大(实测 1.0 比 2.0 明显更吃并发)")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
