#!/usr/bin/env python3
"""Demo: Fun-ASR-Nano with vLLM inference backend.

Usage:
    # Single GPU (greedy decoding)
    python demo_vllm.py

    # Multi-GPU tensor parallel
    python demo_vllm.py --tensor-parallel-size 2

    # Batch inference from wav.scp
    python demo_vllm.py --input wav.scp --tensor-parallel-size 4 --batch-size 32

    # With hotwords and language
    python demo_vllm.py --input audio.wav --language 中文 --hotwords 开放时间 周一
"""

import argparse
import os
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano vLLM Inference Demo")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="FunAudioLLM/Fun-ASR-Nano-2512",
        help="Model name (from hub) or local directory path",
    )
    parser.add_argument("--input", type=str, default=None, help="Audio file, wav.scp, or jsonl")
    parser.add_argument("--hub", type=str, default="ms", choices=["ms", "hf"])
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for audio encoder")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Number of GPUs for vLLM"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--language", type=str, default="中文", help="Language hint")
    parser.add_argument("--hotwords", type=str, nargs="*", default=[], help="Hotwords list")
    parser.add_argument("--no-itn", action="store_true", help="Disable inverse text normalization")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    args = parser.parse_args()

    from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

    print(f"=" * 60)
    print(f"Fun-ASR-Nano vLLM Inference")
    print(f"=" * 60)
    print(f"  Model: {args.model_dir}")
    print(f"  Tensor Parallel: {args.tensor_parallel_size} GPU(s)")
    print(f"  Dtype: {args.dtype}")
    print(f"  Language: {args.language}")
    print(f"  Hotwords: {args.hotwords or '(none)'}")
    print()

    t_load = time.perf_counter()
    engine = FunASRNanoVLLM.from_pretrained(
        model=args.model_dir,
        hub=args.hub,
        device=args.device,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    print(f"Model loaded in {time.perf_counter() - t_load:.1f}s\n")

    # Determine input files
    if args.input is None:
        # Use default example audio
        example_dir = os.path.join(engine.model_dir, "example")
        if os.path.isdir(example_dir):
            wav_files = [
                os.path.join(example_dir, f)
                for f in sorted(os.listdir(example_dir))
                if f.endswith((".wav", ".mp3", ".flac"))
            ]
        else:
            print("No --input specified and no example/ directory found.")
            print("Usage: python demo_vllm.py --input <audio_file_or_scp>")
            return
        if not wav_files:
            print("No audio files found in example/ directory.")
            return
        audio_files = wav_files
        print(f"Using example audio: {audio_files}")
    elif args.input.endswith(".scp"):
        audio_files = []
        with open(args.input, "r") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    audio_files.append(parts[1])
                elif len(parts) == 1:
                    audio_files.append(parts[0])
        print(f"Loaded {len(audio_files)} files from {args.input}")
    elif args.input.endswith(".jsonl"):
        import json

        audio_files = []
        with open(args.input, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                audio_files.append(item["source"])
        print(f"Loaded {len(audio_files)} files from {args.input}")
    else:
        audio_files = [args.input]

    # Run inference in batches
    all_results = []
    total_audio_time = 0
    total_infer_time = 0

    print(f"\nProcessing {len(audio_files)} audio file(s)...")
    for i in range(0, len(audio_files), args.batch_size):
        batch = audio_files[i : i + args.batch_size]
        t0 = time.perf_counter()
        results = engine.generate(
            inputs=batch,
            hotwords=args.hotwords if args.hotwords else None,
            language=args.language,
            itn=not args.no_itn,
            max_new_tokens=args.max_new_tokens,
        )
        t1 = time.perf_counter()
        batch_time = t1 - t0
        total_infer_time += batch_time
        all_results.extend(results)

        batch_num = i // args.batch_size + 1
        total_batches = (len(audio_files) + args.batch_size - 1) // args.batch_size
        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} files in {batch_time:.2f}s")

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Results: {len(all_results)} samples, total inference time: {total_infer_time:.2f}s")
    print(f"{'=' * 60}")
    for r in all_results:
        print(f"\n[{r['key']}]")
        print(f"  Text: {r['text']}")
        if "timestamps" in r and r["timestamps"]:
            ts_preview = r["timestamps"][:5]
            ts_str = " | ".join(
                [f"{t['token']}({t['start_time']:.2f}-{t['end_time']:.2f}s)" for t in ts_preview]
            )
            if len(r["timestamps"]) > 5:
                ts_str += f" ... ({len(r['timestamps'])} total)"
            print(f"  Timestamps: {ts_str}")

    # Save results to file
    if args.output:
        import json

        with open(args.output, "w", encoding="utf-8") as f:
            for r in all_results:
                # Remove non-serializable fields
                out = {k: v for k, v in r.items() if k != "timestamps"}
                if "timestamps" in r:
                    out["timestamps"] = r["timestamps"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
