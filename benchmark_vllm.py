#!/usr/bin/env python3
"""FunASR vLLM Benchmark: unified speed + CER comparison for all supported models.

Supports: Fun-ASR-Nano, GLM-ASR-Nano (and any model via AutoModelVLLM).

Usage:
    # Fun-ASR-Nano
    CUDA_VISIBLE_DEVICES=0 python benchmark_vllm.py \
        --model FunAudioLLM/Fun-ASR-Nano-2512 \
        --audio-dir /path/to/benchmark_audio \
        --label-json /path/to/benchmark_testset.json

    # GLM-ASR-Nano
    CUDA_VISIBLE_DEVICES=0 python benchmark_vllm.py \
        --model zai-org/GLM-ASR-Nano-2512 \
        --audio-dir /path/to/benchmark_audio \
        --label-json /path/to/benchmark_testset.json

    # Quick test (first N files)
    CUDA_VISIBLE_DEVICES=0 python benchmark_vllm.py \
        --model FunAudioLLM/Fun-ASR-Nano-2512 \
        --audio-dir /path/to/benchmark_audio \
        --label-json /path/to/benchmark_testset.json \
        --max-files 20

    # Skip PyTorch (only test vLLM)
    CUDA_VISIBLE_DEVICES=0 python benchmark_vllm.py \
        --model FunAudioLLM/Fun-ASR-Nano-2512 \
        --skip-pytorch \
        --audio-dir /path/to/benchmark_audio \
        --label-json /path/to/benchmark_testset.json
"""

import argparse
import json
import os
import re
import time

import kaldialign
import numpy as np
import soundfile as sf
import torch


def normalize_zh(text):
    text = re.sub(r'[^\w一-鿿]', '', text)
    return text.upper()


def compute_cer(refs, hyps):
    total_ref = 0
    total_errs = 0
    for ref, hyp in zip(refs, hyps):
        r = list(normalize_zh(ref))
        h = list(normalize_zh(hyp))
        total_ref += len(r)
        ali = kaldialign.align(r, h, '*')
        total_errs += sum(1 for a, b in ali if a != b)
    return total_errs / total_ref * 100 if total_ref > 0 else 0


def vad_segment(files, device="cuda:0"):
    from funasr import AutoModel
    vad_model = AutoModel(model="fsmn-vad", device=device, disable_update=True)
    all_segments = []
    for fi, wav_path in enumerate(files):
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        res = vad_model.generate(input=wav_path, dynamic_silence=False)
        for seg in res[0]["value"]:
            s0 = int(seg[0] * sr / 1000)
            s1 = int(seg[1] * sr / 1000)
            seg_audio = audio[s0:s1]
            if len(seg_audio) > sr * 0.5:
                all_segments.append((fi, seg_audio))
    return all_segments


def concat_results(all_segments, seg_texts, n_files):
    file_texts = {}
    for (fi, _), text in zip(all_segments, seg_texts):
        file_texts.setdefault(fi, []).append(text)
    return ["".join(file_texts.get(fi, [])) for fi in range(n_files)]


def run_pytorch(model_name, seg_files, device="cuda:0"):
    from funasr import AutoModel

    kwargs = {"model": model_name, "device": device, "disable_update": True}
    if "Fun-ASR-Nano" in model_name:
        kwargs["trust_remote_code"] = True
        kwargs["remote_code"] = os.path.join(
            os.path.dirname(__file__),
            "examples/industrial_data_pretraining/fun_asr_nano/model.py"
        )

    model = AutoModel(**kwargs)
    model.generate(input=seg_files[0])  # warmup

    t0 = time.perf_counter()
    texts = []
    for f in seg_files:
        res = model.generate(input=f)
        texts.append(res[0]["text"])
    t1 = time.perf_counter()
    return t1 - t0, texts


def run_vllm(model_name, seg_files, device="cuda:0", hub="ms"):
    if "Fun-ASR-Nano" in model_name:
        from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
        engine = FunASRNanoVLLM.from_pretrained(
            model=model_name, hub=hub, device=device, dtype="bf16",
            max_model_len=4096, gpu_memory_utilization=0.5)
        engine.generate(inputs=[seg_files[0]], language="中文")  # warmup
        t0 = time.perf_counter()
        results = engine.generate(inputs=seg_files, language="中文", max_new_tokens=500)
        t1 = time.perf_counter()
        texts = [r["text"] for r in results]

    elif "GLM-ASR" in model_name:
        from funasr.models.glm_asr.inference_vllm import GLMASRVLLMEngine
        engine = GLMASRVLLMEngine.from_pretrained(
            model=model_name, hub=hub, device=device, dtype="bf16",
            gpu_memory_utilization=0.4, max_model_len=4096)
        engine.generate(inputs=[seg_files[0]])  # warmup
        t0 = time.perf_counter()
        results = engine.generate(inputs=seg_files, max_new_tokens=500)
        t1 = time.perf_counter()
        texts = [r["text"] for r in results]

    else:
        from funasr.auto.auto_model_vllm import AutoModelVLLM
        engine = AutoModelVLLM(model=model_name, hub=hub, device=device)
        engine.generate(inputs=[seg_files[0]])
        t0 = time.perf_counter()
        results = engine.generate(inputs=seg_files, max_new_tokens=500)
        t1 = time.perf_counter()
        texts = [r["text"] for r in results]

    return t1 - t0, texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FunASR vLLM Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--hub", type=str, default="ms", choices=["ms", "hf"])
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--label-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--skip-pytorch", action="store_true")
    args = parser.parse_args()

    with open(args.label_json) as f:
        dataset = json.load(f)

    files = []
    refs = []
    for item in dataset:
        wav_path = os.path.join(args.audio_dir, f"{item['id']:03d}.wav")
        if os.path.exists(wav_path):
            files.append(wav_path)
            refs.append(item["ref"])
    if args.max_files > 0:
        files = files[:args.max_files]
        refs = refs[:args.max_files]

    total_audio = sum(sf.info(f).duration for f in files)
    print(f"{'='*60}")
    print(f"FunASR vLLM Benchmark")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Dataset: {len(files)} files, {total_audio:.0f}s audio")

    # VAD
    print(f"\n>>> VAD pre-segmenting...")
    all_segments = vad_segment(files, device=args.device)
    print(f"    {len(all_segments)} segments")

    os.makedirs("/tmp/benchmark_vllm_segs", exist_ok=True)
    seg_files = []
    for i, (fi, audio) in enumerate(all_segments):
        path = f"/tmp/benchmark_vllm_segs/{i:04d}.wav"
        sf.write(path, audio, 16000)
        seg_files.append(path)

    # PyTorch
    cer_pt = None
    pt_time = None
    if not args.skip_pytorch:
        print(f"\n>>> PyTorch native...")
        pt_time, pt_seg_texts = run_pytorch(args.model, seg_files, device=args.device)
        pt_texts = concat_results(all_segments, pt_seg_texts, len(files))
        cer_pt = compute_cer(refs, pt_texts)
        print(f"    Time: {pt_time:.1f}s | RTFx: {total_audio/pt_time:.1f} | CER: {cer_pt:.2f}%")

        del torch.cuda.memory_allocated
        torch.cuda.empty_cache()

    # vLLM
    print(f"\n>>> vLLM...")
    vllm_time, vllm_seg_texts = run_vllm(args.model, seg_files, device=args.device, hub=args.hub)
    vllm_texts = concat_results(all_segments, vllm_seg_texts, len(files))
    cer_vllm = compute_cer(refs, vllm_texts)
    print(f"    Time: {vllm_time:.1f}s | RTFx: {total_audio/vllm_time:.1f} | CER: {cer_vllm:.2f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'-'*60}")
    print(f"{'Method':<20} {'Time':<10} {'RTFx':<10} {'CER'}")
    print(f"{'-'*60}")
    if not args.skip_pytorch:
        print(f"{'PyTorch':<20} {pt_time:<10.1f} {total_audio/pt_time:<10.1f} {cer_pt:.2f}%")
    print(f"{'vLLM':<20} {vllm_time:<10.1f} {total_audio/vllm_time:<10.1f} {cer_vllm:.2f}%")
    if not args.skip_pytorch:
        print(f"{'-'*60}")
        speedup = (total_audio/vllm_time) / (total_audio/pt_time)
        print(f"Speedup: {speedup:.1f}x | CER diff: {cer_vllm - cer_pt:+.2f}%")
    print(f"{'='*60}")
