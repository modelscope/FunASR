#!/usr/bin/env python3
"""Export Fun-ASR-Nano audio encoder + adaptor weights to a GGUF file.

Packs all `audio_encoder.*` and `audio_adaptor.*` tensors (bf16 -> f32) plus
architecture metadata into funasr-encoder.gguf, for the ggml C++ forward pass.
Tensor names are kept verbatim (e.g. audio_encoder.encoders.3.norm1.weight) so
the C++ side can look them up directly.
"""
import argparse, os
import numpy as np
import torch
import gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wtype", default="f32", choices=["f32", "f16", "q8_0"],
                    help="dtype for 2D Linear (matmul) weights; norms/bias/fsmn stay f32")
    args = ap.parse_args()

    sd = torch.load(args.model_pt, map_location="cpu")
    sd = sd.get("state_dict", sd)

    w = gguf.GGUFWriter(args.out, "funasr-sensevoice-encoder")

    # --- architecture metadata (from config.yaml) ---
    w.add_uint32("funasr.enc.input_size", 560)      # lfr_m(7) * n_mels(80)
    w.add_uint32("funasr.enc.output_size", 512)
    w.add_uint32("funasr.enc.attention_heads", 4)
    w.add_uint32("funasr.enc.linear_units", 2048)
    w.add_uint32("funasr.enc.num_blocks", 50)       # encoders0(1) + encoders(49)
    w.add_uint32("funasr.enc.tp_blocks", 20)
    w.add_uint32("funasr.enc.kernel_size", 11)
    w.add_uint32("funasr.enc.sanm_shfit", 0)
    w.add_uint32("funasr.adp.llm_dim", 1024)
    w.add_uint32("funasr.adp.encoder_dim", 512)
    w.add_uint32("funasr.adp.ffn_dim", 2048)
    w.add_uint32("funasr.adp.n_layer", 2)
    w.add_uint32("funasr.adp.attention_heads", 8)
    w.add_uint32("funasr.adp.downsample_rate", 1)
    w.add_uint32("funasr.frontend.n_mels", 80)
    w.add_uint32("funasr.frontend.lfr_m", 7)
    w.add_uint32("funasr.frontend.lfr_n", 6)

    n = 0
    for k, v in sd.items():
        if not (k.startswith("audio_encoder.") or k.startswith("audio_adaptor.")):
            continue
        arr = v.detach().to(torch.float32).contiguous().numpy()
        # FSMN depthwise kernel: store as (K, D) so the C++ side can slice a
        # contiguous per-tap [D] vector and do an exact f32 shift-accumulate
        # (avoids the F16-only ggml_conv_1d_dw path).
        if k.endswith("fsmn_block.weight"):       # (D, 1, K) -> (K, D)
            arr = np.ascontiguousarray(arr[:, 0, :].T)
        # matmul (Linear) weights -> optional f16; norms/biases/fsmn stay f32
        elif args.wtype == "f16" and arr.ndim == 2 and "norm" not in k:
            arr = arr.astype(np.float16)
        if args.wtype == "q8_0" and arr.ndim == 2 and "norm" not in k and "fsmn_block" not in k and arr.shape[1] % 32 == 0:
            from gguf import quants as _q, GGMLQuantizationType as _QT
            w.add_tensor(k, _q.quantize(arr, _QT.Q8_0), raw_dtype=_QT.Q8_0)
        else:
            w.add_tensor(k, arr)
        n += 1
    print(f"writing {n} tensors to {args.out}")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"done: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
