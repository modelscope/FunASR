#!/usr/bin/env python3
"""Export SenseVoiceSmall (encoder + CTC head + query embeddings + CMVN) to GGUF
for the ggml C++ runtime. The encoder is the same SAN-M architecture as
Fun-ASR-Nano, so the C++ forward is shared.
"""
import argparse, os, re
import numpy as np, torch, gguf


def parse_mvn(path):
    """am.mvn (kaldi nnet): two `<LearnRateCoef> 0 [ ... ]` blocks -> shift, scale.
    apply: out = (in + shift) * scale, per-dim (560)."""
    txt = open(path).read()
    blocks = re.findall(r"\[([^\]]*)\]", txt)
    shift = np.array([float(x) for x in blocks[0].split()], dtype=np.float32)
    scale = np.array([float(x) for x in blocks[1].split()], dtype=np.float32)
    return shift, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pt", required=True)
    ap.add_argument("--mvn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wtype", default="f32", choices=["f32", "f16"])
    ap.add_argument("--spm", default=None, help="sentencepiece .bpe.model; default: next to model_pt")
    args = ap.parse_args()

    sd = torch.load(args.model_pt, map_location="cpu"); sd = sd.get("state_dict", sd)
    w = gguf.GGUFWriter(args.out, "sensevoice-small")
    w.add_uint32("sv.input_size", 560)
    w.add_uint32("sv.output_size", 512)
    w.add_uint32("sv.attention_heads", 4)
    w.add_uint32("sv.num_blocks", 50)
    w.add_uint32("sv.tp_blocks", 20)
    w.add_uint32("sv.kernel_size", 11)
    w.add_uint32("sv.vocab_size", 25055)
    w.add_uint32("sv.blank_id", 0)
    # query token embed indices used at inference: [lid(auto=0), 1, 2, textnorm(woitn=15)]
    w.add_array("sv.query_tokens", [0, 1, 2, 14])  # 14=withitn (use_itn=True), matches authoritative
    import glob
    spm_path = args.spm or (glob.glob(os.path.join(os.path.dirname(args.model_pt), "*.bpe.model")) + [None])[0]
    if spm_path and os.path.exists(spm_path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=spm_path)
        pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        w.add_array("sv.vocab", pieces)
        print(f"embedded sv.vocab ({len(pieces)} pieces) from {spm_path}")
    else:
        print("WARNING: *.bpe.model not found - gguf will have no vocab (binary falls back to ids)")

    shift, scale = parse_mvn(args.mvn)
    w.add_tensor("cmvn.shift", shift)   # (560,)
    w.add_tensor("cmvn.scale", scale)   # (560,)

    n = 0
    for k, v in sd.items():
        if not (k.startswith("encoder.") or k.startswith("ctc.") or k == "embed.weight"):
            continue
        arr = v.detach().to(torch.float32).contiguous().numpy()
        if k.endswith("fsmn_block.weight"):           # (D,1,K) -> (K,D)
            arr = np.ascontiguousarray(arr[:, 0, :].T)
        elif args.wtype == "f16" and arr.ndim == 2 and "norm" not in k:
            arr = arr.astype(np.float16)
        w.add_tensor(k, arr)
        n += 1
    print(f"writing {n} tensors (+cmvn) to {args.out}")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"done: {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
