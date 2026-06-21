#!/usr/bin/env python3
"""Export Paraformer (SANM encoder + CIF predictor + SANM decoder) to GGUF.
Encoder reuses the shared SAN-M forward. Predictor (CIF) runs on host in C++.
"""
import argparse, os, re
import numpy as np, torch, gguf


def parse_mvn(path):
    # am.mvn (kaldi nnet) has 3 bracketed blocks: [Splice idx], [AddShift=shift],
    # [Rescale=scale]. Take the two 560-length vectors (shift then scale).
    # apply: out = (x + shift) * scale
    blocks = [np.array([float(x) for x in b.split()], np.float32)
              for b in re.findall(r"\[([^\]]*)\]", open(path).read())]
    vecs = [b for b in blocks if b.size > 1]
    return vecs[0], vecs[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pt", required=True); ap.add_argument("--mvn", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--wtype", default="f32", choices=["f32","f16","q8_0"])
    ap.add_argument("--tokens", default=None, help="tokens.json (id->token); default: next to model_pt")
    a = ap.parse_args()
    sd = torch.load(a.model_pt, map_location="cpu"); sd = sd.get("state_dict", sd)
    w = gguf.GGUFWriter(a.out, "paraformer")
    w.add_uint32("pf.enc.output_size", 512); w.add_uint32("pf.enc.attention_heads", 4)
    w.add_uint32("pf.enc.num_blocks", 50); w.add_uint32("pf.enc.kernel_size", 11)
    w.add_uint32("pf.dec.num_blocks", 16); w.add_uint32("pf.dec.att_layer_num", 16)
    w.add_uint32("pf.dec.decoders3", 1); w.add_uint32("pf.dec.attention_heads", 4)
    w.add_uint32("pf.dec.kernel_size", 11); w.add_uint32("pf.vocab_size", 8404)
    import json, glob
    tp = a.tokens or (glob.glob(os.path.join(os.path.dirname(a.model_pt), "tokens.json")) + [None])[0]
    if tp and os.path.exists(tp):
        with open(tp, encoding="utf-8") as f: toks = json.load(f)
        w.add_array("pf.vocab", toks)
        print(f"embedded pf.vocab ({len(toks)} tokens) from {tp}")
    else:
        print("WARNING: tokens.json not found - gguf will have no vocab (binary falls back to ids)")
    w.add_float32("pf.predictor.tail_threshold", 0.45)
    w.add_float32("pf.predictor.threshold", 1.0)
    shift, scale = parse_mvn(a.mvn)
    w.add_tensor("cmvn.shift", shift); w.add_tensor("cmvn.scale", scale)
    n = 0
    for k, v in sd.items():
        if not (k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("predictor.")):
            continue
        if k == "decoder.embed.0.weight":   # token embedding, unused at NAR inference
            continue
        arr = v.detach().to(torch.float32).contiguous().numpy()
        if k.endswith("fsmn_block.weight") and arr.ndim == 3:   # (D,1,K)->(K,D)
            arr = np.ascontiguousarray(arr[:, 0, :].T)
        elif args_f16(a) and arr.ndim == 2 and "norm" not in k and "cif_output" not in k:
            arr = arr.astype(np.float16)
        if a.wtype == "q8_0" and arr.ndim == 2 and "norm" not in k and "fsmn_block" not in k and "predictor" not in k and arr.shape[1] % 32 == 0:
            from gguf import quants as _q, GGMLQuantizationType as _QT
            w.add_tensor(k, _q.quantize(arr, _QT.Q8_0), raw_dtype=_QT.Q8_0)
        else:
            w.add_tensor(k, arr)
        n += 1
    print(f"writing {n} tensors (+cmvn) to {a.out}")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"done: {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")


def args_f16(a): return a.wtype == "f16"

if __name__ == "__main__":
    main()
