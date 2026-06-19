#!/usr/bin/env python3
"""Compute micro-CER (normalize_zh) and RTF for an ASR system's hypotheses.

Usage:
    python compute_cer.py --refs testset.json --hyp_dir <dir of {key}.txt> \
        [--time_file <key compute_seconds per line>]

- micro-CER = sum(edit distance) / sum(reference chars), over all files.
- normalize_zh(text) = re.sub(r'[^\\w一-鿿]', '', text).upper()  (the FunASR口径)
- RTF = sum(compute_time) / sum(audio_duration)  (model-load excluded)
testset.json: list of {"id" or "key", "ref", "duration"}.
"""
import argparse, json, glob, os, re
import numpy as np

def normalize_zh(s):
    s = re.sub(r"<\|[^|]*\|>", "", s)          # drop SenseVoice meta tags, if any
    return re.sub(r"[^\w一-鿿]", "", s).upper()

def edist(r, h):
    r, h = list(r), list(h)
    if not r: return len(h)
    d = np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        prev = d[0]; d[0] = i
        for j in range(1, len(h)+1):
            cur = d[j]
            d[j] = min(d[j]+1, d[j-1]+1, prev + (r[i-1] != h[j-1]))
            prev = cur
    return int(d[len(h)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--hyp_dir", required=True)
    ap.add_argument("--time_file", default=None)
    a = ap.parse_args()
    refs = {}
    dur = {}
    for it in json.load(open(a.refs)):
        k = f"{it.get('id', it.get('key')):03d}" if isinstance(it.get('id', it.get('key')), int) else str(it.get('key'))
        refs[k] = it["ref"]; dur[k] = float(it.get("duration", 0))
    times = {}
    if a.time_file and os.path.exists(a.time_file):
        for ln in open(a.time_file):
            p = ln.split()
            if len(p) >= 2:
                try: times[p[0]] = float(p[1])
                except ValueError: pass
    E = N = 0; rt = ad = 0.0; n = 0
    for p in glob.glob(os.path.join(a.hyp_dir, "*.txt")):
        k = os.path.splitext(os.path.basename(p))[0]
        if k not in refs: continue
        h = normalize_zh(open(p).read()); r = normalize_zh(refs[k])
        E += edist(r, h); N += len(r); n += 1
        if k in times: rt += times[k]; ad += dur[k]
    cer = E / max(N, 1) * 100
    print(f"files={n}  micro-CER={cer:.2f}%", end="")
    if ad > 0: print(f"  RTF={rt/ad:.4f} ({ad/rt:.1f}x real-time)")
    else: print()

if __name__ == "__main__":
    main()
