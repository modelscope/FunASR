#!/usr/bin/env python3
"""Detokenize Paraformer token ids -> text via tokens.json.
Usage: python detok_paraformer.py <tokens.json> <ids.txt>"""
import sys, json
toks = json.load(open(sys.argv[1]))
ids = [int(x) for x in open(sys.argv[2]).read().split() if int(x) not in (1, 2)]  # drop sos/eos
out = "".join(toks[i] for i in ids if 0 <= i < len(toks))
print(out.replace("@@", "").replace("▁", " ").strip())
