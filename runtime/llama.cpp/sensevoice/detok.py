#!/usr/bin/env python3
"""Detokenize SenseVoice CTC token ids -> text via the SentencePiece bpe model.
Usage: python detok.py <bpe.model> <ids.txt>   (ids.txt = space-separated ints)
"""
import sys
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
ids = [int(x) for x in open(sys.argv[2]).read().split()]
print(sp.decode(ids))
