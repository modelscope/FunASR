import json, math, random, sys, time, shutil, os, string, re, fileinput
import numpy as np

"""
How to use:

python make_data.py demo.jsonl 3 4096

This will:
==> shuffle & duplicate demo.jsonl (for 3 epochs, good for finetuning) note: this will be very slow for large jsonl and we need more efficient code.
==> load jsonl and tokenize
==> save as demo.bin & demo.idx
==> compute "magic_prime" for ctxlen 4096

Example:

Assume your source jsonl is:
{"text":"aa"}
{"text":"bb"}
{"text":"cc"}
{"text":"dd"}

The final binidx will be like (here "/" means end_of_doc, which is actually token [0]):
bb/aa/dd/cc/dd/aa/bb/cc/dd/bb/cc/aa/

where the data is repeated 3 times (each time with different shuffle)
"""

########################################################################################################
# MMapIndexedDatasetBuilder
########################################################################################################

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
from src.binidx import MMapIndexedDataset
def index_file_path(prefix_path):
    return prefix_path + ".idx"
def data_file_path(prefix_path):
    return prefix_path + ".bin"
class MMapIndexedDatasetBuilder(object):
    def __init__(self, out_file, dtype=np.uint16):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]
    def add_item(self, np_array):
        assert np_array.dtype == self._dtype
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)
    def end_document(self):
        self._doc_idx.append(len(self._sizes))
    def finalize(self, index_file):
        self._data_file.close()
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
cnt = 0
def add_raw(raw):
    global builder, cnt
    out = tokenizer.encode(raw)
    if tokenizer.decode(out) != raw:
        print("ERROR" * 100)
        exit(0)
    out.append(0)  # [0] = end_of_doc for rwkv tokenizer
    builder.add_item(np.array(out, dtype=np.uint16))
    builder.end_document()
    if cnt % 500 == 0:
        print(cnt, end=" ", flush=True)
    cnt += 1
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

########################################################################################################

N_EPOCH = int(sys.argv[2].strip())
IN_FILE = sys.argv[1].strip()
OUT_NAME = os.path.splitext(os.path.basename(IN_FILE))[0]
CTX_LEN = int(sys.argv[3].strip())
TEMP_FILE = "make_data_temp.jsonl"

print(f"### Convert {IN_FILE} to {OUT_NAME}.bin/idx...")

with open(IN_FILE, "r", encoding="utf-8") as file:
    non_empty_lines = [line.strip() for line in file if line.strip()]

print(f"### Found {len(non_empty_lines)} non-empty lines in {IN_FILE}")

file = open(TEMP_FILE, "w", encoding="utf-8")
for i in range(N_EPOCH):
    print(f"Shuffle: {i+1} out of {N_EPOCH}")
    random.shuffle(non_empty_lines)
    for entry in non_empty_lines:
        file.write(entry + "\n")
file.close()

########################################################################################################

print("### Building binidx...")

builder = MMapIndexedDatasetBuilder(f"{OUT_NAME}.bin")
with fileinput.input(TEMP_FILE, encoding="utf-8") as ffff:
    for line in ffff:
        x = json.loads(line)["text"]
        add_raw(x)
builder.finalize((f"{OUT_NAME}.idx"))
print("done")

print("### Verifying result...")
data = MMapIndexedDataset(OUT_NAME)
data_len = len(data)
data_size = len(data._bin_buffer) // data._index._dtype_size

TODO = [0, data_len - 1]
PREVIEW_LIMIT = 100
for idx in TODO:
    ptr, size = data._index[idx]
    dix = data.get(idx=idx, offset=0, length=size).astype(int)
    print("-" * 70 + f"[{OUT_NAME} idx {idx} sz {size}]")
    assert dix[-1] == 0
    dix = dix[:-1]
    if len(dix) > PREVIEW_LIMIT:
        try:
            print(tokenizer.decode(dix[:PREVIEW_LIMIT]))
        except:
            try:
                print(tokenizer.decode(dix[: PREVIEW_LIMIT + 1]))
            except:
                print(tokenizer.decode(dix[: PREVIEW_LIMIT + 2]))
        print("· " * 30)
        try:  # avoid utf-8 bug
            print(tokenizer.decode(dix[-PREVIEW_LIMIT:]))
        except:
            try:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT - 1 :]))
            except:
                print(tokenizer.decode(dix[-PREVIEW_LIMIT - 2 :]))
    else:
        print(tokenizer.decode(dix))

print(f"{'-'*80}\n### Final {OUT_NAME}.bin/idx has {data_size} tokens, {data_len} items. Dtype {data._index.dtype}")

if data_size >= CTX_LEN * 3:
    n_chunk = int(data_size // CTX_LEN) - 1
    for i in range(n_chunk, 0, -1):
        if i % 3 == 2:
            if is_prime(i):
                print(f"\n### magic_prime = {i} (for ctxlen {CTX_LEN})\n")
                exit(0)
