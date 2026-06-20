#!/usr/bin/env python3
"""Export FSMN-VAD (encoder + CMVN) to GGUF for the ggml C++ runtime."""
import argparse, os, re
import numpy as np, torch, gguf

def parse_mvn(path):
    b=[np.array([float(x) for x in m.split()],np.float32) for m in re.findall(r"\[([^\]]*)\]",open(path).read())]
    v=[x for x in b if x.size>1]; return v[0], v[1]   # shift, scale (both 400-dim)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_pt",required=True); ap.add_argument("--mvn",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    sd=torch.load(a.model_pt,map_location="cpu"); sd=sd.get("state_dict",sd)
    w=gguf.GGUFWriter(a.out,"fsmn-vad")
    w.add_uint32("vad.input_dim",400); w.add_uint32("vad.input_affine_dim",140)
    w.add_uint32("vad.linear_dim",250); w.add_uint32("vad.proj_dim",128)
    w.add_uint32("vad.fsmn_layers",4); w.add_uint32("vad.lorder",20)
    w.add_uint32("vad.output_affine_dim",140); w.add_uint32("vad.output_dim",248)
    w.add_uint32("vad.n_mels",80); w.add_uint32("vad.lfr_m",5); w.add_uint32("vad.lfr_n",1)
    shift,scale=parse_mvn(a.mvn); w.add_tensor("cmvn.shift",shift); w.add_tensor("cmvn.scale",scale)
    n=0
    for k,v in sd.items():
        if not k.startswith("encoder."): continue
        arr=v.detach().to(torch.float32).contiguous().numpy()
        if k.endswith("conv_left.weight"):   # (C,1,lorder,1) -> (lorder,C) tap-major
            arr=np.ascontiguousarray(arr[:,0,:,0].T)
        w.add_tensor(k,arr); n+=1
    print(f"writing {n} tensors (+cmvn) to {a.out}")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"done: {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")

if __name__=="__main__": main()
