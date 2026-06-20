#!/usr/bin/env python3
"""One-step FunASR -> GGUF converter for the llama.cpp runtime.

Downloads a model checkpoint from Hugging Face (or ModelScope) and exports it to a
GGUF the C++ runtime can load — no manual `export_*.py` invocation, mirroring
whisper.cpp's `convert` flow.

    python convert-funasr-to-gguf.py sensevoice                 # -> sensevoice-small.gguf
    python convert-funasr-to-gguf.py paraformer  --wtype f16    # -> paraformer-f16.gguf
    python convert-funasr-to-gguf.py fsmn-vad                   # -> fsmn-vad.gguf
    python convert-funasr-to-gguf.py nano-encoder --wtype f16   # -> funasr-encoder-f16.gguf
    python convert-funasr-to-gguf.py sensevoice --src modelscope

Fun-ASR-Nano also needs the Qwen3-0.6B LLM GGUF (a standard llama.cpp conversion of the
HF checkpoint) — see `--help` notes; this tool covers the audio encoder/adaptor half.
"""
import argparse, glob, os, subprocess, sys

# model key -> (hf repo, modelscope id, export script, needs am.mvn, default out stem)
MODELS = {
    "sensevoice":   ("FunAudioLLM/SenseVoiceSmall",
                     "iic/SenseVoiceSmall",
                     "export_sensevoice_gguf.py", True,  "sensevoice-small"),
    "paraformer":   ("funasr/paraformer-zh",
                     "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                     "export_paraformer_gguf.py", True,  "paraformer"),
    "fsmn-vad":     ("funasr/fsmn-vad",
                     "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                     "export_vad_gguf.py",        True,  "fsmn-vad"),
    "nano-encoder": ("FunAudioLLM/Fun-ASR-Nano-2512",
                     "iic/Fun-ASR-Nano",
                     "export_encoder_gguf.py",    False, "funasr-encoder"),
}

def find_script(name):
    """Locate an export_*.py relative to this file (works in every repo layout)."""
    here = os.path.dirname(os.path.abspath(__file__))
    hits = glob.glob(os.path.join(here, "**", name), recursive=True)
    if not hits:
        sys.exit(f"error: cannot find {name} next to {here}")
    return hits[0]

def download(key, src):
    hf_repo, ms_id, _, _, _ = MODELS[key]
    try:
        if src == "modelscope":
            from modelscope import snapshot_download
            return snapshot_download(ms_id)
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_repo)
    except ModuleNotFoundError as e:
        pkg = "modelscope" if src == "modelscope" else "huggingface_hub"
        sys.exit(f"error: missing {e.name} - install it with: pip install -U {pkg}")

def pick(d, *names):
    for n in names:
        hits = glob.glob(os.path.join(d, "**", n), recursive=True)
        if hits:
            return hits[0]
    return None

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=list(MODELS), help="which model to convert")
    ap.add_argument("--src", choices=["hf", "modelscope"], default="hf", help="checkpoint source")
    ap.add_argument("--wtype", choices=["f32", "f16"], default="f32",
                    help="matmul weight dtype in the GGUF (norm/bias stay f32)")
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--out", default=None, help="output filename (default: <stem>[-f16].gguf)")
    a = ap.parse_args()

    hf_repo, ms_id, script_name, needs_mvn, stem = MODELS[a.model]
    print(f"[1/3] downloading {a.model} from {a.src} "
          f"({ms_id if a.src=='modelscope' else hf_repo}) ...", flush=True)
    d = download(a.model, a.src)
    pt  = pick(d, "model.pt", "model.pb", "*.pt")
    mvn = pick(d, "am.mvn") if needs_mvn else None
    if not pt:                       sys.exit(f"error: no model.pt under {d}")
    if needs_mvn and not mvn:        sys.exit(f"error: no am.mvn under {d}")

    out = a.out or f"{stem}{'-f16' if a.wtype=='f16' else ''}.gguf"
    out = os.path.join(a.outdir, out)
    os.makedirs(a.outdir, exist_ok=True)

    cmd = [sys.executable, find_script(script_name), "--model_pt", pt, "--out", out]
    if needs_mvn: cmd += ["--mvn", mvn]
    # export_vad_gguf.py has no --wtype flag (tiny model, always f32)
    if script_name != "export_vad_gguf.py": cmd += ["--wtype", a.wtype]
    print(f"[2/3] exporting -> {out}", flush=True)
    subprocess.run(cmd, check=True)
    sz = os.path.getsize(out) / 1e6
    print(f"[3/3] done: {out} ({sz:.1f} MB)")
    if a.model == "nano-encoder":
        print("note: Fun-ASR-Nano also needs the Qwen3-0.6B LLM GGUF — convert it with "
              "llama.cpp's convert_hf_to_gguf.py on the HF checkpoint, then optionally quantize "
              "(Q8_0 recommended). Run with: llama-funasr-cli --enc <this> -m <qwen3.gguf> -a a.wav")

if __name__ == "__main__":
    main()
