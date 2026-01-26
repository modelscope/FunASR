import numpy as np
import soundfile as sf
import torch

from model import FunASRNano
from tools.utils import load_audio


def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device=device)
    tokenizer = kwargs.get("tokenizer", None)
    m.eval()

    wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    res = m.inference(data_in=[wav_path], **kwargs)
    text = res[0][0]
    print(text)

    chunk_size = 0.72
    duration = sf.info(wav_path).duration
    cum_durations = np.arange(chunk_size, duration + chunk_size, chunk_size)
    prev_text = ""
    for idx, cum_duration in enumerate(cum_durations):
        audio, rate = load_audio(wav_path, 16000, duration=round(cum_duration, 3))
        prev_text = m.inference([torch.tensor(audio)], prev_text=prev_text, **kwargs)[0][0]["text"]
        if idx != len(cum_durations) - 1:
            prev_text = tokenizer.decode(tokenizer.encode(prev_text)[:-5]).replace("�", "")
        if prev_text:
            print(prev_text)


if __name__ == "__main__":
    main()
