import os
import sys
import soundfile as sf
from funasr.utils.misc import load_scp_as_list


if __name__ == '__main__':
    wav_scp = sys.argv[1]
    out_file = sys.argv[2]
    frame_shift = 0.01

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    out_file = open(out_file, "wt")
    for uttid, wav_path in load_scp_as_list(wav_scp):
        wav, sr = sf.read(wav_path)
        num_frame = wav.shape[0] // int(sr * frame_shift)
        out_file.write(f"{uttid} {num_frame}\n")
        out_file.flush()

    out_file.close()
