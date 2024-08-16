import sys
import wave
import json
import base64

if __name__ == "__main__":
    wav_path = sys.argv[1]
    chunk_size = [int(x) for x in sys.argv[2].split(",")]
    chunk_interval = int(sys.argv[3])

    with wave.open(wav_path, "rb") as wav_file:
        params = wav_file.getparams()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_bytes = bytes(frames)

    stride = int(60 * chunk_size[1] / chunk_interval / 1000 * sample_rate * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1

    result = {
        "sample_rate": sample_rate,
        "stride": stride,
        "chunk_num": chunk_num,
        "audio_bytes": base64.b64encode(audio_bytes).decode('utf-8')
    }

    print(json.dumps(result))