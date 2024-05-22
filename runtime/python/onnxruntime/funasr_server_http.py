import argparse
import base64
import io
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Body

app = FastAPI()

from funasr_onnx import Paraformer

model_dir = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
model = Paraformer(model_dir, batch_size=1, quantize=True)


async def recognition_onnx(waveform):
    result = model(waveform)[0]["preds"][0]
    return result


@app.post("/api/asr")
async def asr(item: dict = Body(...)):
    try:
        audio_bytes = base64.b64decode(bytes(item["wav_base64"], "utf-8"))
        waveform, _ = sf.read(io.BytesIO(audio_bytes))
        result = await recognition_onnx(waveform)
        ret = {"results": result, "code": 0}
    except:
        print("请求出错，这里是处理出错的")
        ret = {"results": "", "code": 1}
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Service")
    parser.add_argument("--listen", default="0.0.0.0", type=str, help="the network to listen")
    parser.add_argument("--port", default=8888, type=int, help="the port to listen")
    args = parser.parse_args()

    print("start...")
    print("server on:", args)

    uvicorn.run(app, host=args.listen, port=args.port)
