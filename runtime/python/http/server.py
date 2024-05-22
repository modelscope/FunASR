import argparse
import logging
import os
import uuid

import aiofiles
import ffmpeg
import uvicorn
from fastapi import FastAPI, File, UploadFile
from modelscope.utils.logger import get_logger

from funasr import AutoModel

logger = get_logger(log_level=logging.INFO)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=8000, required=False, help="server port")
parser.add_argument(
    "--asr_model",
    type=str,
    default="paraformer-zh",
    help="asr model from https://github.com/alibaba-damo-academy/FunASR?tab=readme-ov-file#model-zoo",
)
parser.add_argument("--asr_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--vad_model",
    type=str,
    default="fsmn-vad",
    help="vad model from https://github.com/alibaba-damo-academy/FunASR?tab=readme-ov-file#model-zoo",
)
parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--punc_model",
    type=str,
    default="ct-punc-c",
    help="model from https://github.com/alibaba-damo-academy/FunASR?tab=readme-ov-file#model-zoo",
)
parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")
parser.add_argument(
    "--hotword_path",
    type=str,
    default="hotwords.txt",
    help="hot word txt path, only the hot word model works",
)
parser.add_argument("--certfile", type=str, default=None, required=False, help="certfile for ssl")
parser.add_argument("--keyfile", type=str, default=None, required=False, help="keyfile for ssl")
parser.add_argument("--temp_dir", type=str, default="temp_dir/", required=False, help="temp dir")
args = parser.parse_args()
logger.info("-----------  Configuration Arguments -----------")
for arg, value in vars(args).items():
    logger.info("%s: %s" % (arg, value))
logger.info("------------------------------------------------")

os.makedirs(args.temp_dir, exist_ok=True)

logger.info("model loading")
# load funasr model
model = AutoModel(
    model=args.asr_model,
    model_revision=args.asr_model_revision,
    vad_model=args.vad_model,
    vad_model_revision=args.vad_model_revision,
    punc_model=args.punc_model,
    punc_model_revision=args.punc_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)
logger.info("loaded models!")

app = FastAPI(title="FunASR")

param_dict = {"sentence_timestamp": True, "batch_size_s": 300}
if args.hotword_path is not None and os.path.exists(args.hotword_path):
    with open(args.hotword_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    hotword = " ".join(lines)
    logger.info(f"热词：{hotword}")
    param_dict["hotword"] = hotword


@app.post("/recognition")
async def api_recognition(audio: UploadFile = File(..., description="audio file")):
    suffix = audio.filename.split(".")[-1]
    audio_path = f"{args.temp_dir}/{str(uuid.uuid1())}.{suffix}"
    async with aiofiles.open(audio_path, "wb") as out_file:
        content = await audio.read()
        await out_file.write(content)
    try:
        audio_bytes, _ = (
            ffmpeg.input(audio_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        logger.error(f"读取音频文件发生错误，错误信息：{e}")
        return {"msg": "读取音频文件发生错误", "code": 1}
    rec_results = model.generate(input=audio_bytes, is_final=True, **param_dict)
    # 结果为空
    if len(rec_results) == 0:
        return {"text": "", "sentences": [], "code": 0}
    elif len(rec_results) == 1:
        # 解析识别结果
        rec_result = rec_results[0]
        text = rec_result["text"]
        sentences = []
        for sentence in rec_result["sentence_info"]:
            # 每句话的时间戳
            sentences.append(
                {"text": sentence["text"], "start": sentence["start"], "end": sentence["end"]}
            )
        ret = {"text": text, "sentences": sentences, "code": 0}
        logger.info(f"识别结果：{ret}")
        return ret
    else:
        logger.info(f"识别结果：{rec_results}")
        return {"msg": "未知错误", "code": -1}


if __name__ == "__main__":
    uvicorn.run(
        app, host=args.host, port=args.port, ssl_keyfile=args.keyfile, ssl_certfile=args.certfile
    )
