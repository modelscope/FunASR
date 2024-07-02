import os
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
import sys
sys.path.insert(1, "/mnt/workspace/workgroup/hupao/project/FunASR")
from funasr import AutoModel
import json
import time
import numpy as np
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Response
import time
from pydantic import BaseModel
import uuid


class MessageItem(BaseModel):
    role: str
    content: str
    def __init__(self, role, content):
        self.content = content
        self.role = role
    
    def __getitem__(self, key):
        # 这允许你使用 item['content'] 的方式来访问属性
        return getattr(self, key)


## model_init
device = "cuda:0" # the device to load the model onto

ckpt_dir = "/mnt/workspace/workgroup/hupao/model/exp7_3_model/"
ckpt_id = "model.pt.ep20"
device = "cuda:0"
output_dir = os.path.join(ckpt_dir, f"inference-{ckpt_id}")

Model = AutoModel(
    model=ckpt_dir,
    init_param=f"{os.path.join(ckpt_dir, ckpt_id)}",
    output_dir=output_dir,
    device=device,
    fp16=False,
    bf16=False,
    llm_dtype="fp16",
)

model = Model.model
frontend = Model.kwargs['frontend']

model_name_or_path = "/mnt/workspace/workgroup/hupao/model/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

app = FastAPI()

## do infer
@app.get("/speech_qwen2_stream/")
async def stream_sse(query: str):
    def get_results(query: str):
        '''
        n = 0
        for i in range(10):
            yield f" {n}\n\n"
            n += 1
            time.sleep(0.5)  # 暂停100毫秒
        
        '''
        try:
            messages = json.loads(query)
            print ("load message success:" + json.dumps(messages, ensure_ascii=False))
            if not isinstance(messages, list):
                raise ValueError()
            # mock assistant（funasr model input required）
            print ("------------------11111---------------------")
            #messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<|startofspeech|>!/mnt/workspace/workgroup/hupao/project/FunASR/tests/sft.wav<|endofspeech|>", "text_content": "你抄完没有？"}]
            messages.append({"role": "assistant", "content": "hello"})
            key = str(uuid.uuid4())
            print ("------------------22222---------------------")
            inputs_embeds, contents, batch, source_ids, meta_data = model.inference_prepare([messages], None, key, tokenizer, frontend, device="cuda:0")
            model_inputs = {}
            model_inputs['inputs_embeds'] = inputs_embeds
            streamer = TextIteratorStreamer(tokenizer)
            generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=200)
            thread = Thread(target=model.llm.generate, kwargs=generation_kwargs)
            thread.start()
            print ("------------------33333---------------------")
            #return streamer
            for new_text in streamer:
                yield new_text + "\n"
                #return new_text
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid query format, must be a JSON array of objects." + query)
        
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(get_results(query), headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8091)
