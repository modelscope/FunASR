from funasr_onnx import CamPlusPlus
from pathlib import Path
import torch
import numpy as np

def cosine_similarity(u, v):
    u = u.flatten()
    v = v.flatten()
    dot_product = np.dot(u, v) 
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v) 
    if norm_u == 0 or norm_v == 0:
        return 0    
    similarity = dot_product / (norm_u * norm_v)    
    return similarity

model_dir = "iic/speech_campplus_sv_zh-cn_16k-common"
wav_path1 = "{}/.cache/modelscope/hub/iic/speech_campplus_sv_zh-cn_16k-common/examples/speaker1_a_cn_16k.wav".format(
    Path.home()
)
wav_path2 = "{}/.cache/modelscope/hub/iic/speech_campplus_sv_zh-cn_16k-common/examples/speaker1_b_cn_16k.wav".format(
    Path.home()
)
model = CamPlusPlus(model_dir,quantize=True)
embedding1 = model(wav_path1)
embedding2 = model(wav_path2)

print("Computing the similarity score...")
scores=cosine_similarity(embedding1[0],embedding2[0])
print("scores={:.4f}".format(scores))
