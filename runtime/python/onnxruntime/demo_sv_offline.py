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


model_dir = "/workspace/models/weights/camplus_sv_zh-cn-16k-common-onnx"
wav_path1 = "/home/wzp/project/FunASR/speaker1_a_cn_16k.wav".format(
    Path.home()
)
wav_path2 = "/home/wzp/project/FunASR/speaker1_b_cn_16k.wav".format(
    Path.home()
)
model = CamPlusPlus(model_dir)
embedding1 = model(wav_path1)
# embedding2 = model(wav_path2)

# # # compute similarity score
# print(' Computing the similarity score...')
# scores=cosine_similarity(embedding1[0],embedding2[0])
# print(scores)
