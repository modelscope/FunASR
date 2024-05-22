import os
import json
import torch
import logging
import concurrent.futures
import librosa
import torch.distributed as dist
from typing import Collection
import torch
import torchaudio
from torch import nn
import random
import re
import string
from funasr.tokenizer.cleaner import TextCleaner
from funasr.register import tables


@tables.register("preprocessor_classes", "TextPreprocessRemovePunctuation")
class TextPreprocessRemovePunctuation(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, text, **kwargs):
        # 定义英文标点符号
        en_punct = string.punctuation
        # 定义中文标点符号（部分常用的）
        cn_punct = "。？！，、；：“”‘’（）《》【】…—～·"
        # 合并英文和中文标点符号
        all_punct = en_punct + cn_punct
        # 创建正则表达式模式，匹配任何在all_punct中的字符
        punct_pattern = re.compile("[{}]".format(re.escape(all_punct)))
        # 使用正则表达式的sub方法替换掉这些字符
        return punct_pattern.sub("", text)
