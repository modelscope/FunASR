from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np
import sys
import os
import torch
from kaldiio import WriteHelper
import re

text_file_json = sys.argv[1]
out_ark = sys.argv[2]
out_scp = sys.argv[3]
out_shape = sys.argv[4]
device = int(sys.argv[5])
model_path = sys.argv[6]

model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
extractor = pipeline(task="feature-extraction", model=model, tokenizer=tokenizer, device=device)

with open(text_file_json, "r") as f:
    js = f.readlines()


f_shape = open(out_shape, "w")
with WriteHelper("ark,scp:{},{}".format(out_ark, out_scp)) as writer:
    with torch.no_grad():
        for idx, line in enumerate(js):
            id, tokens = line.strip().split(" ", 1)
            tokens = re.sub(" ", "", tokens.strip())
            tokens = " ".join([j for j in tokens])
            token_num = len(tokens.split(" "))
            outputs = extractor(tokens)
            outputs = np.array(outputs)
            embeds = outputs[0, 1:-1, :]

            token_num_embeds, dim = embeds.shape
            if token_num == token_num_embeds:
                writer(id, embeds)
                shape_line = "{} {},{}\n".format(id, token_num_embeds, dim)
                f_shape.write(shape_line)
            else:
                print(
                    "{}, size has changed, {}, {}, {}".format(
                        id, token_num, token_num_embeds, tokens
                    )
                )


f_shape.close()
