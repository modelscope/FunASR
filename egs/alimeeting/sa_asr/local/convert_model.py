import codecs
import pdb
import sys
import torch

char1 = sys.argv[1]
char2 = sys.argv[2]
model1 = torch.load(sys.argv[3], map_location='cpu')
model2_path = sys.argv[4]

d_new = model1
char1_list = []
map_list = []


with codecs.open(char1) as f:
    for line in f.readlines():
        char1_list.append(line.strip())

with codecs.open(char2) as f:
    for line in f.readlines():
        map_list.append(char1_list.index(line.strip()))
print(map_list)

for k, v in d_new.items():
    if k == 'ctc.ctc_lo.weight' or k == 'ctc.ctc_lo.bias' or k == 'decoder.output_layer.weight' or k == 'decoder.output_layer.bias' or k == 'decoder.embed.0.weight':
        d_new[k] = v[map_list]
    
torch.save(d_new, model2_path)
