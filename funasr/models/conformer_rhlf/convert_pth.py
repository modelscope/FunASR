import sys
import torch

input_path = sys.argv[1]
output_path = sys.argv[2]

m = torch.load(input_path)
new_dict = {}
for k, v in m["state_dict"].items():
    new_dict["ref_model." + k] = v

m["state_dict"].update(new_dict)

torch.save(m, output_path)