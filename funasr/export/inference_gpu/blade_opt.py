import torch
import torch_blade
import pdb

import os
# os.environ["TORCH_BLADE_DEBUG_LOG"] = "true"
# os.environ["TORCH_BLADE_MHLO_DEBUG_LOG"] = "true"
# os.environ["PYTORCH_JIT_LOG_LEVEL"] = ">>>mhlo_conversion"
# os.environ["TORCH_BLADE_DEBUG_ENABLE_ERROR_FALLBACK"] = "true"
# os.environ["TORCH_DISC_ENABLE_REPLAY_ON_CLUSTER"] = "true"
# os.environ["TORCH_BLADE_DEBUG_ENABLE_ERROR_FALLBACK"] = "true"


# inputs = torch.load('../encoders0_inputs.pth')
# module = torch.jit.load('../model.encoders0.pt')
# module = torch.jit.load('../model.encoders0.fixfp16.10.pt')

# inputs = torch.load('../encoders_inputs.pth')
# module = torch.jit.load('../model.encoders.pt')
# module = torch.jit.load('../model.encoders.fixfp16.10.pt')

"""
inputs = torch.load('../encoders36_inputs.pth')
import os
fp16 = float(os.environ.get('FP16', False))
if fp16:
    inputs = (inputs[0] / fp16, inputs[1])
    _istuple = lambda x: isinstance(x, tuple)
    _mod = lambda x, func: tuple([func(i) for i in x]) if _istuple(x) else func(x)
    _half = lambda x: x.half()
    inputs = tuple([_mod(d, _half) for d in inputs])

module = torch.jit.load('../model.encoders36.fixfp16.10.pt')
module.half()
"""

inputs = torch.load('data/test_data_1x93x560.pt')
# module = torch.jit.load('../model.encoder.pt')
module = torch.jit.load('../model.encoder.fixfp16.10.pt')

# inputs = torch.load('../decoder_inputs.pth')
# module = torch.jit.load('../model.decoder.pt')

module = module.eval()
inputs = tuple([tuple([j.cuda() for j in i]) if isinstance(i, tuple) else i.cuda() for i in inputs])

# pdb.set_trace()
with torch.no_grad():
    module(*inputs)

fp16 = True
if fp16:
    torch_config = torch_blade.config.Config()
    torch_config.enable_fp16 = True
    with torch.no_grad():
        with torch_config:
            blade_module = torch_blade.optimize(module, allow_tracing=True, model_inputs=inputs)
else:
    with torch.no_grad():
        blade_module = torch_blade.optimize(module, allow_tracing=True, model_inputs=inputs)

# run the optimized module
with torch.no_grad():
    blade_module(*inputs)

torch.jit.save(blade_module, 'blade.fp16.pt' if fp16 else 'blade.pt')
