import torch
import numpy as np

if __name__ == '__main__':
	onnx_path = "/nfs/zhifu.gzf/export/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.torchscripts"
	loaded = torch.jit.load(onnx_path)
	
	x = torch.rand([2, 21, 560])
	x_len = torch.IntTensor([6, 21])
	res = loaded(x, x_len)
	print(res[0].size(), res[1])
	
	x = torch.rand([5, 50, 560])
	x_len = torch.IntTensor([6, 21, 10, 30, 50])
	res = loaded(x, x_len)
	print(res[0].size(), res[1])
	