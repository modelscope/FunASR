import torch
from torch.nn.utils.rnn import pad_sequence

def slice_padding_fbank(speech, speech_lengths, vad_segments):
	speech_list = []
	speech_lengths_list = []
	for i, segment in enumerate(vad_segments):
		
		bed_idx = int(segment[0][0]*16)
		end_idx = min(int(segment[0][1]*16), speech_lengths[0])
		speech_i = speech[0, bed_idx: end_idx]
		speech_lengths_i = end_idx-bed_idx
		speech_list.append(speech_i)
		speech_lengths_list.append(speech_lengths_i)
	feats_pad = pad_sequence(speech_list, batch_first=True, padding_value=0.0)
	speech_lengths_pad = torch.Tensor(speech_lengths_list).int()
	return feats_pad, speech_lengths_pad
	
