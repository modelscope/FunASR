#!/usr/bin/python3

import pickle
import numpy as np
import sys
import math
import torch


def alignment(a, size):
    print('a.shape is {}'.format(a.shape))
    a = a.reshape((-1))
    align_size = int(math.ceil(a.size / size) * size)
    return np.pad(a, (0, align_size - a.size), 'constant', constant_values=(0, 0))


def encoder(prefix, model, wfid):
    a = model['{}.feed_forward.w_1.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.feed_forward.w_1.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.feed_forward.w_2.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.feed_forward.w_2.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.norm1.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.norm1.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.norm2.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.norm2.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.self_attn.fsmn_block.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.self_attn.linear_out.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.self_attn.linear_out.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    scale = 128**(-0.5)
    a = model['{}.self_attn.linear_q_k_v.bias'.format(prefix)]
    a[0:512][:] = scale * a[0:512][:]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())
    a = model['{}.self_attn.linear_q_k_v.weight'.format(prefix)]
    a[0:512] = scale * a[0:512]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())


def decoder(prefix, model, wfid):
    a = model['{}.feed_forward.norm.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.feed_forward.norm.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.feed_forward.w_1.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.feed_forward.w_1.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.feed_forward.w_2.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.norm1.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.norm1.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.norm2.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.norm2.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.norm3.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.norm3.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.self_attn.fsmn_block.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.src_attn.linear_k_v.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.src_attn.linear_k_v.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.src_attn.linear_out.bias'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.src_attn.linear_out.weight'.format(prefix)]
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    scale = 128**(-0.5)
    a = model['{}.src_attn.linear_q.bias'.format(prefix)]
    a = scale * a
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())

    a = model['{}.src_attn.linear_q.weight'.format(prefix)]
    a = scale * a
    a = alignment(a.numpy(), 32.0)
    wfid.write(a.tobytes())


filename = sys.argv[1]
model = torch.load(filename, map_location='cpu')
wfid = open('paraformer_online.bin', 'wb')
encoder('encoder.encoders0.0', model, wfid)
for i in range(0, 49):
    encode_prefix = 'encoder.encoders.{}'.format(i)
    print('i is {}'.format(encode_prefix))
    encoder(encode_prefix, model, wfid)

a = model['encoder.after_norm.bias']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['encoder.after_norm.weight']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['predictor.cif_conv1d.bias']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['predictor.cif_conv1d.weight']
a = torch.permute(a, (1, 0, 2))
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['predictor.cif_output.bias']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['predictor.cif_output.weight']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

print(' decoder param !!!!!')

for i in range(0, 16):
    decode_prefix = 'decoder.decoders.{}'.format(i)
    print('i is {}'.format(decode_prefix))
    decoder(decode_prefix, model, wfid)


a = model['decoder.decoders3.0.feed_forward.norm.bias']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.decoders3.0.feed_forward.norm.weight']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.decoders3.0.feed_forward.w_1.bias']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.decoders3.0.feed_forward.w_1.weight']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.decoders3.0.feed_forward.w_2.weight']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.decoders3.0.norm1.bias']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.decoders3.0.norm1.weight']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.after_norm.bias']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.after_norm.weight']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.output_layer.bias']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())

a = model['decoder.output_layer.weight']
a = alignment(a.numpy(), 32.0)
wfid.write(a.tobytes())


wfid.close()
