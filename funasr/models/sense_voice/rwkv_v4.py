########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch

# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if "RWKV_JIT_ON" in os.environ and os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

########################################################################################################
# CUDA Kernel
########################################################################################################

wkv_cuda = None


def load_rwkv_kernel(
    HEAD_SIZE: int = 64,
    RWKV_CTXLEN: int = 512,
    T_MAX: int = 512,
):
    from torch.utils.cpp_extension import load

    global wkv_cuda

    if wkv_cuda is not None:
        return

    absolute_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(absolute_file_path)
    wkv_cuda = load(
        name="wkv",
        sources=[f"{cur_dir}/cuda/wkv_op.cpp", f"{cur_dir}/cuda/wkv_cuda.cu"],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            f"-DTmax={T_MAX}",
        ],
    )


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        # assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        if "32" in os.environ["RWKV_FLOAT_MODE"]:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device="cuda", memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if "32" in os.environ["RWKV_FLOAT_MODE"]:
            return y
        elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
            return y.half()
        elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
            return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device="cuda").contiguous()
        gu = torch.zeros((B, C), device="cuda").contiguous()
        gk = torch.zeros((B, T, C), device="cuda").contiguous()
        gv = torch.zeros((B, T, C), device="cuda").contiguous()
        if "32" in os.environ["RWKV_FLOAT_MODE"]:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        else:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        if "32" in os.environ["RWKV_FLOAT_MODE"]:
            return (None, None, None, gw, gu, gk, gv)
        elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        load_rwkv_kernel()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0

            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)

            # fancy time_mix
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method
    def jit_func(self, x):

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch,Time,Channel)

        sr, k, v = self.jit_func(x)

        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0

            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * config.n_embd
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


# class Block(nn.Module):
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args
#         self.layer_id = layer_id
#
#         self.ln1 = nn.LayerNorm(args.n_embd)
#         self.ln2 = nn.LayerNorm(args.n_embd)
#
#         if self.layer_id == 0:
#             self.ln0 = nn.LayerNorm(args.n_embd)
#
#         self.att = RWKV_Tmix_x060(args, layer_id)
#
#         self.ffn = RWKV_CMix_x060(args, layer_id)
#
#         if args.dropout > 0:
#             self.drop0 = nn.Dropout(p=args.dropout)
#             self.drop1 = nn.Dropout(p=args.dropout)
#
#     def forward(self, x, x_emb=None):
#         args = self.args
#         B, T, C = x.size()
#         if self.layer_id == 0:
#             x = self.ln0(x)
#
#         if self.args.dropout == 0:
#             if self.layer_id == 0 and args.pre_ffn > 0:
#                 x = x + self.ffnPre(self.ln1(x))
#             else:
#                 x = x + self.att(self.ln1(x))
#             x = x + self.ffn(self.ln2(x))
#         else:
#             if self.layer_id == 0 and args.pre_ffn > 0:
#                 x = self.drop0(x + self.ffnPre(self.ln1(x)))
#             else:
#                 x = self.drop0(x + self.att(self.ln1(x)))
#             x = self.drop1(x + self.ffn(self.ln2(x)))
#
#         return x


class RWKVLayer(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        if args.dim_ffn is None:
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
        self.ln0 = None
        if self.layer_id == 0 and args.get("ln0", True):
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.ln1 = None
        if args.get("ln1", True):
            self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_TimeMix(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

        # init
        if args.get("init_rwkv", True):
            print("init_rwkv")
            nn.init.orthogonal_(self.att.receptance.weight, gain=1)
            nn.init.orthogonal_(self.att.key.weight, gain=0.1)
            nn.init.orthogonal_(self.att.value.weight, gain=1)
            nn.init.orthogonal_(self.att.gate.weight, gain=0.1)
            nn.init.zeros_(self.att.output.weight)

            nn.init.orthogonal_(self.ffn.key.weight, gain=1)
            nn.init.zeros_(self.ffn.value.weight)
            nn.init.zeros_(self.ffn.receptance.weight)
            scale = ((1 + layer_id) / args.get("n_layer")) ** 0.7
            nn.init.constant_(self.ln2.weight, scale)
            if self.ln0 is not None:
                nn.init.constant_(self.ln0.weight, scale)
            if self.ln1 is not None:
                nn.init.constant_(self.ln1.weight, scale)

    def forward(self, x, x_emb=None, mask=None, **kwargs):

        args = self.args
        if args.get("datatype", "bf16") == "bf16":
            x = x.bfloat16()
        B, T, C = x.size()
        if self.layer_id == 0 and self.ln0 is not None:
            x = self.ln0(x)

        if self.args.dropout == 0:
            if self.ln1 is None:
                x = x + self.att(x)
            else:
                x = x + self.att(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            if self.ln1 is None:
                x = self.drop0(x + self.att(x))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            x = self.drop1(x + self.ffn(self.ln2(x)))

        if args.get("datatype", "bf16") == "bf16":
            x = x.to(torch.float32)
        return x


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        if not hasattr(args, "dim_ffn"):
            if "-f4" in os.environ["RWKV_MY_TESTING"]:
                args.dim_ffn = int((args.n_embd * 4) // 32 * 32)
            else:
                args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size
        if not hasattr(args, "tiny_att_layer"):
            args.tiny_att_layer = -1
        if not hasattr(args, "tiny_att_dim"):
            args.tiny_att_dim = -1
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        x_emb = x

        if args.dropout > 0:
            x = self.drop0(x)
        if args.tiny_att_dim > 0:
            for block in self.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                else:
                    x = block(x, x_emb)
        else:
            for block in self.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)

        x = self.ln_out(x)

        if args.head_qk > 0:
            q = self.head_q(x)[:, :T, :]
            k = self.head_k(x)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

            x = self.head(x) + c
        else:
            x = self.head(x)

        return x
