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

wkv5_cuda = None


def load_rwkv_kernel(
    HEAD_SIZE: int = 64,
    RWKV_CTXLEN: int = 512,
):
    from torch.utils.cpp_extension import load

    global wkv5_cuda

    if wkv5_cuda is not None:
        return

    absolute_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(absolute_file_path)

    wkv5_cuda = load(
        name="wkv5",
        sources=[f"{cur_dir}/cuda/wkv5_op.cpp", f"{cur_dir}/cuda/wkv5_cuda.cu"],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
        ],
    )


# dtype = torch.float
dtype = torch.bfloat16


class WKV_5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            # assert r.dtype == torch.bfloat16
            # assert k.dtype == torch.bfloat16
            # assert v.dtype == torch.bfloat16
            # assert w.dtype == torch.bfloat16
            # assert u.dtype == torch.bfloat16
            # assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            eew = (torch.exp(ew)).contiguous()
            ctx.save_for_backward(r, k, v, eew, ew, u)
            y = torch.empty(
                (B, T, C),
                device=r.device,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, eew, ew, u = ctx.saved_tensors
            gr = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            gw = torch.empty(
                (B, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            gu = torch.empty(
                (B, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.bfloat16,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
            gw = torch.sum(gw, 0).view(H, C // H)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
    return WKV_5.apply(B, T, C, H, r, k, v, w, u)


class RWKV_Tmix_x052(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        load_rwkv_kernel(args.head_size_a, args.ctx_len)
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        # assert HEAD_SIZE == self.head_size  # change HEAD_SIZE to match args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x)  # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)

        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x, **kwargs):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa)

        return self.jit_func_2(x, g)


#
# class RWKV_Tmix_x060(MyModule):
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args
#
#         load_rwkv_kernel(args.head_size_a, args.ctx_len)
#
#         self.layer_id = layer_id
#
#         self.head_size = args.head_size_a
#         self.n_head = args.dim_att // self.head_size
#         assert args.dim_att % self.n_head == 0
#
#         with torch.no_grad():
#             ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
#             ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
#             ddd = torch.ones(1, 1, args.n_embd)
#             for i in range(args.n_embd):
#                 ddd[0, 0, i] = i / args.n_embd
#
#             # fancy time_mix
#             self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
#             self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
#             self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
#             self.time_maa_v = nn.Parameter(
#                 1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
#             )
#             self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
#             self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
#
#             D_MIX_LORA = 32  # generate TIME_MIX for w,k,v,r,g
#             self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 5))
#             self.time_maa_w2 = nn.Parameter(
#                 torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01)
#             )
#
#             # fancy time_decay
#             decay_speed = torch.ones(args.dim_att)
#             for n in range(args.dim_att):
#                 decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
#             self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))
#
#             D_DECAY_LORA = 64
#             self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
#             self.time_decay_w2 = nn.Parameter(
#                 torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01)
#             )
#
#             tmp = torch.zeros(args.dim_att)
#             for n in range(args.dim_att):
#                 zigzag = ((n + 1) % 3 - 1) * 0.1
#                 tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag
#
#             self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
#
#         self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
#         self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
#         self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
#
#         self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
#         self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
#         self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
#         self.ln_x = nn.GroupNorm(
#             self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor**2)
#         )
#
#     @MyFunction
#     def jit_func(self, x):
#         B, T, C = x.size()
#
#         xx = self.time_shift(x) - x
#
#         xxx = x + xx * self.time_maa_x
#         xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
#         xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
#         mw, mk, mv, mr, mg = xxx.unbind(dim=0)
#
#         xw = x + xx * (self.time_maa_w + mw)
#         xk = x + xx * (self.time_maa_k + mk)
#         xv = x + xx * (self.time_maa_v + mv)
#         xr = x + xx * (self.time_maa_r + mr)
#         xg = x + xx * (self.time_maa_g + mg)
#
#         r = self.receptance(xr)
#         k = self.key(xk)
#         v = self.value(xv)
#         g = F.silu(self.gate(xg))
#
#         ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
#         w = self.time_decay + ww
#
#         return r, k, v, g, w
#
#     @MyFunction
#     def jit_func_2(self, x, g):
#         B, T, C = x.size()
#         x = x.view(B * T, C)
#
#         x = self.ln_x(x).view(B, T, C)
#         x = self.output(x * g)
#         return x
#
#     def forward(self, x):
#         B, T, C = x.size()
#         H = self.n_head
#
#         r, k, v, g, w = self.jit_func(x)
#         x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)
#
#         return self.jit_func_2(x, g)


class RWKV_CMix_x052(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


# class RWKV_CMix_x060(MyModule):
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args
#         self.layer_id = layer_id
#         self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
#
#         with torch.no_grad():  # fancy init of time_mix
#             ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
#             ddd = torch.ones(1, 1, args.n_embd)
#             for i in range(args.n_embd):
#                 ddd[0, 0, i] = i / args.n_embd
#             self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
#             self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
#
#         self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
#         self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
#         self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)
#
#     @MyFunction
#     def forward(self, x):
#         xx = self.time_shift(x) - x
#         xk = x + xx * self.time_maa_k
#         xr = x + xx * self.time_maa_r
#
#         k = self.key(xk)
#         k = torch.relu(k) ** 2
#         kv = self.value(k)
#         return torch.sigmoid(self.receptance(xr)) * kv


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

        self.att = RWKV_Tmix_x052(args, layer_id)
        self.ln2 = None
        self.ffn = None
        if args.get("use_rwkv_ffn", True):
            self.ln2 = nn.LayerNorm(args.n_embd)
            self.ffn = RWKV_CMix_x052(args, layer_id)

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
            if self.ffn is not None:
                x = x + self.ffn(self.ln2(x))
        else:
            if self.ln1 is None:
                x = self.drop0(x + self.att(x))
            else:
                x = self.drop0(x + self.att(self.ln1(x)))
            if self.ffn is not None:
                x = self.drop1(x + self.ffn(self.ln2(x)))

        if args.get("datatype", "bf16") == "bf16":
            x = x.to(torch.float32)
        return x


# class RWKV(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         if not hasattr(args, "dim_att"):
#             args.dim_att = args.n_embd
#         if not hasattr(args, "dim_ffn"):
#             if "-f4" in os.environ["RWKV_MY_TESTING"]:
#                 args.dim_ffn = int((args.n_embd * 4) // 32 * 32)
#             else:
#                 args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size
#         if not hasattr(args, "tiny_att_layer"):
#             args.tiny_att_layer = -1
#         if not hasattr(args, "tiny_att_dim"):
#             args.tiny_att_dim = -1
#         assert args.n_embd % 32 == 0
#         assert args.dim_att % 32 == 0
#         assert args.dim_ffn % 32 == 0
#
#         self.emb = nn.Embedding(args.vocab_size, args.n_embd)
#
#         self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
#
#         self.ln_out = nn.LayerNorm(args.n_embd)
#         self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
#
#         if args.dropout > 0:
#             self.drop0 = nn.Dropout(p=args.dropout)
#
#     def forward(self, idx):
#         args = self.args
#         B, T = idx.size()
#         assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."
#
#         x = self.emb(idx)
#         x_emb = x
#
#         if args.dropout > 0:
#             x = self.drop0(x)
#         if args.tiny_att_dim > 0:
#             for block in self.blocks:
#                 if args.grad_cp == 1:
#                     x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
#                 else:
#                     x = block(x, x_emb)
#         else:
#             for block in self.blocks:
#                 if args.grad_cp == 1:
#                     x = deepspeed.checkpointing.checkpoint(block, x)
#                 else:
#                     x = block(x)
#
#         x = self.ln_out(x)
#
#         if args.head_qk > 0:
#             q = self.head_q(x)[:, :T, :]
#             k = self.head_k(x)[:, :T, :]
#             c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
#             c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)
#
#             if "32" in os.environ["RWKV_FLOAT_MODE"]:
#                 c = c @ F.one_hot(idx, num_classes=args.vocab_size)
#             elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
#                 c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
#             elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
#                 c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()
#
#             x = self.head(x) + c
#         else:
#             x = self.head(x)
#
#         return x
