# Created on 2024-01-01
# Author: GuAn Zhu

import torch
import numpy as np
import math
import torch.nn.functional as F


class LFR_CMVN_PE(torch.nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        istd: torch.Tensor,
        m: int = 7,
        n: int = 6,
        max_len: int = 5000,
        encoder_input_size: int = 560,
        encoder_output_size: int = 512,
    ):
        super().__init__()

        # LRF
        self.m = m
        self.n = n
        self.subsample = (m - 1) // 2

        # CMVN
        assert mean.shape == istd.shape
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

        # PE
        self.encoder_input_size = encoder_input_size
        self.encoder_output_size = encoder_output_size
        self.max_len = max_len
        self.pe = torch.zeros(self.max_len, self.encoder_input_size)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange((self.encoder_input_size / 2), dtype=torch.float32)
            * -(math.log(10000.0) / (self.encoder_input_size / 2 - 1))
        )
        self.pe[:, 0::1] = torch.cat(
            (torch.sin(position * div_term), torch.cos(position * div_term)), dim=1
        )

    def forward(self, x, cache, offset):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        B, _, D = x.size()
        x = x.unfold(1, self.m, step=self.n).transpose(2, 3)
        x = x.view(B, -1, D * self.m)

        x = (x + self.mean) * self.istd
        x = x * (self.encoder_output_size**0.5)

        index = offset + torch.arange(1, x.size(1) + 1).to(dtype=torch.int32)
        pos_emb = F.embedding(index, self.pe)  # B X T X d_model
        r_cache = x + pos_emb

        r_x = torch.cat((cache, r_cache), dim=1)
        r_offset = offset + x.size(1)
        r_x_len = torch.ones((B, 1), dtype=torch.int32) * r_x.size(1)

        return r_x, r_x_len, r_cache, r_offset


def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3 : (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3 : (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue

    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    means = torch.from_numpy(means)
    vars = torch.from_numpy(vars)
    return means, vars


if __name__ == "__main__":
    means, vars = load_cmvn("am.mvn")
    means = torch.tile(means, (10, 1))
    vars = torch.tile(vars, (10, 1))

    model = LFR_CMVN_PE(means, vars)
    model.eval()

    all_names = [
        "chunk_xs",
        "cache",
        "offset",
        "chunk_xs_out",
        "chunk_xs_out_len",
        "r_cache",
        "r_offset",
    ]
    dynamic_axes = {}

    for name in all_names:
        dynamic_axes[name] = {0: "B"}

    input_data1 = torch.randn(4, 61, 80).to(torch.float32)
    input_data2 = torch.randn(4, 10, 560).to(torch.float32)
    input_data3 = torch.randn(4, 1).to(torch.int32)

    onnx_path = "./1/lfr_cmvn_pe.onnx"
    torch.onnx.export(
        model,
        (input_data1, input_data2, input_data3),
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["chunk_xs", "cache", "offset"],
        output_names=["chunk_xs_out", "chunk_xs_out_len", "r_cache", "r_offset"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    print("export to onnx model succeed!")
