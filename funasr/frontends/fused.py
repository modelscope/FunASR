from funasr.frontends.default import DefaultFrontend
from funasr.frontends.s3prl import S3prlFrontend
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class FusedFrontends(nn.Module):
    def __init__(self, frontends=None, align_method="linear_projection", proj_dim=100, fs=16000):

        super().__init__()
        self.align_method = align_method  # fusing method : linear_projection only for now
        self.proj_dim = proj_dim  # dim of the projection done on each frontend
        self.frontends = []  # list of the frontends to combine

        for i, frontend in enumerate(frontends):
            frontend_type = frontend["frontend_type"]
            if frontend_type == "default":
                n_mels, fs, n_fft, win_length, hop_length = (
                    frontend.get("n_mels", 80),
                    fs,
                    frontend.get("n_fft", 512),
                    frontend.get("win_length"),
                    frontend.get("hop_length", 128),
                )
                window, center, normalized, onesided = (
                    frontend.get("window", "hann"),
                    frontend.get("center", True),
                    frontend.get("normalized", False),
                    frontend.get("onesided", True),
                )
                fmin, fmax, htk, apply_stft = (
                    frontend.get("fmin", None),
                    frontend.get("fmax", None),
                    frontend.get("htk", False),
                    frontend.get("apply_stft", True),
                )

                self.frontends.append(
                    DefaultFrontend(
                        n_mels=n_mels,
                        n_fft=n_fft,
                        fs=fs,
                        win_length=win_length,
                        hop_length=hop_length,
                        window=window,
                        center=center,
                        normalized=normalized,
                        onesided=onesided,
                        fmin=fmin,
                        fmax=fmax,
                        htk=htk,
                        apply_stft=apply_stft,
                    )
                )
            elif frontend_type == "s3prl":
                frontend_conf, download_dir, multilayer_feature = (
                    frontend.get("frontend_conf"),
                    frontend.get("download_dir"),
                    frontend.get("multilayer_feature"),
                )
                self.frontends.append(
                    S3prlFrontend(
                        fs=fs,
                        frontend_conf=frontend_conf,
                        download_dir=download_dir,
                        multilayer_feature=multilayer_feature,
                    )
                )

            else:
                raise NotImplementedError  # frontends are only default or s3prl

        self.frontends = torch.nn.ModuleList(self.frontends)

        self.gcd = np.gcd.reduce([frontend.hop_length for frontend in self.frontends])
        self.factors = [frontend.hop_length // self.gcd for frontend in self.frontends]
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
        if self.align_method == "linear_projection":
            self.projection_layers = [
                torch.nn.Linear(
                    in_features=frontend.output_size(),
                    out_features=self.factors[i] * self.proj_dim,
                )
                for i, frontend in enumerate(self.frontends)
            ]
            self.projection_layers = torch.nn.ModuleList(self.projection_layers)
            self.projection_layers = self.projection_layers.to(torch.device(dev))

    def output_size(self) -> int:
        return len(self.frontends) * self.proj_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # step 0 : get all frontends features
        self.feats = []
        for frontend in self.frontends:
            with torch.no_grad():
                input_feats, feats_lens = frontend.forward(input, input_lengths)
            self.feats.append([input_feats, feats_lens])

        if self.align_method == "linear_projection":  # TODO(Dan): to add other align methods

            # first step : projections
            self.feats_proj = []
            for i, frontend in enumerate(self.frontends):
                input_feats = self.feats[i][0]
                self.feats_proj.append(self.projection_layers[i](input_feats))

            # 2nd step : reshape
            self.feats_reshaped = []
            for i, frontend in enumerate(self.frontends):
                input_feats_proj = self.feats_proj[i]
                bs, nf, dim = input_feats_proj.shape
                input_feats_reshaped = torch.reshape(
                    input_feats_proj, (bs, nf * self.factors[i], dim // self.factors[i])
                )
                self.feats_reshaped.append(input_feats_reshaped)

            # 3rd step : drop the few last frames
            m = min([x.shape[1] for x in self.feats_reshaped])
            self.feats_final = [x[:, :m, :] for x in self.feats_reshaped]

            input_feats = torch.cat(
                self.feats_final, dim=-1
            )  # change the input size of the preencoder : proj_dim * n_frontends
            feats_lens = torch.ones_like(self.feats[0][1]) * (m)

        else:
            raise NotImplementedError

        return input_feats, feats_lens
