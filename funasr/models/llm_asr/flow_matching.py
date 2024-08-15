import logging
import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.train_utils.device_funcs import force_gatherable
import random
from funasr.models.llm_asr.mel_spectrum import (
    mel_spectrogram, power_spectrogram, mel_from_power_spectrogram
)
from torch.nn import functional as F
from funasr.models.transformer.utils.nets_utils import pad_list
from distutils.version import LooseVersion
from contextlib import contextmanager
from funasr.utils.hinter import hint_once
if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class MelSpectrumExtractor(nn.Module):
    def __init__(
            self,
            n_fft=1024,
            num_mels=80,
            sampling_rate=22050,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000,
            spec_type="mel",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.spec_type = spec_type

    def extra_repr(self):
        return f"n_fft={self.n_fft}, num_mels={self.num_mels}, sampling_rate={self.sampling_rate}, " \
               f"hop_size={self.hop_size}, win_size={self.win_size}, fmin={self.fmin}, fmax={self.fmax}"

    def forward(self, x, ilens):
        if self.spec_type == "power":
            feat = power_spectrogram(x, self.n_fft, self.num_mels, self.sampling_rate,
                                     self.hop_size, self.win_size, self.fmin, self.fmax)
        else:
            feat = mel_spectrogram(x, self.n_fft, self.num_mels, self.sampling_rate,
                                   self.hop_size, self.win_size, self.fmin, self.fmax)
        # determine olens by compare the lengths of inputs and outputs
        olens = ilens // (x.shape[1] // feat.shape[2])
        return feat.transpose(1, 2), olens

    def convert_power_to_mel(self, x, ilens):
        feat = mel_from_power_spectrogram(x, self.n_fft, self.num_mels, self.sampling_rate,
                                          self.hop_size, self.win_size, self.fmin, self.fmax)
        return feat, ilens


class QuantizerCodebook(torch.nn.Module):
    def __init__(
            self,
            num_quantizers,
            codebook_size,
            codebook_dim,
            hop_length,
            sampling_rate
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hop_size = hop_length
        self.sampling_rate = sampling_rate
        embed = torch.zeros(num_quantizers, codebook_size, codebook_dim)
        self.register_buffer("embed", embed)
        codec_index_shift = 1024 * torch.arange(32, dtype=torch.float32)[None, None, :]
        self.register_buffer("codec_index_shift", codec_index_shift)

    def save_embedding(self, file_name, dense_emb, emb_lengths):
        import kaldiio
        wav_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(file_name, file_name))
        dense_emb = dense_emb.cpu().numpy()
        for i in range(min(dense_emb.shape[0], 10)):
            wav_writer(str(i), dense_emb[i, :emb_lengths[i]])

        wav_writer.close()

    def forward(self, codec: torch.Tensor, codec_lengths, return_subs=False):
        if len(codec.shape) == 2:
            codec = codec.unsqueeze(-1)
        bz, tt, nq = codec.shape[0], codec.shape[1], codec.shape[2]
        codec_mask = ~make_pad_mask(codec_lengths, maxlen=codec.shape[1]).unsqueeze(-1).to(codec.device)
        codec = codec * codec_mask + self.codec_index_shift[:, :, :nq].long()
        codec = codec.reshape(-1, nq)
        emb = self.embed.reshape(-1, self.codebook_dim)
        codec_emb = F.embedding(codec, emb)  # (BT, Nq, D)
        dense_emb = codec_emb.sum(dim=1)
        dense_emb = dense_emb.reshape(bz, tt, self.codebook_dim)
        if return_subs:
            sub_embs = codec_emb.reshape(bz, tt, nq, self.codebook_dim) * codec_mask.unsqueeze(-2)
            return (dense_emb * codec_mask, sub_embs), codec_lengths
        return dense_emb * codec_mask, codec_lengths


class BaseDiffWithXvec(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int = 80,
            xvec_size: int = 198,
            output_type: str = "mel",
            encoder_conf: Dict = {},
            decoder_conf: Dict = {},
            mel_feat_conf: Dict = {},
            codec_conf: Dict = {},
            length_regulator_conf: Dict = None,
            prompt_conf: Dict = None,
            vocab_size: int = None,
            token_list: List = None,
            **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.encoder_conf = encoder_conf
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.token_list = token_list
        self.output_type = output_type
        self.prompt_conf = prompt_conf
        self.input_frame_rate = kwargs.get("input_frame_rate", 50)
        logging.info(f"input frame rate={self.input_frame_rate}")
        if output_type == 'mel':
            self.mel_extractor = MelSpectrumExtractor(**mel_feat_conf)
        elif output_type == 'codec':
            num_quantizers = codec_conf.get("num_quantizers", 32)
            codebook_size = codec_conf.get("codebook_size", 1024)
            codebook_dim = codec_conf.get("codebook_dim", 128)
            hop_length = codec_conf.get("hop_length", 640)
            sampling_rate = codec_conf.get("sampling_rate", 16000)
            self.quantizer_codebook = QuantizerCodebook(num_quantizers, codebook_size, codebook_dim,
                                                        hop_length, sampling_rate)
        if vocab_size is not None and vocab_size > 0:
            self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.xvec_proj = torch.nn.Linear(xvec_size, output_size)
        self.encoder = self.build_encoder()
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)

        self.decoder = self.build_decoder()

        self.length_regulator_conf = length_regulator_conf
        self.length_regulator = self.build_length_regulator()

    def build_encoder(self):
        encoder_name = self.encoder_conf.pop("name", "transformer")
        model = None
        if encoder_name == "transformer":
            from funasr.models.llm_asr.conformer_encoder import ConformerEncoder
            model = ConformerEncoder(
                **self.encoder_conf,
                input_size=self.input_size,
                use_cnn_module=False,
                macaron_style=False,
            )
        elif encoder_name == "conformer":
            from funasr.models.llm_asr.conformer_encoder import ConformerEncoder
            model = ConformerEncoder(
                **self.encoder_conf,
                input_size=self.input_size,
            )

        self.encoder_conf["name"] = encoder_name

        return model

    def build_decoder(self):
        decoder_name = self.decoder_conf.pop("name", "transformer")
        model = None

        if decoder_name == "matcha":
            from funasr.models.llm_asr.diffusion_models.flow_matching import CFM
            model = CFM(
                **self.decoder_conf,
                in_channels=self.output_size * 2,  # 2 for noise_y and mu
                out_channel=self.output_size,
                spk_emb_dim=self.output_size
            )

        self.decoder_conf["name"] = decoder_name

        return model

    def select_target_prompt(self, y, y_lengths):
        prompt_conf = self.prompt_conf
        prompt_list = []
        prompt_lengths = []
        for i, y_len in enumerate(y_lengths):
            prompt_len = random.randint(
                int(y_len * prompt_conf["prompt_with_range_ratio"][0]),
                int(y_len * prompt_conf["prompt_with_range_ratio"][1])
            )
            prompt_pos = random.randint(0, y_len - prompt_len)
            prompt_list.append(y[i, prompt_pos:prompt_pos+prompt_len])
            prompt_lengths.append(prompt_len)
        prompt = pad_list(prompt_list, 0.0)
        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.int64, device=y.device)

        if "cgf_prob" in prompt_conf and prompt_conf["cgf_prob"] > 0:
            cgf_mask = torch.rand([y.shape[0], 1, 1], dtype=torch.float32, device=y.device) < prompt_conf["cgf_prob"]
            prompt = prompt * cgf_mask
        return prompt, prompt_lengths

    def build_length_regulator(self):
        name = self.length_regulator_conf.pop("name", None)
        model = None
        if name == "upsampling":
            from funasr.models.llm_asr.diffusion_models.length_regulator import UpSamplingRegulator
            model = UpSamplingRegulator(self.output_size, self.length_regulator_conf.get("sampling_ratios"))
        elif name == "downsampling":
            from funasr.models.llm_asr.diffusion_models.length_regulator import DownSamplingRegulator
            model = DownSamplingRegulator(self.output_size, self.length_regulator_conf.get("sampling_ratios"))
        elif name == "interpolate":
            from funasr.models.llm_asr.diffusion_models.length_regulator import InterpolateRegulator
            model = InterpolateRegulator(self.output_size, **self.length_regulator_conf)
        else:
            raise ValueError(f"Unknown length_regulator {name}")

        self.length_regulator_conf["name"] = name

        return model

    def forward(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            audio: torch.Tensor,
            audio_lengths: torch.Tensor,
            xvec: torch.Tensor,
            xvec_lengths: torch.Tensor,
    ):
        batch_size = audio.shape[0]
        # for data parallel
        x = text[:, :text_lengths.max()]
        y = audio[:, :audio_lengths.max()]
        xvec = xvec[:, :xvec_lengths.max()]
        if self.vocab_size is not None and self.vocab_size > 0:
            mask = (x != -1).float().unsqueeze(-1)
            x = self.input_embedding(torch.clamp(x, min=0)) * mask

        # random select a xvec from xvec matrix
        xvec_list = []
        for i, ilen in enumerate(xvec_lengths):
            idx = random.randint(0, ilen-1)
            while torch.any(~torch.isfinite(xvec[i, idx])):
                idx = random.randint(0, ilen - 1)
            xvec_list.append(xvec[i, idx])
        rand_xvec = torch.vstack(xvec_list)
        rand_xvec = self.xvec_proj(rand_xvec)

        y, y_lengths = self.extract_feat(y, audio_lengths)
        h, h_lengths, _ = self.encoder(x, text_lengths)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, h_lengths, y, y_lengths)
        if self.prompt_conf is not None:
            target_prompt = self.select_target_prompt(y, y_lengths)
            conditions = dict(
                xvec=rand_xvec,
                target_prompt=target_prompt,
            )
        else:
            conditions = None

        mask = (~make_pad_mask(y_lengths)).to(y)
        # y, h in (B, T, D)
        loss, _ = self.decoder.compute_loss(
            y.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            rand_xvec,
            cond=conditions
        )

        stats = dict(loss=torch.clone(loss.detach()))

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @torch.no_grad()
    def extract_feat(self, y: torch.Tensor, y_lengths: torch.Tensor):
        if self.output_type == 'mel':
            return self.mel_extractor(y, y_lengths)
        elif self.output_type == "codec":
            return self.quantizer_codebook(y.long(), y_lengths)
        else:
            return y, y_lengths

    @torch.no_grad()
    def inference(self, text, text_lens, xvec, xvec_lens, diff_steps=10, temperature=1.0, prompt=None):
        avg_xvec = torch.mean(xvec, dim=1)
        avg_xvec = self.xvec_proj(avg_xvec)
        if self.vocab_size is not None and self.vocab_size > 0:
            mask = (text != -1).float().unsqueeze(-1)
            text = self.input_embedding(torch.clamp(text, min=0)) * mask
        h, h_lengths, _ = self.encoder(text, text_lens)
        h = self.encoder_proj(h)
        if self.output_type == "mel":
            coeff = ((self.mel_extractor.sampling_rate / self.mel_extractor.hop_size) /
                     self.input_frame_rate)
        else:
            coeff = ((self.quantizer_codebook.sampling_rate / self.quantizer_codebook.hop_size) /
                     self.input_frame_rate)
        y = torch.zeros([1, int(h.shape[1] * coeff), 80], device=text.device)
        y_lens = (text_lens * coeff).long()
        h, h_lengths = self.length_regulator(h, h_lengths, y, y_lens)
        mask = (~make_pad_mask(y_lens)).to(y)
        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            n_timesteps=diff_steps,
            temperature=temperature,
            spks=avg_xvec,
            cond=None,
        )
        return feat

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass


class MaskedDiffWithXvec(BaseDiffWithXvec):
    def __init__(self, input_size: int, output_size: int = 80, xvec_size: int = 198, output_type: str = "mel",
                 encoder_conf: Dict = {}, decoder_conf: Dict = {}, mel_feat_conf: Dict = {}, codec_conf: Dict = {},
                 length_regulator_conf: Dict = None, prompt_conf: Dict = None, vocab_size: int = None,
                 token_list: List = None, **kwargs):
        super().__init__(input_size, output_size, xvec_size, output_type, encoder_conf, decoder_conf, mel_feat_conf,
                         codec_conf, length_regulator_conf, prompt_conf, vocab_size, token_list, **kwargs)
        if self.prompt_conf is not None:
            self.masker = self.build_masker()
            self.cgf_prob = prompt_conf.get("cgf_prob", 0.0)
            self.prompt_dropout_rate = prompt_conf.get("prompt_dropout", 0.0)
            if self.prompt_dropout_rate > 0:
                self.prompt_dropout = nn.Dropout(self.prompt_dropout_rate)
            else:
                self.prompt_dropout = None
        self.only_mask_loss = kwargs.get("only_mask_loss", False)
        self.io_ratio = kwargs.get("io_ratio", None)
        if self.io_ratio == "auto":
            self.io_ratio = mel_feat_conf["sampling_rate"] / mel_feat_conf["hop_size"] / self.input_frame_rate
        self.first_package_conf = kwargs.get("first_package_conf", None)
        self.length_normalizer_ratio = kwargs.get("length_normalizer_ratio", None)

    def build_masker(self):
        prompt_type = self.prompt_conf.get("prompt_type", "free")
        if prompt_type == "prefix":
            from funasr.models.specaug.mask_along_axis import PrefixMaskVariableMaxWidth
            masker = PrefixMaskVariableMaxWidth(
                mask_width_ratio_range=self.prompt_conf["prompt_width_ratio_range"],
            )
        else:
            from funasr.models.specaug.mask_along_axis import MaskAlongAxisVariableMaxWidth
            masker = MaskAlongAxisVariableMaxWidth(
                mask_width_ratio_range=self.prompt_conf["prompt_width_ratio_range"],
                num_mask=1,
            )
        return masker

    @staticmethod
    def norm_and_sample_xvec(xvec, xvec_lengths):
        xvec_list = []
        for i, ilen in enumerate(xvec_lengths):
            if ilen == 1:
                idx = 0
            else:
                idx = random.randint(0, ilen - 1)
                while torch.any(~torch.isfinite(xvec[i, idx])):
                    idx = random.randint(0, ilen - 1)
            if torch.any(~torch.isfinite(xvec[i, idx])):
                to_add = torch.zeros_like(xvec[i, idx])
            else:
                to_add = xvec[i, idx]
            xvec_list.append(to_add)
        rand_xvec = torch.vstack(xvec_list)
        rand_xvec = F.normalize(rand_xvec, dim=1)

        return rand_xvec

    def select_target_prompt(self, y: torch.Tensor, y_lengths: torch.Tensor):
        _, _, cond_mask = self.masker(y, y_lengths, return_mask=True)
        cond_mask = ~cond_mask

        if self.cgf_prob > 0:
            cgf_mask = torch.rand([y.shape[0], 1, 1], dtype=torch.float32, device=y.device)
            cond_mask = cond_mask * (cgf_mask > self.cgf_prob)

        return cond_mask

    def build_decoder(self):
        decoder_name = self.decoder_conf.pop("name", "transformer")
        model = None

        if decoder_name == "matcha":
            from funasr.models.llm_asr.diffusion_models.flow_matching import CFM
            model = CFM(
                **self.decoder_conf,
            )

        self.decoder_conf["name"] = decoder_name

        return model

    def sample_first_package(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            audio: torch.Tensor,
            audio_lengths: torch.Tensor,
    ):
        sample_rate = self.first_package_conf["sample_rate"]
        min_token_len, max_token_len = self.first_package_conf["token_len_range"]
        random_start = self.first_package_conf.get("random_start", False)
        bs = text.shape[0]
        sample_mask = torch.rand((bs, ), device=text_lengths.device) < sample_rate
        if random_start:
            text_list, text_lengths_list, audio_list, audio_lengths_list = [], [], [], []
            for i, total_len in enumerate(text_lengths):
                total_len = total_len.item()
                if sample_mask[i].item():
                    if isinstance(min_token_len, float) and 0.0 < min_token_len <= 1.0:
                        min_token_len = math.floor(min_token_len * total_len)
                    if isinstance(max_token_len, float) and 0.0 < max_token_len <= 1.0:
                        max_token_len = math.floor(max_token_len * total_len)
                    if total_len > max_token_len > min_token_len:
                        fp_len = random.randint(min_token_len, max_token_len)
                        start = random.randint(0, total_len - fp_len)
                        audio_st, audio_len = self.calc_target_len(torch.tensor(start)), self.calc_target_len(torch.tensor(fp_len))
                    else:
                        start, fp_len = 0, total_len
                        audio_st, audio_len = 0, self.calc_target_len(fp_len)
                    text_list.append(text[i, start: start+fp_len])
                    text_lengths_list.append(fp_len)
                    audio_list.append(audio[i, audio_st: audio_st+audio_len])
                    audio_lengths_list.append(audio_list[-1].shape[0])
                else:
                    text_list.append(text[i])
                    text_lengths_list.append(text_lengths[i])
                    audio_list.append(audio[i, :min(self.calc_target_len(text_lengths[i]), audio_lengths[i])])
                    audio_lengths_list.append(audio_list[-1].shape[0])
            text = pad_list(text_list, pad_value=0.0).to(text)
            new_text_lengths = torch.tensor(text_lengths_list, dtype=torch.int64, device=text.device)
            audio = pad_list(audio_list, pad_value=0.0).to(audio)
            new_audio_lengths = torch.tensor(audio_lengths_list, dtype=torch.int64, device=audio.device)
        else:
            fp_token_len = torch.randint(min_token_len, max_token_len + 1, (bs,))
            fp_token_len = torch.minimum(fp_token_len.to(text_lengths), text_lengths)
            fp_audio_len = self.calc_target_len(fp_token_len)
            fp_audio_len = torch.minimum(fp_audio_len.to(audio_lengths), audio_lengths)
            new_text_lengths = torch.where(sample_mask, fp_token_len, text_lengths)
            new_audio_lengths = torch.where(sample_mask, fp_audio_len, audio_lengths)
            text = text * (~make_pad_mask(new_text_lengths, maxlen=text.shape[1]).unsqueeze(-1)).to(text.device)
            audio = audio * (~make_pad_mask(new_audio_lengths, maxlen=audio.shape[1]).unsqueeze(-1)).to(audio.device)

        return text, new_text_lengths, audio, new_audio_lengths

    @staticmethod
    def clip_both_side(y, y_lengths, raw_lengths):
        res_list = []
        new_length = []
        for i, (new_len, org_len) in enumerate(zip(y_lengths, raw_lengths)):
            if new_len >= org_len:
                res_list.append(y[i, :new_len])
            else:
                left = (org_len - new_len) // 2
                right = org_len - new_len - left
                res_list.append(y[i, left: org_len-right])

            new_length.append(res_list[-1].shape[0])

        new_length = torch.tensor(new_length).to(y_lengths)
        return pad_list(res_list, 0.0), new_length

    def forward(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            audio: torch.Tensor,
            audio_lengths: torch.Tensor,
            xvec: Optional[torch.Tensor] = None,
            xvec_lengths: Optional[torch.Tensor] = None,
            **kwargs
    ):
        batch_size = audio.shape[0]
        # for data parallel
        with autocast(False):
            x = text[:, :text_lengths.max()]
            y = audio[:, :audio_lengths.max()]
            if self.vocab_size is not None and self.vocab_size > 0:
                mask = (x != -1).float().unsqueeze(-1)
                x = self.input_embedding(torch.clamp(x, min=0)) * mask

            # random select a xvec from xvec matrix
            rand_xvec = None
            if xvec is not None:
                xvec = xvec[:, :xvec_lengths.max()]
                rand_xvec = self.norm_and_sample_xvec(xvec, xvec_lengths)
                rand_xvec = self.xvec_proj(rand_xvec)

            y, y_lengths = self.extract_feat(y, audio_lengths)
            if self.length_normalizer_ratio is not None:
                max_y_lengths = torch.round(text_lengths * self.length_normalizer_ratio).long()
                raw_lengths = y_lengths.clone()
                y_lengths = torch.where(y_lengths > max_y_lengths, max_y_lengths, y_lengths)
                y, new_y_lengths = self.clip_both_side(y, y_lengths, raw_lengths)
                logging.info(f"normalized y_length from {raw_lengths.cpu().tolist()} to {y_lengths.cpu().tolist()} "
                             f"new_y_length {new_y_lengths.cpu().tolist()}, with text_lengths {text_lengths.cpu().tolist()}")
                y = y[:, :new_y_lengths.max()]
            elif self.io_ratio is not None:
                hint_once(f"cut output with ratio {self.io_ratio}", "print_ratio", rank=0)
                max_y_lengths = (text_lengths * self.io_ratio + 3).long()
                if y_lengths.max() > max_y_lengths.max():
                    logging.info(f"cut output with ratio {self.io_ratio} from {y_lengths.max()} to {max_y_lengths.max()}")
                    y_lengths = torch.where(y_lengths > max_y_lengths, max_y_lengths, y_lengths)
                    y = y[:, :y_lengths.max()]

            if self.first_package_conf is not None:
                x, text_lengths, y, y_lengths = self.sample_first_package(
                    x, text_lengths, y, y_lengths
                )
                x = x[:, :text_lengths.max()]
                y = y[:, :y_lengths.max()]
        h, _, _ = self.encoder(x, text_lengths)
        h_lengths = text_lengths
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, h_lengths, y, y_lengths)
        if self.prompt_conf is not None:
            cond_mask = self.select_target_prompt(y, y_lengths)
            if self.prompt_dropout is not None:
                hint_once(f"prompt dropout {self.prompt_dropout_rate}", "prompt dropout")
                y = self.prompt_dropout(y)
            conditions = (y * cond_mask).transpose(1, 2)
        else:
            cond_mask, conditions = None, None

        stats = dict(
            batch_size=batch_size,
            in_lengths=text_lengths.max(),
            out_lengths=y_lengths.max(),
        )

        mask = (~make_pad_mask(y_lengths)).to(y)
        # y, h in (B, T, D)
        loss, _ = self.decoder.compute_loss(
            y.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            rand_xvec,
            cond=conditions,
            reduction="none",
        )
        loss = loss.transpose(1, 2)
        all_loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * loss.shape[-1])
        if cond_mask is not None:
            masked_loss_mask = mask.unsqueeze(-1) * (~cond_mask)
        else:
            masked_loss_mask = mask.unsqueeze(-1)
        masked_loss = (loss * masked_loss_mask).sum() / (masked_loss_mask.sum() * loss.shape[-1])
        stats["all_loss"] = all_loss.item()
        stats["masked_loss"] = masked_loss.item()

        loss = masked_loss if self.only_mask_loss else all_loss

        stats["loss"] = loss.item()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @staticmethod
    def concat_prompt(prompt, prompt_lengths, text, text_lengths):
        xs_list, x_len_list = [], []
        for idx, (_prompt_len, _text_len) in enumerate(zip(prompt_lengths, text_lengths)):
            xs_list.append(torch.concat([prompt[idx, :_prompt_len], text[idx, :_text_len]], dim=0))
            x_len_list.append(_prompt_len + _text_len)

        xs = pad_list(xs_list, pad_value=0.0)
        x_lens = torch.tensor(x_len_list, dtype=torch.int64).to(xs.device)

        return xs, x_lens

    @staticmethod
    def remove_prompt(prompt, prompt_lengths, padded, padded_lengths):
        xs_list = []
        for idx, (_prompt_len, _x_len) in enumerate(zip(prompt_lengths, padded_lengths)):
            xs_list.append(padded[idx, _prompt_len: _x_len])

        xs = pad_list(xs_list, pad_value=0.0)

        return xs, padded_lengths - prompt_lengths

    @staticmethod
    def norm_and_avg_xvec(xvec: torch.Tensor, xvec_lens: torch.Tensor):
        mask = torch.isfinite(xvec.norm(dim=-1, keepdim=True))
        norm_xvec = F.normalize(xvec, dim=-1) * mask
        avg_xvec = F.normalize(torch.sum(norm_xvec, dim=1) / mask.sum(), dim=-1)
        return avg_xvec

    def calc_target_len(self, in_len):
        if self.input_frame_rate == 25 and self.output_type == "mel":
            if self.length_normalizer_ratio is not None:
                if isinstance(in_len, int):
                    in_len = torch.tensor(in_len)
                ll = torch.round(in_len * self.length_normalizer_ratio)
            else:
                ll = (in_len * 4 + 4) * 160 + 400
                ll = ll / 16000 * self.mel_extractor.sampling_rate / self.mel_extractor.hop_size
            if isinstance(in_len, int):
                ll = int(round(ll))
            else:
                ll = torch.round(ll).long()
            return ll
        if self.input_frame_rate == 50 and self.output_type == "mel":
            if self.length_normalizer_ratio is not None:
                if isinstance(in_len, int):
                    in_len = torch.tensor(in_len)
                ll = torch.round(in_len * self.length_normalizer_ratio)
            else:
                ll = (in_len * 2 + 2) * 160 + 400
                ll = ll / 16000 * self.mel_extractor.sampling_rate / self.mel_extractor.hop_size
            if isinstance(in_len, int):
                ll = int(round(ll))
            else:
                ll = torch.round(ll).long()
            return ll
        elif self.output_type == "codec":
            return in_len
        else:
            raise ValueError(f"Frame rate {self.input_frame_rate} has not implemented.")

    @torch.no_grad()
    def inference(self, text, text_lens,
                  xvec=None, xvec_lens=None,
                  diff_steps=10, temperature=1.0, prompt: dict = None, y_lens=None):
        rand_xvec = None
        if xvec is not None:
            if xvec.dim() == 2:
                xvec = xvec.unsqueeze(1)
                xvec_lens = torch.ones_like(xvec_lens)
            rand_xvec = self.norm_and_avg_xvec(xvec, xvec_lens)
            rand_xvec = self.xvec_proj(rand_xvec)

        prompt_text, prompt_text_lens = prompt.get("prompt_text", (None, None))
        prompt_audio, prompt_audio_lens = prompt.get("prompt_audio", (None, None))

        if self.vocab_size is not None and self.vocab_size > 0:
            if prompt_text is not None:
                text, text_lens = self.concat_prompt(prompt_text, prompt_text_lens, text, text_lens)
            mask = (text != -1).float().unsqueeze(-1)
            text = self.input_embedding(torch.clamp(text, min=0)) * mask

        h, h_lengths, _ = self.encoder(text, text_lens)
        h = self.encoder_proj(h)
        if y_lens is None:
            y_lens = self.calc_target_len(text_lens)
        y = torch.zeros([1, y_lens.max().item(), self.output_size], device=text.device)
        h, h_lengths = self.length_regulator(h, h_lengths, y, y_lens)

        # get conditions
        if prompt_audio is not None:
            if prompt_audio.ndim == 2:
                prompt_audio, prompt_audio_lens = self.extract_feat(prompt_audio, prompt_audio_lens)
            for i, _len in enumerate(prompt_audio_lens):
                y[i, :_len] = prompt_audio[i]
        conds = y.transpose(1, 2)

        mask = (~make_pad_mask(y_lens)).to(y)
        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            n_timesteps=diff_steps,
            temperature=temperature,
            spks=rand_xvec,
            cond=conds,
        )

        if prompt_text is not None and prompt_audio is not None:
            feat = feat.transpose(1, 2)
            feat_lens = torch.tensor([feat.shape[1]], dtype=torch.int64, device=feat.device)
            feat, feat_lens = self.remove_prompt(None, prompt_audio_lens, feat, feat_lens)
            feat = feat.transpose(1, 2)

        # if prompt_audio is not None:
        #     feat_rmq = torch.sqrt(torch.mean(torch.pow(feat, 2), dim=[1, 2], keepdim=True))
        #     prompt_rmq = torch.sqrt(torch.mean(torch.pow(prompt_audio, 2), dim=[1, 2], keepdim=True))
        #     feat = feat / feat_rmq * prompt_rmq

        return feat


class MaskedDiffTTS(MaskedDiffWithXvec):

    def __init__(self, input_size: int, output_size: int = 80, xvec_size: int = 198, output_type: str = "mel",
                 encoder_conf: Dict = {}, decoder_conf: Dict = {}, mel_feat_conf: Dict = {}, codec_conf: Dict = {},
                 length_regulator_conf: Dict = None, prompt_conf: Dict = None, vocab_size: int = None,
                 token_list: List = None, **kwargs):
        super().__init__(input_size, output_size, xvec_size, output_type, encoder_conf, decoder_conf, mel_feat_conf,
                         codec_conf, length_regulator_conf, prompt_conf, vocab_size, token_list, **kwargs)
        self.length_loss_weight = kwargs.get("length_loss_weight", 0.0)
        if self.length_loss_weight > 0.0:
            self.length_predictor = nn.Linear(self.encoder.output_size(), 1)

    def calc_target_len(self, enc_outs, enc_lens):
        text_durs = self.length_predictor(enc_outs)
        text_durs = torch.exp(text_durs)
        mask = ~make_pad_mask(enc_lens, xs=text_durs)
        utt_durs = (text_durs * mask).sum(dim=1).squeeze(-1)
        return utt_durs

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor, audio: torch.Tensor, audio_lengths: torch.Tensor,
                xvec: Optional[torch.Tensor] = None, xvec_lengths: Optional[torch.Tensor] = None):
        batch_size = audio.shape[0]
        # for data parallel
        x = text[:, :text_lengths.max()]
        y = audio[:, :audio_lengths.max()]
        if self.vocab_size is not None and self.vocab_size > 0:
            mask = (x != -1).float().unsqueeze(-1)
            x = self.input_embedding(torch.clamp(x, min=0)) * mask

        # random select a xvec from xvec matrix
        rand_xvec = None
        if xvec is not None:
            xvec = xvec[:, :xvec_lengths.max()]
            rand_xvec = self.norm_and_sample_xvec(xvec, xvec_lengths)
            rand_xvec = self.xvec_proj(rand_xvec)

        y, y_lengths = self.extract_feat(y, audio_lengths)
        h, _, _ = self.encoder(x, text_lengths)
        h_lengths = text_lengths
        h_durs = self.calc_target_len(h, h_lengths)
        utt_dur_loss = self.length_loss_weight * F.l1_loss(h_durs, y_lengths)

        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, h_lengths, y, y_lengths)
        if self.prompt_conf is not None:
            cond_mask = self.select_target_prompt(y, y_lengths)
            conditions = (y * cond_mask).transpose(1, 2)
        else:
            cond_mask, conditions = None, None

        stats = dict(
            batch_size=batch_size,
            in_lengths=text_lengths.max(),
            out_lengths=y_lengths.max(),
        )

        mask = (~make_pad_mask(y_lengths)).to(y)
        # y, h in (B, T, D)
        loss, _ = self.decoder.compute_loss(
            y.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            rand_xvec,
            cond=conditions,
            reduction="none",
        )
        loss = loss.transpose(1, 2)
        all_loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * loss.shape[-1])
        if cond_mask is not None:
            masked_loss_mask = mask.unsqueeze(-1) * (~cond_mask)
        else:
            masked_loss_mask = mask.unsqueeze(-1)
        masked_loss = (loss * masked_loss_mask).sum() / (masked_loss_mask.sum() * loss.shape[-1])
        stats["all_loss"] = all_loss.item()
        stats["masked_loss"] = masked_loss.item()

        loss = masked_loss if self.only_mask_loss else all_loss
        stats["mel_loss"] = loss.item()

        loss = loss + utt_dur_loss
        stats["loss"] = loss.item()
        stats["utt_dur_loss"] = utt_dur_loss.item()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(self, text, text_lens, xvec=None, xvec_lens=None, diff_steps=10, temperature=1.0,
                  prompt: dict = None):
        rand_xvec = None
        if xvec is not None:
            if xvec.dim() == 2:
                xvec = xvec.unsqueeze(1)
                xvec_lens = torch.ones_like(xvec_lens)
            rand_xvec = self.norm_and_avg_xvec(xvec, xvec_lens)
            rand_xvec = self.xvec_proj(rand_xvec)

        prompt_text, prompt_text_lens = prompt.get("prompt_text", (None, None))
        prompt_audio, prompt_audio_lens = prompt.get("prompt_audio", (None, None))

        if self.vocab_size is not None and self.vocab_size > 0:
            if prompt_text is not None:
                text, text_lens = self.concat_prompt(prompt_text, prompt_text_lens, text, text_lens)
            mask = (text != -1).float().unsqueeze(-1)
            text = self.input_embedding(torch.clamp(text, min=0)) * mask

        h, _, _ = self.encoder(text, text_lens)
        h_lengths = text_lens
        y_lens = self.calc_target_len(h, h_lengths).round().long()
        y = torch.zeros([1, y_lens.max().item(), self.output_size], device=text.device)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, h_lengths, y, y_lens)

        # get conditions
        if prompt_audio is not None:
            if prompt_audio.ndim == 2:
                prompt_audio, prompt_audio_lens = self.extract_feat(prompt_audio, prompt_audio_lens)
            for i, _len in enumerate(prompt_audio_lens):
                y[i, :_len] = prompt_audio[i]
        conds = y.transpose(1, 2)

        mask = (~make_pad_mask(y_lens)).to(y)
        feat = self.decoder.forward(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            n_timesteps=diff_steps,
            temperature=temperature,
            spks=rand_xvec,
            cond=conds,
        )

        if prompt_text is not None and prompt_audio is not None:
            feat = feat.transpose(1, 2)
            feat_lens = torch.tensor([feat.shape[1]], dtype=torch.int64, device=feat.device)
            feat, feat_lens = self.remove_prompt(None, prompt_audio_lens, feat, feat_lens)
            feat = feat.transpose(1, 2)

        return feat

