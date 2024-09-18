import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.hinter import hint_once
from funasr.models.transformer.utils.nets_utils import pad_list
import numpy as np
import random
from funasr.train_utils.set_all_random_seed import set_all_random_seed


def norm_and_sample_xvec(xvec, xvec_lengths):
    xvec_list = []
    for i, ilen in enumerate(xvec_lengths):
        idx = random.randint(0, ilen - 1)
        while torch.any(~torch.isfinite(xvec[i, idx])):
            idx = random.randint(0, ilen - 1)
        xvec_list.append(xvec[i, idx])
    rand_xvec = torch.vstack(xvec_list)
    rand_xvec = F.normalize(rand_xvec, dim=1)

    return rand_xvec


class UpsampleCtcTokenDiffModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            vocab_size: int,
            token_list: list,
            token_vocab_size: int,
            endofprompt_token_id: int = None,
            text_encoder_conf: dict = None,
            aggregator_conf: dict = None,
            am_config: dict = None,
            fm_config: dict = None,
            xvec_size: int = None,
            **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.text_vocab_size = vocab_size
        self.token_list = token_list
        self.token_vocab_size = token_vocab_size
        self.endofprompt_token_id = endofprompt_token_id
        self.text_encoder_conf = text_encoder_conf
        self.aggregator_conf = aggregator_conf
        self.am_config = am_config
        self.fm_config = fm_config
        self.xvec_size = xvec_size

        # build nn
        self.text_embedding = nn.Embedding(vocab_size, output_size)
        self.xvec_proj = None
        if xvec_size is not None:
            self.xvec_proj = nn.Linear(xvec_size, output_size)
        self.text_encoder = self.build_text_encoder()
        self.am_model = self.build_am_model()
        self.fm_model = self.build_fm_model()
        self.am_aggregator = self.build_aggregator()
        self.fm_aggregator = self.build_aggregator()

        # set optional parameters
        self.xvec_drop_rate = kwargs.get('xvec_drop_rate', None)
        self.use_prompt_as_xvec = kwargs.get('use_prompt_as_xvec', False)
        if self.use_prompt_as_xvec:
            self.spk_aggregator = self.build_aggregator()
            spk_query = torch.randn(1, 1, self.output_size)
            torch.nn.init.xavier_normal_(spk_query)
            self.spk_query = torch.nn.Parameter(spk_query, requires_grad=True)

    def build_aggregator(self):
        name = self.aggregator_conf.pop("name", None)
        model = None
        if name == "transformer":
            from funasr.models.llm_asr.tts_models.transformer_decoder import TransformerDecoder
            model = TransformerDecoder(self.output_size, **self.aggregator_conf)
        self.aggregator_conf["name"] = name

        return model

    def build_text_encoder(self):
        name = self.text_encoder_conf.pop("name", None)
        model = None
        if name == "transformer":
            from funasr.models.llm_asr.conformer_encoder import ConformerEncoder
            model = ConformerEncoder(
                **self.text_encoder_conf,
                input_size=self.output_size,
                use_cnn_module=False,
                macaron_style=False,
            )
        elif name == "conformer":
            from funasr.models.llm_asr.conformer_encoder import ConformerEncoder
            model = ConformerEncoder(
                **self.text_encoder_conf,
                input_size=self.output_size,
            )

        self.text_encoder_conf["name"] = name
        return model

    def build_am_model(self):
        name = self.am_config.pop("name", None)
        model = None
        if name == "nar_ctc_model":
            from funasr.models.llm_asr.tts_models.nar_acoustic_model import NARCTCModel
            model = NARCTCModel(**self.am_config)
        elif name == "nar_ctc_prob_model":
            from funasr.models.llm_asr.tts_models.nar_acoustic_model import NARCTCProbModel
            model = NARCTCProbModel(**self.am_config)

        self.am_config["name"] = name
        return model

    def build_fm_model(self):
        name = self.fm_config.pop("name", None)
        model = None
        if name == "masked_diff_with_xvec":
            from funasr.models.llm_asr.flow_matching import MaskedDiffWithXvec
            model = MaskedDiffWithXvec(**self.fm_config)

        self.fm_config["name"] = name
        return model

    def split_prompt(
            self,
            text_emb: torch.Tensor,
            text_emb_lens: torch.Tensor,
            text: torch.Tensor,
            text_lens: torch.Tensor,
    ):
        prompts, prompt_lens = [], []
        outs, outs_lens = [], []
        batch_size = text.shape[0]
        for i in range(batch_size):
            delta = text_emb_lens[i] - text_lens[i]
            # 1 for exclude <|endofprompt|> token
            pos = torch.where(text[i] == self.endofprompt_token_id)[0][0].item() + delta + 1
            _x = text_emb[i, pos:text_emb_lens[i]]
            outs.append(_x)
            outs_lens.append(_x.shape[0])

            _prompt = text_emb[i, :pos]
            prompts.append(_prompt)
            prompt_lens.append(_prompt.shape[0])

        outs = pad_list(outs, pad_value=0.0)
        outs_lens = torch.tensor(outs_lens, dtype=torch.int64, device=text_emb.device)

        prompts = pad_list(prompts, pad_value=0.0)
        prompt_lens = torch.tensor(prompt_lens, dtype=torch.int64, device=text_emb.device)

        return prompts, prompt_lens, outs, outs_lens

    def forward(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            speech_token: torch.Tensor,
            speech_token_lengths: torch.Tensor,
            audio: torch.Tensor,
            audio_lengths: torch.Tensor,
            xvec: Optional[torch.Tensor] = None,
            xvec_lengths: Optional[torch.Tensor] = None,
            **kwargs
    ):
        text = text[:, :text_lengths.max()]
        speech_token = speech_token[:, :speech_token_lengths.max()]
        audio = audio[:, :audio_lengths.max()]
        batch_size = text.shape[0]

        # embed text inputs
        mask = (text != -1).float().unsqueeze(-1)
        text_emb = self.text_embedding(torch.clamp(text, min=0)) * mask
        text_emb_lengths = text_lengths

        prompt, prompt_lens, text_emb, text_emb_lengths = self.split_prompt(
            text_emb, text_emb_lengths, text, text_lengths
        )

        if self.use_prompt_as_xvec:
            prompt_xvec, _ = self.spk_aggregator(
                prompt, prompt_lens,
                self.spk_query.expand([batch_size, -1, -1]), torch.tensor([1]*batch_size).to(prompt_lens)
            )
            endofprompt_emb = self.text_embedding(torch.tensor([self.endofprompt_token_id]*batch_size).to(text).unsqueeze(1))
            prompt = torch.cat([prompt_xvec, endofprompt_emb], dim=1)
            prompt_lens = torch.tensor([2]*batch_size).to(prompt_lens)
            hint_once("using prompt as speaker embedding.", "use_prompt_spk_emb")

        # random select a xvec from xvec matrix
        if not self.use_prompt_as_xvec and self.xvec_proj is not None and xvec is not None:
            xvec = xvec[:, :xvec_lengths.max()]
            rand_xvec = norm_and_sample_xvec(xvec, xvec_lengths)
            rand_xvec = self.xvec_proj(rand_xvec)
            if self.xvec_drop_rate is not None:
                xvec_mask = (torch.rand((rand_xvec.shape[0], 1)) >= self.xvec_drop_rate).to(rand_xvec)
                rand_xvec = rand_xvec * xvec_mask
                hint_once(f"randomly drop out xvec with mask {xvec_mask.squeeze()}", "xvec_drop_out")
            rand_xvec = rand_xvec.unsqueeze(1)
            prompt = torch.cat([rand_xvec, prompt], dim=1)
            prompt_lens = prompt_lens + 1
            hint_once("using speaker embedding as slot.", "use_spk_emb")

        # remove prompt text
        # prompt, prompt_lens, text_emb, text_emb_lengths = self.split_prompt(
        #     text_emb, text_emb_lengths, text, text_lengths
        # )
        outs_tuple = self.text_encoder(text_emb, ilens=text_emb_lengths)
        text_enc = outs_tuple[0]
        text_enc_lens = text_emb_lengths
        text_enc, _ = self.am_aggregator(
            prompt, prompt_lens,
            text_enc, text_enc_lens
        )

        states = dict(
            batch_size=float(batch_size),
            text_len=float(text_emb.shape[1]),
            speech_len=float(speech_token.shape[1]),
            token_text_ratio=float(speech_token.shape[1]) / float(text_emb.shape[1]),
        )
        # forward AM model
        am_retvals = self.am_model.force_align_text(
            speech_token, speech_token_lengths,
            text_enc, text_enc_lens,
            **kwargs
        )
        am_loss, aligned_token_emb, am_states = am_retvals
        # update AM states
        for key, val in am_states.items():
            states[f"am_{key}"] = val

        aligned_token_emb, _ = self.fm_aggregator(
            prompt, prompt_lens,
            aligned_token_emb, speech_token_lengths
        )

        # forward FM model
        fm_loss, fm_states, _ = self.fm_model.forward(
            aligned_token_emb, speech_token_lengths,
            audio, audio_lengths,
            **kwargs,
        )
        # update FM states
        for key, val in fm_states.items():
            states[f"fm_{key}"] = val

        loss = am_loss + fm_loss
        states["loss"] = loss.item()

        loss, states, weight = force_gatherable((loss, states, batch_size), loss.device)

        return loss, states, weight

    def inference(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            xvec: Optional[torch.Tensor] = None,
            xvec_lengths: Optional[torch.Tensor] = None,
            **kwargs
    ):
        blank_penalty = kwargs.get("blank_penalty", 0.0)
        sampling = kwargs.get("sampling", "greedy")
        prompt_dict = kwargs.get("prompt_dict", {})
        prompt_token = prompt_dict.get("prompt_token", (None, None))
        prompt_audio = prompt_dict.get("prompt_audio", (None, None))
        # fully un-causal mode
        use_causal_prob = kwargs.get("use_causal_prob", 1.0)
        # embed text inputs
        mask = (text != -1).float().unsqueeze(-1)
        text_emb = self.text_embedding(torch.clamp(text, min=0)) * mask
        text_emb_lengths = text_lengths
        batch_size = text.shape[0]

        prompt, prompt_lens, text_emb, text_emb_lengths = self.split_prompt(
            text_emb, text_emb_lengths, text, text_lengths
        )

        if self.use_prompt_as_xvec:
            prompt_xvec, _ = self.spk_aggregator(
                prompt, prompt_lens,
                self.spk_query.expand([batch_size, -1, -1]), torch.tensor([1] * batch_size).to(prompt_lens)
            )
            endofprompt_emb = self.text_embedding(
                torch.tensor([self.endofprompt_token_id] * batch_size).to(text).unsqueeze(1))
            prompt = torch.cat([prompt_xvec, endofprompt_emb], dim=1)
            prompt_lens = torch.tensor([2] * batch_size).to(prompt_lens)
            hint_once("using prompt as speaker embedding.", "use_prompt_spk_emb")

        # using the xvec
        if self.xvec_proj is not None and not self.use_prompt_as_xvec:
            if xvec is not None:
                hint_once("using speaker embedding as slot.", "use_spk_emb")
                xvec = xvec[:, :xvec_lengths.max()]
                rand_xvec = norm_and_sample_xvec(xvec, xvec_lengths)
                rand_xvec = self.xvec_proj(rand_xvec)
                rand_xvec = rand_xvec.unsqueeze(1)
            else:
                hint_once("using zeros as speaker embedding.", "use_spk_emb")
                rand_xvec = torch.zeros([text_emb.shape[0], 1, self.output_size]).to(text_emb)
            prompt = torch.cat([rand_xvec, prompt], dim=1)
            prompt_lens = prompt_lens + 1

        outs_tuple = self.text_encoder(text_emb, ilens=text_emb_lengths)
        text_enc = outs_tuple[0]
        text_enc_lens = text_emb_lengths
        text_enc, _ = self.am_aggregator(
            prompt, prompt_lens,
            text_enc, text_enc_lens
        )

        # forward AM model
        tokens, aligned_token_emb, aligned_token_lens = self.am_model.inference(
            text_enc, text_enc_lens,
            sampling=sampling,
            blank_penalty=blank_penalty,
            text_is_embedding=True,
            return_hidden=True,
            use_causal_prob=use_causal_prob,
        )
        if isinstance(tokens, tuple):
            tokens, fa_tokens = tokens

        aligned_token_emb, _ = self.fm_aggregator(
            prompt, prompt_lens,
            aligned_token_emb, aligned_token_lens
        )

        # forward FM model
        set_all_random_seed(0)
        feat = self.fm_model.inference(
            aligned_token_emb, aligned_token_lens,
            prompt=dict(
                prompt_text=prompt_token,
                prompt_audio=prompt_audio,
            ),
            **kwargs,
        )
        feat = self.rms_rescale_feat(feat)

        return tokens, feat

    def rms_rescale_feat(self, feat, target_feat_rms=3.5, feat_sil_th=0.5):
        feat_power = feat.exp().sum(1)
        # not silence
        if feat_power.max() > feat_sil_th:
            mask = feat_power > feat_sil_th
            feat_rms = torch.sqrt(torch.mean(torch.square(feat_power)))
            feat = feat + mask.unsqueeze(1) * np.log(target_feat_rms / feat_rms.cpu().numpy().item())

        return feat

    def get_hop_lens(self, fa_tokens, lookahead_size):
        if lookahead_size == 0:
            return 0, 0

        fa_tokens = fa_tokens[0].cpu().tolist()
        upsample_rate = np.cumprod(self.am_model.encoder.upsample_ratios)[-1]
        lookahead_tokens = [[x-1] for x in fa_tokens[-lookahead_size*upsample_rate:] if x > 0]
        lookahead_token_len = len(lookahead_tokens)
        return lookahead_token_len

    def streaming_inference(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            xvec: Optional[torch.Tensor] = None,
            xvec_lengths: Optional[torch.Tensor] = None,
            **kwargs
    ):
        device = text.device
        use_causal_prob = kwargs.get("use_causal_prob", 1.0)
        # streaming related config
        chunk_size = kwargs.get("streaming_chunk_size", 1)
        chunk_size_maxium = kwargs.get("chunk_size_maxium", 16)
        try:
            lookahead_size = self.am_model.encoder.pre_lookahead_len
        except AttributeError:
            lookahead_size = 0
        hint_once(f"chunk_size={chunk_size}, chunk_size_maxium={chunk_size_maxium}, "
                  f"pre lookahead size={lookahead_size}.",
                  "pre_lookahead_len")
        given_rtf = kwargs.get("given_rtf", 0.5)

        blank_penalty = kwargs.get("blank_penalty", 0.0)
        sampling = kwargs.get("sampling", "greedy")
        prompt_dict = kwargs.get("prompt_dict", {})
        prompt_token = list(prompt_dict.get("prompt_token", [None, None]))
        prompt_audio = list(prompt_dict.get("prompt_audio", [None, None]))
        streaming_mode = kwargs.get("streaming_mode", "v2")

        if prompt_token[0] is None:
            prompt_token[0] = torch.zeros([1, 0, self.output_size], device=device, dtype=torch.float32)
            prompt_token[1] = torch.tensor([0], device=device, dtype=torch.long)
        if prompt_audio[0] is None:
            prompt_audio[0] = torch.zeros(
                [1, 0, self.fm_model.mel_extractor.num_mels],
                device=device, dtype=torch.float32
            )
            prompt_audio[1] = torch.tensor([0], device=device, dtype=torch.long)

        # embed text inputs
        mask = (text != -1).float().unsqueeze(-1)
        text_emb = self.text_embedding(torch.clamp(text, min=0)) * mask
        text_emb_lengths = text_lengths

        batch_size = text.shape[0]

        prompt, prompt_lens, text_emb, text_emb_lengths = self.split_prompt(
            text_emb, text_emb_lengths, text, text_lengths
        )

        if self.use_prompt_as_xvec:
            prompt_xvec, _ = self.spk_aggregator(
                prompt, prompt_lens,
                self.spk_query.expand([batch_size, -1, -1]), torch.tensor([1] * batch_size).to(prompt_lens)
            )
            endofprompt_emb = self.text_embedding(
                torch.tensor([self.endofprompt_token_id] * batch_size).to(text).unsqueeze(1))
            prompt = torch.cat([prompt_xvec, endofprompt_emb], dim=1)
            prompt_lens = torch.tensor([2] * batch_size).to(prompt_lens)
            hint_once("using prompt as speaker embedding.", "use_prompt_spk_emb")

        # using the xvec
        if self.xvec_proj is not None and not self.use_prompt_as_xvec:
            if xvec is not None:
                hint_once("using speaker embedding as slot.", "use_spk_emb")
                xvec = xvec[:, :xvec_lengths.max()]
                rand_xvec = norm_and_sample_xvec(xvec, xvec_lengths)
                rand_xvec = self.xvec_proj(rand_xvec)
                rand_xvec = rand_xvec.unsqueeze(1)
            else:
                hint_once("using zeros as speaker embedding.", "use_spk_emb")
                rand_xvec = torch.zeros([text_emb.shape[0], 1, self.output_size]).to(text_emb)
            prompt = torch.cat([rand_xvec, prompt], dim=1)
            prompt_lens = prompt_lens + 1

        chunk_id = 0
        chunk_start = 0
        while True:
            _st_time = time.time()
            _size = max(int(round(chunk_size / (given_rtf ** chunk_id))), chunk_size_maxium)
            chunk_end = chunk_start + _size
            chunk_text_emb = text_emb[:, :chunk_end+lookahead_size]
            chunk_text_emb_lengths = torch.tensor([chunk_text_emb.shape[1]], dtype=torch.long, device=device)

            outs_tuple = self.text_encoder(chunk_text_emb, ilens=chunk_text_emb_lengths)
            text_enc = outs_tuple[0]
            text_enc_lens = chunk_text_emb_lengths
            text_enc, _ = self.am_aggregator(
                prompt, prompt_lens,
                text_enc, text_enc_lens
            )

            # forward AM model
            tokens, aligned_token_emb, aligned_token_lens = self.am_model.inference(
                text_enc, text_enc_lens,
                sampling=sampling,
                blank_penalty=blank_penalty,
                text_is_embedding=True,
                return_hidden=True,
                use_causal_prob=use_causal_prob,
            )
            token_hop_len, mel_hop_len = 0, 0
            if isinstance(tokens, tuple):
                tokens, fa_tokens = tokens
                token_hop_len = self.get_hop_lens(fa_tokens, lookahead_size)
                mel_hop_len = int(round(token_hop_len * self.fm_model.length_normalizer_ratio))

            # exclude empty tokens.
            if aligned_token_emb.shape[1] > prompt_token[0].shape[1]:
                aligned_token_emb, _ = self.fm_aggregator(
                    prompt, prompt_lens,
                    aligned_token_emb, aligned_token_lens
                )
                cur_token = aligned_token_emb[:, prompt_token[0].shape[1]:]
                cur_token_len = aligned_token_lens - prompt_token[1]

                # v2: excluding lookahead tokens for not-last packages
                if streaming_mode == "v2":
                    if chunk_end + lookahead_size < text_emb.shape[1]:
                        cur_token = cur_token[:, :cur_token.shape[1]-token_hop_len, :]
                        cur_token_len = cur_token_len - token_hop_len

                # forward FM model
                feat = self.fm_model.inference(
                    cur_token, cur_token_len,
                    prompt=dict(
                        prompt_text=prompt_token,
                        prompt_audio=prompt_audio,
                    ),
                    **kwargs,
                )
                feat = self.rms_rescale_feat(feat)
                cost = time.time() - _st_time
                if chunk_id == 0:
                    logging.info(f"First package delay: {cost*1000.0:.2f}ms")
                print_token = tokens.cpu().squeeze().tolist()
                logging.info(f"pack {chunk_id}: valid_tokens: {print_token[:len(print_token)-token_hop_len]}, "
                             f"pad_tokens: {print_token[len(print_token)-token_hop_len:]}.")

                if streaming_mode == "v1":
                    # v1: excluding lookahead parts for not-last packages
                    if chunk_end + lookahead_size < text_emb.shape[1]:
                        cur_token = cur_token[:, :cur_token.shape[1]-token_hop_len, :]
                        feat = feat[:, :, :feat.shape[2] - mel_hop_len]

                if streaming_mode == "v2":
                    # v2: reback token and mel feat
                    if chunk_end + lookahead_size < text_emb.shape[1]:
                        text_reback = 2 if chunk_id == 0 else 4
                        token_hop_len_2 = self.get_hop_lens(fa_tokens, lookahead_size + text_reback)
                        token_reback = token_hop_len_2 - token_hop_len
                        cur_token = cur_token[:, :cur_token.shape[1] - token_reback, :]
                        feat_reback = int(round(token_reback * self.fm_model.length_normalizer_ratio))
                        feat = feat[:, :, :feat.shape[2] - feat_reback]
                        chunk_end = chunk_end - text_reback

                # update values and lens of prompt token and audio
                prompt_token[1] = prompt_token[1] + cur_token.shape[1]
                prompt_token[0] = torch.concat([prompt_token[0], cur_token], dim=1)
                prompt_audio[1] = prompt_audio[1] + feat.shape[2]
                prompt_audio[0] = torch.concat([prompt_audio[0], feat.transpose(1, 2)], dim=1)

            chunk_id += 1
            chunk_start = chunk_end
            if chunk_end + lookahead_size >= text_emb.shape[1]:
                break

        return tokens, prompt_audio[0].transpose(1, 2)

    def collect_feats(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        pass


class UCTDXvecSlotModel(UpsampleCtcTokenDiffModel):
    def __init__(self, input_size: int, output_size: int, vocab_size: int, token_list: list, token_vocab_size: int,
                 endofprompt_token_id: int = None, text_encoder_conf: dict = None, aggregator_conf: dict = None,
                 am_config: dict = None, fm_config: dict = None, xvec_size: int = None, **kwargs):
        super().__init__(input_size, output_size, vocab_size, token_list, token_vocab_size, endofprompt_token_id,
                         text_encoder_conf, aggregator_conf, am_config, fm_config, xvec_size, **kwargs)
        # remove am and fm aggregator
        self.am_aggregator = None
        self.fm_aggregator = None

        # build speaker aggregator for Prompt Text
        self.spk_aggregator = self.build_aggregator()
        spk_query = torch.randn(1, 1, self.output_size)
        torch.nn.init.xavier_normal_(spk_query)
        self.spk_query = torch.nn.Parameter(spk_query, requires_grad=True)
        self.prompt_xvec_proj = nn.Linear(self.output_size, self.xvec_size)
        self.outside_prompt_dim = kwargs.get("outside_prompt_dim", None)
        if self.outside_prompt_dim is not None:
            self.outside_prompt_poj = nn.Linear(self.outside_prompt_dim, self.output_size)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor, speech_token: torch.Tensor,
                speech_token_lengths: torch.Tensor, audio: torch.Tensor, audio_lengths: torch.Tensor,
                xvec: Optional[torch.Tensor] = None, xvec_lengths: Optional[torch.Tensor] = None, **kwargs):
        text = text[:, :text_lengths.max()]
        speech_token = speech_token[:, :speech_token_lengths.max()]
        audio = audio[:, :audio_lengths.max()]
        batch_size = text.shape[0]

        # embed text inputs
        mask = (text != -1).float().unsqueeze(-1)
        text_emb = self.text_embedding(torch.clamp(text, min=0)) * mask
        text_emb_lengths = text_lengths

        prompt, prompt_lens, text_emb, text_emb_lengths = self.split_prompt(
            text_emb, text_emb_lengths, text, text_lengths
        )
        if "outside_prompt" in kwargs and "outside_prompt_lengths" in kwargs:
            prompt = kwargs["outside_prompt"]
            prompt_lens = kwargs["outside_prompt_lengths"]
            prompt = self.outside_prompt_poj(prompt)
            hint_once("use outside_prompt", "outside_prompt")

        # textual prompt xvec
        if self.text_mel_xvec_rand_ratios[0] > 0:
            prompt_xvec = self.spk_aggregator(
                prompt, prompt_lens,
                self.spk_query.expand([batch_size, -1, -1]), torch.tensor([1] * batch_size).to(prompt_lens)
            )[0]
            prompt_xvec = self.prompt_xvec_proj(prompt_xvec)
        else:
            prompt_xvec = torch.zeros([batch_size, 1, self.xvec_size]).to(text_emb)

        # mel prompt xvec
        if self.text_mel_xvec_rand_ratios[1] > 0:
            audio_rand_lens = torch.rand_like(audio_lengths, dtype=torch.float32) * (
                        self.audio_prompt_lens[1] - self.audio_prompt_lens[0]) + self.audio_prompt_lens[0]
            audio_rand_lens = (audio_rand_lens * audio_lengths).round().long()
            audio_rand_lens = torch.clamp(audio_rand_lens, min=round(self.mel_spec_fn.sampling_rate * 0.5))
            audio_prompt = [audio[i, :audio_rand_lens[i]] for i in range(batch_size)]
            audio_rand_lens = torch.tensor([x.shape[0] for x in audio_prompt]).to(text_lengths)
            audio_prompt = pad_list(audio_prompt, 0.0)
            mel_feat, feat_lens = self.mel_spec_fn(audio_prompt, audio_rand_lens)
            mel_xvec = self.mel_xvec_fn(mel_feat).unsqueeze(1)
        else:
            mel_xvec = torch.zeros([batch_size, 1, self.xvec_size]).to(text_emb)

        if xvec is not None:
            # random select a xvec from xvec matrix
            xvec = xvec[:, :xvec_lengths.max()]
        else:
            xvec = torch.zeros([batch_size, 1, self.xvec_size]).to(text_emb)

        # random using prompt, mel and input xvecs
        mixup_rand = torch.tensor(self.text_mel_xvec_rand_ratios).to(text_emb).multinomial(batch_size,
                                                                                           replacement=True).unsqueeze(
            1).unsqueeze(2)
        hint_once(f"xvec mixup prob: {mixup_rand}", "xvec mixup prob")
        rand_xvec = (
                (mixup_rand == 0) * prompt_xvec +
                (mixup_rand == 1) * mel_xvec +
                (mixup_rand == 2) * xvec
        )
        rand_xvec_lens = torch.tensor([1] * batch_size).to(xvec_lengths)

        outs_tuple = self.text_encoder(text_emb, ilens=text_emb_lengths)
        text_enc = outs_tuple[0]
        text_enc_lens = text_emb_lengths

        states = dict(
            batch_size=float(batch_size),
            text_len=float(text_emb.shape[1]),
            speech_len=float(speech_token.shape[1]),
            token_text_ratio=float(speech_token.shape[1]) / float(text_emb.shape[1]),
        )
        # forward AM model
        am_retvals = self.am_model.force_align_text(
            speech_token, speech_token_lengths,
            text_enc, text_enc_lens,
            rand_xvec, rand_xvec_lens,
            **kwargs
        )
        am_loss, aligned_token_emb, am_states = am_retvals
        # update AM states
        for key, val in am_states.items():
            states[f"am_{key}"] = val

        # forward FM model
        fm_loss, fm_states, _ = self.fm_model.forward(
            aligned_token_emb, speech_token_lengths,
            audio, audio_lengths,
            rand_xvec, rand_xvec_lens,
            **kwargs,
        )
        # update FM states
        for key, val in fm_states.items():
            states[f"fm_{key}"] = val

        loss = am_loss + fm_loss
        states["loss"] = loss.item()

        loss, states, weight = force_gatherable((loss, states, batch_size), loss.device)

        return loss, states, weight

    def inference(self, text: torch.Tensor, text_lengths: torch.Tensor, xvec: Optional[torch.Tensor] = None,
                  xvec_lengths: Optional[torch.Tensor] = None, **kwargs):
        blank_penalty = kwargs.get("blank_penalty", 0.0)
        sampling = kwargs.get("sampling", "greedy")
        prompt_dict = kwargs.get("prompt_dict", {})
        prompt_token = prompt_dict.get("prompt_token", (None, None))
        prompt_audio = prompt_dict.get("prompt_audio", (None, None))
        # fully un-causal mode
        use_causal_prob = kwargs.get("use_causal_prob", 1.0)
        # embed text inputs
        mask = (text != -1).float().unsqueeze(-1)
        text_emb = self.text_embedding(torch.clamp(text, min=0)) * mask
        text_emb_lengths = text_lengths
        batch_size = text.shape[0]

        prompt, prompt_lens, text_emb, text_emb_lengths = self.split_prompt(
            text_emb, text_emb_lengths, text, text_lengths
        )
        if "outside_prompt" in kwargs:
            prompt = kwargs["outside_prompt"].to(text.device)
            if "outside_prompt_lengths" in kwargs:
                prompt_lens = kwargs["outside_prompt_lengths"]
            else:
                prompt_lens = torch.tensor([prompt.shape[1]]).to(text_lengths)
            prompt = self.outside_prompt_poj(prompt)
            hint_once("use outside_prompt", "outside_prompt")

        if xvec is not None:
            # using the xvec
            hint_once("using speaker embedding for slot.", "use_spk_emb")
            xvec = xvec[:, :xvec_lengths.max()]
        else:
            # textual prompt xvec
            hint_once("using textual prompt for slot.", "use_spk_emb")
            prompt_xvec = self.spk_aggregator(
                prompt, prompt_lens,
                self.spk_query.expand([batch_size, -1, -1]), torch.tensor([1] * batch_size).to(prompt_lens)
            )[0]
            xvec = self.prompt_xvec_proj(prompt_xvec)
            xvec_lengths = torch.tensor([1] * batch_size).to(text_lengths)

        outs_tuple = self.text_encoder(text_emb, ilens=text_emb_lengths)
        text_enc = outs_tuple[0]
        text_enc_lens = text_emb_lengths

        # forward AM model
        tokens, aligned_token_emb, aligned_token_lens = self.am_model.inference(
            text_enc, text_enc_lens,
            xvec, xvec_lengths,
            sampling=sampling,
            blank_penalty=blank_penalty,
            text_is_embedding=True,
            return_hidden=True,
            use_causal_prob=use_causal_prob,
        )
        if isinstance(tokens, tuple):
            tokens, fa_tokens = tokens

        # forward FM model
        feat = self.fm_model.inference(
            aligned_token_emb, aligned_token_lens,
            xvec, xvec_lengths,
            prompt=dict(
                prompt_text=prompt_token,
                prompt_audio=prompt_audio,
            ),
            **kwargs,
        )
        feat = self.rms_rescale_feat(feat)

        return tokens, feat

    def streaming_inference(self, text: torch.Tensor, text_lengths: torch.Tensor, xvec: Optional[torch.Tensor] = None,
                            xvec_lengths: Optional[torch.Tensor] = None, **kwargs):
        device = text.device
        use_causal_prob = kwargs.get("use_causal_prob", 1.0)
        # streaming related config
        chunk_size = kwargs.get("streaming_chunk_size", 4)
        chunk_size_maxium = kwargs.get("chunk_size_maxium", 16)
        try:
            lookahead_size = self.am_model.encoder.pre_lookahead_len
        except AttributeError:
            lookahead_size = 0
        hint_once(f"chunk_size={chunk_size}, chunk_size_maxium={chunk_size_maxium}, "
                  f"pre lookahead size={lookahead_size}.",
                  "pre_lookahead_len")
        given_rtf = kwargs.get("given_rtf", 0.5)

        blank_penalty = kwargs.get("blank_penalty", 0.0)
        sampling = kwargs.get("sampling", "greedy")
        prompt_dict = kwargs.get("prompt_dict", {})
        prompt_token = list(prompt_dict.get("prompt_token", [None, None]))
        prompt_audio = list(prompt_dict.get("prompt_audio", [None, None]))
        streaming_mode = kwargs.get("streaming_mode", "v2")

        ftype = self.text_embedding.weight.dtype
        if prompt_token[0] is None:
            prompt_token[0] = torch.zeros([1, 0, self.output_size], device=device, dtype=ftype)
            prompt_token[1] = torch.tensor([0], device=device, dtype=torch.long)
        if prompt_audio[0] is None:
            prompt_audio[0] = torch.zeros(
                [1, 0, self.fm_model.mel_extractor.num_mels],
                device=device, dtype=ftype
            )
            prompt_audio[1] = torch.tensor([0], device=device, dtype=torch.long)

        # embed text inputs
        mask = (text != -1).float().unsqueeze(-1)
        text_emb = self.text_embedding(torch.clamp(text, min=0)) * mask
        text_emb_lengths = text_lengths

        batch_size = text.shape[0]

        prompt, prompt_lens, text_emb, text_emb_lengths = self.split_prompt(
            text_emb, text_emb_lengths, text, text_lengths
        )
        if "outside_prompt" in kwargs:
            prompt = kwargs["outside_prompt"].to(device)
            if "outside_prompt_lengths" in kwargs:
                prompt_lens = kwargs["outside_prompt_lengths"]
            else:
                prompt_lens = torch.tensor([prompt.shape[1]]).to(text_lengths)
            prompt = self.outside_prompt_poj(prompt)
            hint_once("use outside_prompt", "outside_prompt")

        if xvec is not None:
            # using speaker embedding
            hint_once("using speaker embedding for slot.", "use_spk_emb")
            xvec = xvec[:, :xvec_lengths.max()]
        else:
            # textual prompt xvec
            hint_once("using textual prompt for slot.", "use_spk_emb")
            prompt_xvec = self.spk_aggregator(
                prompt, prompt_lens,
                self.spk_query.expand([batch_size, -1, -1]), torch.tensor([1] * batch_size).to(prompt_lens)
            )[0]
            xvec = self.prompt_xvec_proj(prompt_xvec)
            xvec_lengths = torch.tensor([1] * batch_size).to(text_lengths)

        chunk_id = 0
        chunk_start = 0
        while True:
            _st_time = time.time()
            _size = max(int(round(chunk_size / (given_rtf ** chunk_id))), chunk_size_maxium)
            chunk_end = chunk_start + _size
            chunk_text_emb = text_emb[:, :chunk_end + lookahead_size]
            chunk_text_emb_lengths = torch.tensor([chunk_text_emb.shape[1]], dtype=torch.long, device=device)

            text_enc_st_time = time.time()
            outs_tuple = self.text_encoder(chunk_text_emb, ilens=chunk_text_emb_lengths)
            if chunk_id == 0:
                logging.info(f"text_enc cost time: {(time.time() - text_enc_st_time) * 1000.0:.2f} ms")
            text_enc = outs_tuple[0]
            text_enc_lens = chunk_text_emb_lengths

            # forward AM model
            am_st_time = time.time()
            tokens, aligned_token_emb, aligned_token_lens = self.am_model.inference(
                text_enc, text_enc_lens,
                xvec, xvec_lengths,
                sampling=sampling,
                blank_penalty=blank_penalty,
                text_is_embedding=True,
                return_hidden=True,
                use_causal_prob=use_causal_prob,
            )
            if chunk_id == 0:
                logging.info(f"am cost time: {(time.time() - am_st_time) * 1000.0:.2f} ms")
            token_hop_len, mel_hop_len = 0, 0
            if isinstance(tokens, tuple):
                tokens, fa_tokens = tokens
                token_hop_len = self.get_hop_lens(fa_tokens, lookahead_size)
                mel_hop_len = int(round(token_hop_len * self.fm_model.length_normalizer_ratio))

            # exclude empty tokens.
            if aligned_token_emb.shape[1] > prompt_token[0].shape[1]:
                cur_token = aligned_token_emb[:, prompt_token[0].shape[1]:]
                cur_token_len = aligned_token_lens - prompt_token[1]

                # v2: excluding lookahead tokens for not-last packages
                if streaming_mode == "v2":
                    if chunk_end + lookahead_size < text_emb.shape[1]:
                        cur_token = cur_token[:, :cur_token.shape[1] - token_hop_len, :]
                        cur_token_len = cur_token_len - token_hop_len

                # forward FM model
                fm_st_time = time.time()
                feat = self.fm_model.inference(
                    cur_token, cur_token_len,
                    xvec, xvec_lengths,
                    prompt=dict(
                        prompt_text=prompt_token,
                        prompt_audio=prompt_audio,
                    ),
                    **kwargs,
                )
                if chunk_id == 0:
                    logging.info(f"fm cost time: {(time.time() - fm_st_time) * 1000.0:.2f} ms")
                feat = self.rms_rescale_feat(feat)
                cost = time.time() - _st_time
                if chunk_id == 0:
                    logging.info(f"First package delay: {cost * 1000.0:.2f}ms")
                print_token = tokens.cpu().squeeze().tolist()
                logging.info(f"pack {chunk_id}: valid_tokens: {print_token[:len(print_token) - token_hop_len]}, "
                             f"pad_tokens: {print_token[len(print_token) - token_hop_len:]}.")

                if streaming_mode == "v1":
                    # v1: excluding lookahead parts for not-last packages
                    if chunk_end + lookahead_size < text_emb.shape[1]:
                        cur_token = cur_token[:, :cur_token.shape[1] - token_hop_len, :]
                        feat = feat[:, :, :feat.shape[2] - mel_hop_len]

                if streaming_mode == "v2":
                    # v2: reback token and mel feat
                    if chunk_end + lookahead_size < text_emb.shape[1]:
                        text_reback = 2 if chunk_id == 0 else 4
                        token_hop_len_2 = self.get_hop_lens(fa_tokens, lookahead_size + text_reback)
                        token_reback = token_hop_len_2 - token_hop_len
                        cur_token = cur_token[:, :cur_token.shape[1] - token_reback, :]
                        feat_reback = int(round(token_reback * self.fm_model.length_normalizer_ratio))
                        feat = feat[:, :, :feat.shape[2] - feat_reback]
                        chunk_end = chunk_end - text_reback

                # update values and lens of prompt token and audio
                prompt_token[1] = prompt_token[1] + cur_token.shape[1]
                prompt_token[0] = torch.concat([prompt_token[0], cur_token], dim=1)
                prompt_audio[1] = prompt_audio[1] + feat.shape[2]
                prompt_audio[0] = torch.concat([prompt_audio[0], feat.transpose(1, 2)], dim=1)

            chunk_id += 1
            chunk_start = chunk_end
            if chunk_end + lookahead_size >= text_emb.shape[1]:
                break

        return tokens, prompt_audio[0].transpose(1, 2)

    def cross_fade(self, pre: torch.Tensor, feat: torch.Tensor, hop_size: int):
        if pre is not None:
            hop_len = min(hop_size, feat.shape[1], pre.shape[1])
            sin_wind = torch.tensor(np.sin((np.arange(hop_len * 2) + 1) / (hop_len * 2 + 1) * np.pi)[None, :, None]).to(feat)
            cf_overlap = ((pre * sin_wind[:, -hop_len:] +
                           feat[:, :hop_len] * sin_wind[:, :hop_len]) /
                          (sin_wind[:, :hop_len] + sin_wind[:, -hop_len:]))
            feat[:, :hop_len] = cf_overlap

        return feat

    def streaming_one_step(
            self, text: torch.Tensor, text_lengths: torch.Tensor,
            xvec: Optional[torch.Tensor] = None, xvec_lengths: Optional[torch.Tensor] = None,
            chunk_idx=0,
            **kwargs
    ):
        device = text.device
        use_causal_prob = kwargs.get("use_causal_prob", 1.0)
        # streaming related config
        chunk_size = kwargs.get("streaming_chunk_size", 4)
        chunk_size_maxium = kwargs.get("chunk_size_maxium", 16)
        lookahead_size = self.am_model.encoder.pre_lookahead_len
        hint_once(f"chunk_size={chunk_size}, chunk_size_maxium={chunk_size_maxium}, "
                  f"pre lookahead size={lookahead_size}.",
                  "pre_lookahead_len")
        wav_vocoder = kwargs.get("vocoder", None)
        blank_penalty = kwargs.get("blank_penalty", 0.0)
        sampling = kwargs.get("sampling", "greedy")
        prompt_dict = kwargs.get("prompt_dict", {})
        prompt_token, pre_token_lb = list(prompt_dict.get("prompt_token", ([None, None], 0)))
        prompt_audio, pre_feat_lb = list(prompt_dict.get("prompt_audio", ([None, None], 0)))

        ftype = self.text_embedding.weight.dtype
        if prompt_token[0] is None:
            prompt_token[0] = torch.zeros([1, 0, self.output_size], device=device, dtype=ftype)
            prompt_token[1] = torch.tensor([0], device=device, dtype=torch.long)
        if prompt_audio[0] is None:
            prompt_audio[0] = torch.zeros(
                [1, 0, self.fm_model.mel_extractor.num_mels],
                device=device, dtype=ftype
            )
            prompt_audio[1] = torch.tensor([0], device=device, dtype=torch.long)

        # embed text inputs
        mask = (text != -1).float().unsqueeze(-1)
        text_emb = self.text_embedding(torch.clamp(text, min=0)) * mask
        text_emb_lengths = text_lengths

        batch_size = text.shape[0]

        prompt, prompt_lens, text_emb, text_emb_lengths = self.split_prompt(
            text_emb, text_emb_lengths, text, text_lengths
        )
        if "outside_prompt" in kwargs:
            prompt = kwargs["outside_prompt"].to(device)
            if "outside_prompt_lengths" in kwargs:
                prompt_lens = kwargs["outside_prompt_lengths"]
            else:
                prompt_lens = torch.tensor([prompt.shape[1]]).to(text_lengths)
            prompt = self.outside_prompt_poj(prompt)
            hint_once("use outside_prompt", "outside_prompt")

        if xvec is not None:
            # using speaker embedding
            hint_once("using speaker embedding for slot.", "use_spk_emb")
            xvec = xvec[:, :xvec_lengths.max()]
        else:
            # textual prompt xvec
            hint_once("using textual prompt for slot.", "use_spk_emb")
            prompt_xvec = self.spk_aggregator(
                prompt, prompt_lens,
                self.spk_query.expand([batch_size, -1, -1]), torch.tensor([1] * batch_size).to(prompt_lens)
            )[0]
            xvec = self.prompt_xvec_proj(prompt_xvec)
            xvec_lengths = torch.tensor([1] * batch_size).to(text_lengths)

        chunk_text_emb = text_emb
        chunk_text_emb_lengths = torch.tensor([chunk_text_emb.shape[1]], dtype=torch.long, device=device)

        outs_tuple = self.text_encoder(chunk_text_emb, ilens=chunk_text_emb_lengths)
        text_enc = outs_tuple[0]
        text_enc_lens = chunk_text_emb_lengths

        # forward AM model
        tokens, aligned_token_emb, aligned_token_lens = self.am_model.inference(
            text_enc, text_enc_lens,
            xvec, xvec_lengths,
            sampling=sampling,
            blank_penalty=blank_penalty,
            text_is_embedding=True,
            return_hidden=True,
            use_causal_prob=use_causal_prob,
        )
        tokens, fa_tokens = tokens
        token_hop_len = self.get_hop_lens(fa_tokens, lookahead_size)

        cur_token, feat, wav = None, None, None
        token_reback, feat_reback = pre_token_lb, pre_feat_lb
        # generate feat, exclude empty tokens.
        if aligned_token_emb.shape[1] > prompt_token[0].shape[1]:
            # need synthesize extra overlap parts
            cur_token = aligned_token_emb[:, prompt_token[0].shape[1] - pre_token_lb:]
            cur_token_len = aligned_token_lens - prompt_token[1] + pre_token_lb

            # v2: excluding lookahead tokens for not-last packages
            if text[0, -1] != self.endofprompt_token_id+1:
                cur_token = cur_token[:, :cur_token.shape[1] - token_hop_len, :]
                cur_token_len = cur_token_len - token_hop_len

            if cur_token_len[0] < 1:
                return None, None, None, (prompt_token, pre_token_lb), (prompt_audio, pre_feat_lb)
            # forward FM model
            # set_all_random_seed(0)
            feat = self.fm_model.inference(
                cur_token, cur_token_len,
                xvec, xvec_lengths,
                prompt=dict(
                    prompt_text=[prompt_token[0][:, :-pre_token_lb], prompt_token[1] - pre_token_lb],
                    prompt_audio=[prompt_audio[0][:, :-pre_feat_lb], prompt_audio[1] - pre_feat_lb],
                ),
                **kwargs,
            )
            feat = self.rms_rescale_feat(feat)
            print_token = tokens.cpu().squeeze(0).squeeze(-1).tolist()
            logging.info(f"valid_tokens: {print_token[:len(print_token) - token_hop_len]}, "
                         f"pad_tokens: {print_token[len(print_token) - token_hop_len:]}.")
            if prompt_audio[0].shape[1] > 0:
                feat = self.cross_fade(prompt_audio[0], feat.transpose(1, 2), pre_feat_lb).transpose(1, 2)

            wav = wav_vocoder.inference(feat.transpose(1, 2))
            if prompt_audio[0].shape[1] > 0:
                pre_wav = wav_vocoder.inference(prompt_audio[0])
                pre_wav_lb = int(1.0 / pre_token_lb * wav_vocoder.sample_rate)
                wav = self.cross_fade(pre_wav, wav, pre_wav_lb)

            prompt_token = [
                torch.cat([prompt_token[0][:, :-pre_token_lb], cur_token], dim=1),
                prompt_token[1] + cur_token_len - pre_token_lb,
            ]
            prompt_audio = [
                torch.cat([prompt_audio[0][:, :-pre_feat_lb], feat.transpose(1, 2)], dim=1),
                prompt_audio[1] + feat.shape[2] - pre_feat_lb
            ]

            # v2: reback token and mel feat
            if text[0, -1] != self.endofprompt_token_id+1:
                text_reback = 2 if chunk_idx == 0 else 4
                token_hop_len_2 = self.get_hop_lens(fa_tokens, lookahead_size + text_reback)
                token_reback = token_hop_len_2 - token_hop_len
                cur_token = cur_token[:, :cur_token.shape[1] - token_reback, :]
                feat_reback = int(token_reback * self.fm_model.length_normalizer_ratio)
                feat = feat[:, :, :feat.shape[2] - feat_reback]
                wav_reback = int(1.0 / token_reback * wav_vocoder.sample_rate)
                wav = wav[:, :wav.shape[1] - wav_reback]

        return cur_token, feat, wav, (prompt_token, token_reback), (prompt_audio, feat_reback)
