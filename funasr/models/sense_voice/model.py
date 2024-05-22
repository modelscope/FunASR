import logging
from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional
import types
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from torch.cuda.amp import autocast
from funasr.metrics.compute_acc import compute_accuracy
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.train_utils.device_funcs import force_gatherable
from . import whisper_lib as whisper
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils.datadir_writer import DatadirWriter

from funasr.register import tables


@tables.register("model_classes", "SenseVoice")
class SenseVoice(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        dims = kwargs.get("dims", {})
        dims = whisper.model.ModelDimensions(**dims)
        model = whisper.model.Whisper(dims=dims)

        # encoder
        model.encoder.downsample_rate = kwargs.get("downsample_rate", 4)
        model.encoder.use_padmask = kwargs.get("use_padmask", True)
        from .encoder import sense_voice_encode_forward

        model.encoder.forward = types.MethodType(sense_voice_encode_forward, model.encoder)

        # decoder
        model.decoder.use_padmask = kwargs.get("use_padmask", True)
        from .decoder import sense_voice_decode_forward

        model.decoder.forward = types.MethodType(sense_voice_decode_forward, model.decoder)

        self.model = model

        self.encoder_output_size = self.model.dims.n_audio_state

        self.activation_checkpoint = kwargs.get("activation_checkpoint", False)
        self.ignore_id = kwargs.get("ignore_id", -1)
        self.vocab_size = kwargs.get("vocab_size", -1)
        self.length_normalized_loss = kwargs.get("length_normalized_loss", True)
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

        specaug = kwargs.get("specaug", None)
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**kwargs.get("specaug_conf", {}))
        self.specaug = specaug

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)

        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        if self.activation_checkpoint:
            from torch.utils.checkpoint import checkpoint

            encoder_out, encoder_out_lens = checkpoint(
                self.encode, speech, speech_lengths, use_reentrant=False
            )
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, target_mask=target_mask
        )
        loss = loss_att
        stats = {}
        stats["acc"] = acc_att
        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens = self.model.encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)
        stats = {}

        # 1. Forward decoder
        decoder_out = self.model.decoder(
            x=ys_pad, xa=encoder_out, hlens=encoder_out_lens, ys_in_lens=ys_pad_lens
        )

        # 2. Compute attention loss
        mask = torch.ones_like(ys_pad) * (-1)
        ys_pad_mask = (ys_pad * target_mask + mask * (1 - target_mask)).to(torch.int64)
        ys_pad_mask[ys_pad_mask == 0] = -1
        loss_att = self.criterion_att(decoder_out[:, :-1, :], ys_pad_mask[:, 1:])

        with torch.no_grad():
            preds = torch.argmax(decoder_out, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], ys_pad_mask[:, 1:], ignore_label=self.ignore_id
            )

        return loss_att, acc_att, None, None

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])[0, :, :]
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        DecodingOptions = kwargs.get("DecodingOptions", {})
        task = DecodingOptions.get("task", "ASR")
        if isinstance(task, str):
            task = [task]
        task = "".join([f"<|{x}|>" for x in task])
        initial_prompt = kwargs.get("initial_prompt", f"<|startoftranscript|>{task}")
        DecodingOptions["initial_prompt"] = initial_prompt

        language = DecodingOptions.get("language", None)
        language = None if language == "auto" else language
        DecodingOptions["language"] = language

        DecodingOptions["vocab_path"] = kwargs["tokenizer_conf"].get("vocab_path", None)

        if "without_timestamps" not in DecodingOptions:
            DecodingOptions["without_timestamps"] = True

        options = whisper.DecodingOptions(**DecodingOptions)

        result = whisper.decode(self.model, speech, options)
        text = f"{result.text}"
        results = []
        result_i = {"key": key[0], "text": text}

        results.append(result_i)

        return results, meta_data


@tables.register("model_classes", "SenseVoiceRWKV")
class SenseVoiceRWKV(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        dims = kwargs.get("dims", {})
        dims = whisper.model.ModelDimensions(**dims)
        model = whisper.model.Whisper(dims=dims)

        # encoder
        model.encoder.downsample_rate = kwargs.get("downsample_rate", 4)
        model.encoder.use_padmask = kwargs.get("use_padmask", True)
        from .encoder import sense_voice_encode_forward

        model.encoder.forward = types.MethodType(sense_voice_encode_forward, model.encoder)

        # decoder
        del model.decoder
        decoder = kwargs.get("decoder", "SenseVoiceDecoder")
        decoder_class = tables.decoder_classes.get(decoder)
        decoder = decoder_class(
            n_vocab=dims.n_vocab,
            n_ctx=dims.n_text_ctx,
            n_state=dims.n_text_state,
            n_head=dims.n_text_head,
            n_layer=dims.n_text_layer,
            **kwargs.get("decoder_conf"),
        )
        model.decoder = decoder

        self.model = model

        self.encoder_output_size = self.model.dims.n_audio_state

        self.activation_checkpoint = kwargs.get("activation_checkpoint", False)
        self.ignore_id = kwargs.get("ignore_id", -1)
        self.vocab_size = kwargs.get("vocab_size", -1)
        self.length_normalized_loss = kwargs.get("length_normalized_loss", True)
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

        specaug = kwargs.get("specaug", None)
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**kwargs.get("specaug_conf", {}))
        self.specaug = specaug

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)

        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size, frames, _ = speech.shape
        _, text_tokens = text.shape

        if self.activation_checkpoint:
            from torch.utils.checkpoint import checkpoint

            encoder_out, encoder_out_lens = checkpoint(
                self.encode, speech, speech_lengths, use_reentrant=False
            )
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, target_mask=target_mask
        )
        loss = loss_att
        stats = {}
        stats["acc"] = acc_att
        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size
        stats["batch_size_x_frames"] = frames * batch_size
        stats["batch_size_real_frames"] = speech_lengths.sum().item()
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        stats["batch_size_x_tokens"] = text_tokens * batch_size
        stats["batch_size_real_tokens"] = text_lengths.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]
        stats["batch_size_x_frames_plus_tokens"] = (text_tokens + frames) * batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):
            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens = self.model.encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)
        stats = {}

        # 1. Forward decoder
        # ys_pad: [sos, task, lid, text, eos]
        decoder_out = self.model.decoder(
            x=ys_pad, xa=encoder_out, hlens=encoder_out_lens, ys_in_lens=ys_pad_lens
        )

        # 2. Compute attention loss
        mask = torch.ones_like(ys_pad) * (-1)  # [sos, task, lid, text, eos]: [-1, -1, -1, -1]
        ys_pad_mask = (ys_pad * target_mask + mask * (1 - target_mask)).to(
            torch.int64
        )  # [sos, task, lid, text, eos]: [0, 0, 1, 1, 1] + [-1, -1, 0, 0, 0]
        ys_pad_mask[ys_pad_mask == 0] = -1  # [-1, -1, lid, text, eos]
        # decoder_out: [sos, task, lid, text]
        # ys_pad_mask: [-1, lid, text, eos]
        loss_att = self.criterion_att(decoder_out[:, :-1, :], ys_pad_mask[:, 1:])

        with torch.no_grad():
            preds = torch.argmax(decoder_out, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], ys_pad_mask[:, 1:], ignore_label=self.ignore_id
            )

        return loss_att, acc_att, None, None

    def init_beam_search(
        self,
        **kwargs,
    ):
        from .search import BeamSearch

        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        scorers.update(
            decoder=self.model.decoder,
            length_bonus=LengthBonus(self.vocab_size),
        )

        weights = dict(
            decoder=1.0,
            ctc=0.0,
            lm=0.0,
            ngram=0.0,
            length_bonus=kwargs.get("penalty", 0.0),
        )
        beam_search = BeamSearch(
            beam_size=kwargs.get("beam_size", 5),
            weights=weights,
            scorers=scorers,
            sos=None,
            eos=None,
            vocab_size=self.vocab_size,
            token_list=None,
            pre_beam_score_key="full",
        )

        self.beam_search = beam_search

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        # init beamsearch
        if not hasattr(self, "beam_search") or self.beam_search is None:
            logging.info("enable beam_search")
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])[0, :, :]
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        DecodingOptions = kwargs.get("DecodingOptions", {})
        task = DecodingOptions.get("task", "ASR")
        if isinstance(task, str):
            task = [task]
        task = "".join([f"<|{x}|>" for x in task])
        initial_prompt = kwargs.get("initial_prompt", f"<|startoftranscript|>{task}")

        language = DecodingOptions.get("language", None)
        language = None if language == "auto" else language

        sos = f"{initial_prompt}<|{language}|>" if language is not None else initial_prompt
        sos_int = tokenizer.encode(sos, allowed_special="all")
        eos = kwargs.get("model_conf").get("eos")
        eos_int = tokenizer.encode(eos, allowed_special="all")
        self.beam_search.sos = sos_int
        self.beam_search.eos = eos_int[0]

        # Paramterts for rich decoding
        self.beam_search.emo_unk = tokenizer.encode(
            DecodingOptions.get("emo_unk_token", "<|SPECIAL_TOKEN_1|>"), allowed_special="all"
        )[0]
        self.beam_search.emo_unk_score = 1
        self.beam_search.emo_tokens = tokenizer.encode(
            DecodingOptions.get("emo_target_tokens", "<|HAPPY|><|SAD|><|ANGRY|>"),
            allowed_special="all",
        )
        self.beam_search.emo_scores = DecodingOptions.get("emo_target_threshold", [0.1, 0.1, 0.1])

        self.beam_search.event_bg_token = tokenizer.encode(
            DecodingOptions.get("gain_tokens_bg", "<|Speech|><|BGM|><|Applause|><|Laughter|>"),
            allowed_special="all",
        )
        self.beam_search.event_ed_token = tokenizer.encode(
            DecodingOptions.get("gain_tokens_ed", "<|/Speech|><|/BGM|><|/Applause|><|/Laughter|>"),
            allowed_special="all",
        )
        self.beam_search.event_score_ga = DecodingOptions.get("gain_tokens_score", [1, 1, 1, 1])

        encoder_out, encoder_out_lens = self.encode(
            speech[None, :, :].permute(0, 2, 1), speech_lengths
        )

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=encoder_out[0],
            maxlenratio=kwargs.get("maxlenratio", 0.0),
            minlenratio=kwargs.get("minlenratio", 0.0),
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        b, n, d = encoder_out.size()
        for i in range(b):

            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # # remove blank symbol id, which is assumed to be 0
                # token_int = list(
                #     filter(
                #         lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                #     )
                # )

                # Change integer-ids to tokens
                # token = tokenizer.ids2tokens(token_int)
                text = tokenizer.decode(token_int)

                result_i = {"key": key[i], "text": text}
                results.append(result_i)

                if ibest_writer is not None:
                    # ibest_writer["token"][key[i]] = " ".join(token)
                    ibest_writer["text"][key[i]] = text

        return results, meta_data


@tables.register("model_classes", "SenseVoiceFSMN")
class SenseVoiceFSMN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        dims = kwargs.get("dims", {})
        dims = whisper.model.ModelDimensions(**dims)
        model = whisper.model.Whisper(dims=dims)

        # encoder
        model.encoder.downsample_rate = kwargs.get("downsample_rate", 4)
        model.encoder.use_padmask = kwargs.get("use_padmask", True)
        from .encoder import sense_voice_encode_forward

        model.encoder.forward = types.MethodType(sense_voice_encode_forward, model.encoder)

        # decoder
        del model.decoder
        decoder = kwargs.get("decoder", "SenseVoiceDecoder")
        decoder_class = tables.decoder_classes.get(decoder)
        decoder = decoder_class(
            n_vocab=dims.n_vocab,
            n_ctx=dims.n_text_ctx,
            n_state=dims.n_text_state,
            n_head=dims.n_text_head,
            n_layer=dims.n_text_layer,
            **kwargs.get("decoder_conf"),
        )
        model.decoder = decoder

        self.model = model

        self.encoder_output_size = self.model.dims.n_audio_state

        self.activation_checkpoint = kwargs.get("activation_checkpoint", False)
        self.ignore_id = kwargs.get("ignore_id", -1)
        self.vocab_size = dims.n_vocab
        self.length_normalized_loss = kwargs.get("length_normalized_loss", True)
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=kwargs.get("lsm_weight", 0.0),
            normalize_length=self.length_normalized_loss,
        )

        specaug = kwargs.get("specaug", None)
        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**kwargs.get("specaug_conf", {}))
        self.specaug = specaug

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)

        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size, frames, _ = speech.shape
        _, text_tokens = text.shape

        if self.activation_checkpoint:
            from torch.utils.checkpoint import checkpoint

            encoder_out, encoder_out_lens = checkpoint(
                self.encode, speech, speech_lengths, use_reentrant=False
            )
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, target_mask=target_mask
        )
        loss = loss_att
        stats = {}
        stats["acc"] = acc_att
        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size
        stats["batch_size_x_frames"] = frames * batch_size
        stats["batch_size_real_frames"] = speech_lengths.sum().item()
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        stats["batch_size_x_tokens"] = text_tokens * batch_size
        stats["batch_size_real_tokens"] = text_lengths.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]
        stats["batch_size_x_frames_plus_tokens"] = (text_tokens + frames) * batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):
            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Forward encoder
        encoder_out, encoder_out_lens = self.model.encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)
        stats = {}

        # 1. Forward decoder
        decoder_out = self.model.decoder(
            x=ys_pad, xa=encoder_out, hlens=encoder_out_lens, ys_in_lens=ys_pad_lens
        )
        # decoder_out, _ = self.model.decoder(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        # 2. Compute attention loss
        mask = torch.ones_like(ys_pad) * (-1)
        ys_pad_mask = (ys_pad * target_mask + mask * (1 - target_mask)).to(torch.int64)
        ys_pad_mask[ys_pad_mask == 0] = -1
        loss_att = self.criterion_att(decoder_out[:, :-1, :], ys_pad_mask[:, 1:])

        with torch.no_grad():
            preds = torch.argmax(decoder_out, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], ys_pad_mask[:, 1:], ignore_label=self.ignore_id
            )

        return loss_att, acc_att, None, None

    def init_beam_search(
        self,
        **kwargs,
    ):
        from .search import BeamSearch

        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        scorers.update(
            decoder=self.model.decoder,
            length_bonus=LengthBonus(self.vocab_size),
        )

        weights = dict(
            decoder=1.0,
            ctc=0.0,
            lm=0.0,
            ngram=0.0,
            length_bonus=kwargs.get("penalty", 0.0),
        )
        beam_search = BeamSearch(
            beam_size=kwargs.get("beam_size", 5),
            weights=weights,
            scorers=scorers,
            sos=None,
            eos=None,
            vocab_size=self.vocab_size,
            token_list=None,
            pre_beam_score_key="full",
        )

        self.beam_search = beam_search

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        # init beamsearch
        if not hasattr(self, "beam_search") or self.beam_search is None:
            logging.info("enable beam_search")
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )

            if (
                isinstance(kwargs.get("data_type", None), (list, tuple))
                and len(kwargs.get("data_type", [])) > 1
            ):
                audio_sample_list, text_token_int_list = audio_sample_list
                text_token_int = text_token_int_list[0]
            else:
                text_token_int = None

            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])[0, :, :]
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        DecodingOptions = kwargs.get("DecodingOptions", {})
        task = DecodingOptions.get("task", "ASR")
        if isinstance(task, str):
            task = [task]
        task = "".join([f"<|{x}|>" for x in task])
        initial_prompt = kwargs.get("initial_prompt", f"<|startoftranscript|>{task}")

        language = DecodingOptions.get("language", None)
        language = None if language == "auto" else language

        sos = f"{initial_prompt}<|{language}|>" if language is not None else initial_prompt
        sos_int = tokenizer.encode(sos, allowed_special="all")
        eos = kwargs.get("model_conf").get("eos")
        eos_int = tokenizer.encode(eos, allowed_special="all")
        self.beam_search.sos = sos_int
        self.beam_search.eos = eos_int[0]

        # Paramterts for rich decoding
        self.beam_search.emo_unk = tokenizer.encode(
            DecodingOptions.get("emo_unk_token", "<|SPECIAL_TOKEN_1|>"), allowed_special="all"
        )[0]
        self.beam_search.emo_unk_score = 1
        self.beam_search.emo_tokens = tokenizer.encode(
            DecodingOptions.get("emo_target_tokens", "<|HAPPY|><|SAD|><|ANGRY|>"),
            allowed_special="all",
        )
        self.beam_search.emo_scores = DecodingOptions.get("emo_target_threshold", [0.1, 0.1, 0.1])

        self.beam_search.event_bg_token = tokenizer.encode(
            DecodingOptions.get("gain_tokens_bg", "<|Speech|><|BGM|><|Applause|><|Laughter|>"),
            allowed_special="all",
        )
        self.beam_search.event_ed_token = tokenizer.encode(
            DecodingOptions.get("gain_tokens_ed", "<|/Speech|><|/BGM|><|/Applause|><|/Laughter|>"),
            allowed_special="all",
        )
        self.beam_search.event_score_ga = DecodingOptions.get("gain_tokens_score", [1, 1, 1, 1])

        encoder_out, encoder_out_lens = self.encode(
            speech[None, :, :].permute(0, 2, 1), speech_lengths
        )

        if text_token_int is not None:
            i = 0
            results = []
            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"]

            # 1. Forward decoder
            ys_pad = torch.tensor(sos_int + text_token_int, dtype=torch.int64).to(kwargs["device"])[
                None, :
            ]
            ys_pad_lens = torch.tensor([len(sos_int + text_token_int)], dtype=torch.int64).to(
                kwargs["device"]
            )[None, :]
            decoder_out = self.model.decoder(
                x=ys_pad, xa=encoder_out, hlens=encoder_out_lens, ys_in_lens=ys_pad_lens
            )

            token_int = decoder_out.argmax(-1)[0, :].tolist()
            text = tokenizer.decode(token_int)

            result_i = {"key": key[i], "text": text}
            results.append(result_i)

            if ibest_writer is not None:
                # ibest_writer["token"][key[i]] = " ".join(token)
                ibest_writer["text"][key[i]] = text
            return results, meta_data

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=encoder_out[0],
            maxlenratio=kwargs.get("maxlenratio", 0.0),
            minlenratio=kwargs.get("minlenratio", 0.0),
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        b, n, d = encoder_out.size()
        for i in range(b):

            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # # remove blank symbol id, which is assumed to be 0
                # token_int = list(
                #     filter(
                #         lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                #     )
                # )

                # Change integer-ids to tokens
                # token = tokenizer.ids2tokens(token_int)
                text = tokenizer.decode(token_int)

                result_i = {"key": key[i], "text": text}
                results.append(result_i)

                if ibest_writer is not None:
                    # ibest_writer["token"][key[i]] = " ".join(token)
                    ibest_writer["text"][key[i]] = text

        return results, meta_data


@tables.register("model_classes", "SenseVoiceSANM")
class SenseVoiceSANM(nn.Module):

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        encoder: str = None,
        encoder_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
        input_size: int = 80,
        vocab_size: int = -1,
        ignore_id: int = -1,
        blank_id: int = 0,
        sos: int = 1,
        eos: int = 2,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        # extract_feats_in_collect_stats: bool = True,
        share_embedding: bool = False,
        # preencoder: Optional[AbsPreEncoder] = None,
        # postencoder: Optional[AbsPostEncoder] = None,
        **kwargs,
    ):

        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)

        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()

        decoder_class = tables.decoder_classes.get(decoder)
        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            **decoder_conf,
        )

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id

        self.specaug = specaug

        self.encoder = encoder

        self.decoder = decoder

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        self.activation_checkpoint = kwargs.get("activation_checkpoint", False)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)

        # import pdb;
        # pdb.set_trace()
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size, frames, _ = speech.shape
        _, text_tokens = text.shape

        if self.activation_checkpoint:
            from torch.utils.checkpoint import checkpoint

            encoder_out, encoder_out_lens = checkpoint(
                self.encode, speech, speech_lengths, use_reentrant=False
            )
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
            encoder_out, encoder_out_lens, text, text_lengths, target_mask=target_mask
        )

        loss = loss_att
        stats = {}
        stats["acc"] = acc_att
        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size
        stats["batch_size_x_frames"] = frames * batch_size
        stats["batch_size_real_frames"] = speech_lengths.sum().item()
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        stats["batch_size_x_tokens"] = text_tokens * batch_size
        stats["batch_size_real_tokens"] = text_lengths.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]
        stats["batch_size_x_frames_plus_tokens"] = (text_tokens + frames) * batch_size

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((text_lengths + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ):
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            # Data augmentation
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

        # Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)

        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, (tuple, list)):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        **kwargs,
    ):
        target_mask = kwargs.get("target_mask", None)
        stats = {}

        # 1. Forward decoder
        ys_pad[ys_pad == -1] = 0
        decoder_out = self.decoder(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        if isinstance(decoder_out, (list, tuple)):
            decoder_out = decoder_out[0]

        # 2. Compute attention loss
        mask = torch.ones_like(ys_pad) * (-1)
        ys_pad_mask = (ys_pad * target_mask + mask * (1 - target_mask)).to(torch.int64)
        ys_pad_mask[ys_pad_mask == 0] = -1
        loss_att = self.criterion_att(decoder_out[:, :-1, :], ys_pad_mask[:, 1:])

        with torch.no_grad():
            preds = torch.argmax(decoder_out, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], ys_pad_mask[:, 1:], ignore_label=self.ignore_id
            )

        return loss_att, acc_att, None, None

    def init_beam_search(
        self,
        **kwargs,
    ):
        from .search import BeamSearch

        from funasr.models.transformer.scorers.length_bonus import LengthBonus

        # 1. Build ASR model
        scorers = {}

        scorers.update(
            decoder=self.decoder,
            length_bonus=LengthBonus(self.vocab_size),
        )

        weights = dict(
            decoder=1.0,
            ctc=0.0,
            lm=0.0,
            ngram=0.0,
            length_bonus=kwargs.get("penalty", 0.0),
        )
        beam_search = BeamSearch(
            beam_size=kwargs.get("beam_size", 5),
            weights=weights,
            scorers=scorers,
            sos=None,
            eos=None,
            vocab_size=self.vocab_size,
            token_list=None,
            pre_beam_score_key="full",
        )

        self.beam_search = beam_search

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        # init beamsearch
        if not hasattr(self, "beam_search") or self.beam_search is None:
            logging.info("enable beam_search")
            self.init_beam_search(**kwargs)
            self.nbest = kwargs.get("nbest", 1)

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )

            if (
                isinstance(kwargs.get("data_type", None), (list, tuple))
                and len(kwargs.get("data_type", [])) > 1
            ):
                audio_sample_list, text_token_int_list = audio_sample_list
                text_token_int = text_token_int_list[0]
            else:
                text_token_int = None

            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])[0, :, :]
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        DecodingOptions = kwargs.get("DecodingOptions", {})
        task = DecodingOptions.get("task", "ASR")
        if isinstance(task, str):
            task = [task]
        task = "".join([f"<|{x}|>" for x in task])
        initial_prompt = kwargs.get("initial_prompt", f"<|startoftranscript|>{task}")

        language = DecodingOptions.get("language", None)
        language = None if language == "auto" else language

        sos = f"{initial_prompt}<|{language}|>" if language is not None else initial_prompt
        sos_int = tokenizer.encode(sos, allowed_special="all")
        eos = kwargs.get("model_conf").get("eos")
        eos_int = tokenizer.encode(eos, allowed_special="all")
        self.beam_search.sos = sos_int
        self.beam_search.eos = eos_int[0]

        # Paramterts for rich decoding
        self.beam_search.emo_unk = tokenizer.encode(
            DecodingOptions.get("emo_unk_token", "<|SPECIAL_TOKEN_1|>"), allowed_special="all"
        )[0]
        self.beam_search.emo_unk_score = 1
        self.beam_search.emo_tokens = tokenizer.encode(
            DecodingOptions.get("emo_target_tokens", "<|HAPPY|><|SAD|><|ANGRY|>"),
            allowed_special="all",
        )
        self.beam_search.emo_scores = DecodingOptions.get("emo_target_threshold", [0.1, 0.1, 0.1])

        self.beam_search.event_bg_token = tokenizer.encode(
            DecodingOptions.get("gain_tokens_bg", "<|Speech|><|BGM|><|Applause|><|Laughter|>"),
            allowed_special="all",
        )
        self.beam_search.event_ed_token = tokenizer.encode(
            DecodingOptions.get("gain_tokens_ed", "<|/Speech|><|/BGM|><|/Applause|><|/Laughter|>"),
            allowed_special="all",
        )
        self.beam_search.event_score_ga = DecodingOptions.get("gain_tokens_score", [1, 1, 1, 1])

        encoder_out, encoder_out_lens = self.encode(
            speech[None, :, :].permute(0, 2, 1), speech_lengths
        )

        if text_token_int is not None:
            i = 0
            results = []
            ibest_writer = None
            if kwargs.get("output_dir") is not None:
                if not hasattr(self, "writer"):
                    self.writer = DatadirWriter(kwargs.get("output_dir"))
                ibest_writer = self.writer[f"1best_recog"]

            # 1. Forward decoder
            ys_pad = torch.tensor(sos_int + text_token_int, dtype=torch.int64).to(kwargs["device"])[
                None, :
            ]
            ys_pad_lens = torch.tensor([len(sos_int + text_token_int)], dtype=torch.int64).to(
                kwargs["device"]
            )[None, :]
            decoder_out = self.model.decoder(
                x=ys_pad, xa=encoder_out, hlens=encoder_out_lens, ys_in_lens=ys_pad_lens
            )

            token_int = decoder_out.argmax(-1)[0, :].tolist()
            text = tokenizer.decode(token_int)

            result_i = {"key": key[i], "text": text}
            results.append(result_i)

            if ibest_writer is not None:
                # ibest_writer["token"][key[i]] = " ".join(token)
                ibest_writer["text"][key[i]] = text
            return results, meta_data

        # c. Passed the encoder result and the beam search
        nbest_hyps = self.beam_search(
            x=encoder_out[0],
            maxlenratio=kwargs.get("maxlenratio", 0.0),
            minlenratio=kwargs.get("minlenratio", 0.0),
        )

        nbest_hyps = nbest_hyps[: self.nbest]

        results = []
        b, n, d = encoder_out.size()
        for i in range(b):

            for nbest_idx, hyp in enumerate(nbest_hyps):
                ibest_writer = None
                if kwargs.get("output_dir") is not None:
                    if not hasattr(self, "writer"):
                        self.writer = DatadirWriter(kwargs.get("output_dir"))
                    ibest_writer = self.writer[f"{nbest_idx + 1}best_recog"]

                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()

                # # remove blank symbol id, which is assumed to be 0
                # token_int = list(
                #     filter(
                #         lambda x: x != self.eos and x != self.sos and x != self.blank_id, token_int
                #     )
                # )

                # Change integer-ids to tokens
                # token = tokenizer.ids2tokens(token_int)
                text = tokenizer.decode(token_int)

                result_i = {"key": key[i], "text": text}
                results.append(result_i)

                if ibest_writer is not None:
                    # ibest_writer["token"][key[i]] = " ".join(token)
                    ibest_writer["text"][key[i]] = text

        return results, meta_data
