import logging
import os.path
import torchaudio
from typing import Union, Dict, List, Tuple, Optional

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import re
from funasr.models.scama.utils import sequence_mask
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.ctc.ctc import CTC
from funasr.models.transformer.utils.add_sos_eos import add_sos_eos
from funasr.metrics.compute_acc import th_accuracy, compute_accuracy
from funasr.metrics.common import ErrorCalculator
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
import traceback

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@tables.register("model_classes", "LLMASR")
class LLMASR(nn.Module):
    """ """

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
        ctc: str = None,
        ctc_conf: dict = None,
        ctc_weight: float = 0.5,
        llm: str = None,
        llm_conf: dict = None,
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
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        if hub == "ms":
            from funasr import AutoModel

            model = AutoModel(model=audio_encoder, model_revision="master")
            # frontend = model.kwargs.get("frontend")
            audio_encoder_output_size = model.model.encoder_output_size

            audio_encoder = model.model.model.encoder

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        if freeze:
            for name, param in audio_encoder.named_parameters():
                param.requires_grad = False
            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        hub = llm_conf.get("hub", "hf")
        self.llm = None
        if hub == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

            init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")

            model = AutoModelForCausalLM.from_pretrained(
                init_param_path,
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
            )
            freeze = llm_conf.get("freeze", True)
            if freeze:
                for name, param in model.named_parameters():
                    param.requires_grad = False
                model.eval()
            self.llm = model

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor = adaptor_class(**audio_adaptor_conf)

        self.audio_adaptor = audio_adaptor

        self.blank_id = blank_id
        self.sos = sos if sos is not None else vocab_size - 1
        self.eos = eos if eos is not None else vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.specaug = specaug
        self.normalize = normalize

        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_ids: torch.Tensor,
        label_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        if len(text_lengths.size()) > 1:
            text_lengths = text_lengths[:, 0]
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size = speech.shape[0]

        # audio encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # audio_adaptor
        encoder_out = self.audio_adaptor(encoder_out)

        input_ids[input_ids == -1] = 0
        input_ids[input_ids == -100] = 0
        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(input_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)

        if audio_mask is not None:
            batch_size, token_num, dims = inputs_embeds.shape
            _, l, _ = encoder_out.shape
            # [audio, bos, prompt, input, pad]
            encoder_outs_pad = F.pad(encoder_out, (0, 0, 0, token_num - l, 0, 0), value=0.0)
            inputs_embeds = encoder_outs_pad * audio_mask[:, :, None] + inputs_embeds * (
                1.0 - audio_mask[:, :, None]
            )

        model_outputs = self.llm(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels_ids
        )
        loss = model_outputs.loss

        stats = {}
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())

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
        speech = speech.permute(0, 2, 1)
        res = self.audio_encoder(speech)
        if isinstance(res, (list, tuple)):
            encoder_out, encoder_out_lens = res[0], res[1]
        else:
            encoder_out, encoder_out_lens = res, speech_lengths
        return encoder_out, encoder_out_lens

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        prompt = kwargs.get("prompt", "Transcribe speech to text.")

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

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
                fs=frontend.fs,
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
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        # Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # adaptor
        encoder_out = self.audio_adaptor(encoder_out)

        prompt_pre = "USER: \nINSTRUCTION: {}\nINPUT: ".format(prompt)
        prompt_ids = tokenizer.encode(prompt_pre)
        prompt_length = len(prompt_ids)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64).to(kwargs["device"])

        if hasattr(self.llm.model, "embed_tokens"):
            inputs_embeds = self.llm.model.embed_tokens(prompt_ids)
        elif hasattr(self.llm.model.model, "embed_tokens"):
            inputs_embeds = self.llm.model.model.embed_tokens(prompt_ids)
        else:
            inputs_embeds = self.llm.model.model.model.embed_tokens(prompt_ids)

        inputs_embeds = torch.cat(
            (inputs_embeds[None, :, :], encoder_out), dim=1
        )  # [prompt, audio]
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(
            kwargs["device"]
        )

        preds = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_length=kwargs.get("max_length", 200),
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        text = tokenizer.batch_decode(preds, add_special_tokens=False, skip_special_tokens=True)

        text = text[0].split(": ")[-1]
        text = text.strip()

        # preds = torch.argmax(model_outputs.logits, -1)

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        result_i = {"key": key[0], "text": text}
        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = text

        return results, meta_data


@tables.register("model_classes", "LLMASR2")
class LLMASR2(nn.Module):
    """ """

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
        ctc: str = None,
        ctc_conf: dict = None,
        ctc_weight: float = 0.5,
        llm: str = None,
        llm_conf: dict = None,
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

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        if hub == "ms":
            from funasr import AutoModel

            model = AutoModel(model=audio_encoder, model_revision="master")
            # frontend = model.kwargs.get("frontend")
            audio_encoder_output_size = model.model.encoder_output_size

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

        if freeze:
            for name, param in audio_encoder.named_parameters():
                if freeze_layer_num > 0:
                    idx = re.search(r"\.\d+\.", name)
                    if idx is not None:
                        beg, end = idx.regs[0]
                        layer_id = int(name[beg + 1 : end - 1])
                        if layer_id < freeze_layer_num:
                            param.requires_grad = False
                    elif "ln_post." not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
        )
        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = llm_dim
        audio_adaptor = adaptor_class(**audio_adaptor_conf)
        init_param_path = audio_adaptor_conf.get("init_param_path", None)
        if init_param_path is not None:
            src_state = torch.load(init_param_path, map_location="cpu")
            flag = audio_adaptor.load_state_dict(src_state, strict=False)
            logging.info(f"Loading audio_adaptor ckpt: {init_param_path}, status: {flag}")

        self.audio_adaptor = audio_adaptor

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_ids: torch.Tensor,
        fbank_beg: torch.Tensor,
        fbank_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb;
        # pdb.set_trace()
        if len(speech_lengths.size()) > 1:
            speech_lengths = speech_lengths[:, 0]

        batch_size, frames, _ = speech.shape

        with torch.cuda.amp.autocast(enabled=False):
            # audio encoder
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape
        fbank_mask[fbank_mask < 0] = 0
        fbank_fake_lens = fbank_mask.sum(-1).to(torch.int32)
        # _, l, _ = encoder_out.shape
        for batch_idx in range(batch_size):

            fbank_fake_len = fbank_fake_lens[batch_idx].item()
            fbank_beg_idx = fbank_beg[batch_idx, 0].item()
            min_len = min(fbank_fake_len, inputs_embeds.shape[1] - fbank_beg_idx)

            try:
                inputs_embeds[batch_idx, fbank_beg_idx : fbank_beg_idx + min_len, :] = encoder_out[
                    batch_idx, :min_len, :
                ]
            except Exception as e:
                logging.error(f"{str(e)}, {traceback.format_exc()}")
                logging.info(
                    f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, min_len: {min_len}, fbank_fake_len: {fbank_fake_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens[batch_idx].item()}"
                )
                fbank_fake_len = encoder_out_lens[batch_idx].item()
                min_len = min(fbank_fake_len, min_len)
                inputs_embeds[batch_idx, fbank_beg_idx : fbank_beg_idx + min_len, :] = encoder_out[
                    batch_idx, :min_len, :
                ]

        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids == -1] = -100
            attention_mask[attention_mask < 0] = 0
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
                labels=labels_ids,
            )
            loss = model_outputs.loss

        stats = {}
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size
        stats["batch_size_x_frames"] = frames * batch_size
        stats["batch_size_real_frames"] = speech_lengths.sum().item()
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):

        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        input_ids, labels, source_ids, target_ids, fbank, fbank_lens, fbank_mask, fbank_beg = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):

            source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

            splits = pattern.split(source_input)
            source_ids_i = []
            fbank_mask_i = []
            fbank_beg_i = []
            fbank_lens_i = []
            # target_ids_i = []
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids_i += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    if sub_str.startswith("!"):
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(sub_str[1:], fs=frontend.fs)
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as e:
                            logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,
                            is_final=True,
                        )  # speech: [b, T, d]

                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = (
                            speech_lengths.sum().item()
                            * frontend.frame_shift
                            * frontend.lfr_n
                            / 1000
                        )

                        if hasattr(frontend, "permute") and not frontend.permute:
                            # if kwargs.get("permute", True):
                            speech = speech.permute(0, 2, 1)

                        if (
                            kwargs.get("dataset_conf", {}).get("audio_encoder_downsample_rate", 1)
                            == 4
                        ):
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                        elif (
                            kwargs.get("dataset_conf", {}).get("audio_encoder_downsample_rate", 1)
                            == 1
                        ):
                            olens = speech_lengths[0].item()

                        sub_token_len = (olens - 1) // kwargs.get("dataset_conf", {}).get(
                            "audio_adaptor_downsample_rate", 1
                        ) + 1
                        sub_token = [0] * sub_token_len
                        fbank_beg_i = [len(source_ids_i)]
                        source_ids_i += sub_token
                        fbank_mask_i += [1] * len(sub_token)

            source_mask = [-100] * len(source_ids_i)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_ids += source_ids_i + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            fbank_beg.append(fbank_beg_i)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
        source_ids = torch.tensor(source_ids_i, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        fbank = speech[0, :, :]
        fbank_lens = speech_lengths
        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)

        output = {
            "speech": fbank[None, :, :],
            "speech_lengths": fbank_lens[:, None],
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None,],
            "input_ids": input_ids[None, :],
            "attention_mask": attention_mask[None, :],
            "labels_ids": labels[None, :],
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        meta_data = {}
        prompt = kwargs.get("prompt", None)

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        batch = to_device(output, kwargs["device"])

        # audio encoder
        speech = batch["speech"]
        speech_lengths = batch["speech_lengths"][:, 0]
        # fp16
        if kwargs.get("fp16", False):
            speech = speech.to(torch.float16)
        elif kwargs.get("bf16", False):
            speech = speech.to(torch.bfloat16)
        # audio encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # audio_adaptor
        encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        if not kwargs.get("tearchforing", False):
            input_ids = source_ids
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape
        fbank_beg = batch["fbank_beg"]
        for batch_idx in range(batch_size):

            min_len = encoder_out_lens[batch_idx].item()
            fbank_beg_idx = fbank_beg[batch_idx]
            inputs_embeds[batch_idx, fbank_beg_idx : fbank_beg_idx + min_len, :] = encoder_out[
                batch_idx, :min_len, :
            ]

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.cuda.amp.autocast(
            enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            label = contents["assistant"][0]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])

            if not kwargs.get("tearchforing", False):

                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds, max_new_tokens=kwargs.get("max_length", 512)
                )
                # generated_ids = [
                #     output_ids[len(input_id) :]
                #     for input_id, output_ids in zip(input_ids, generated_ids)
                # ]
                response = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=kwargs.get("skip_special_tokens", True)
                )[0]

                loss = None
            else:

                labels_ids = batch["labels_ids"]
                labels_ids[labels_ids == -1] = -100
                attention_mask = batch.get("attention_mask", None)
                # attention_mask = attention_mask.to(dtype_map[llm_dtype])
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels_ids
                )

                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1] :]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                loss = model_outputs.loss.item()

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {"key": key[0], "text": response, "text_tn": response_clean, "label": label}
        if loss is not None:
            result_i["loss"] = loss
        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response
            ibest_writer["label"][key[0]] = label
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data


@tables.register("model_classes", "LLMASR3")
class LLMASR3(LLMASR2):
    """ """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        return encoder_out, encoder_out_lens


@tables.register("model_classes", "LLMASR4")
class LLMASR4(nn.Module):
    """ """

    def __init__(
        self,
        specaug: str = None,
        specaug_conf: dict = None,
        normalize: str = None,
        normalize_conf: dict = None,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        decoder: str = None,
        decoder_conf: dict = None,
        ctc: str = None,
        ctc_conf: dict = None,
        ctc_weight: float = 0.5,
        llm: str = None,
        llm_conf: dict = None,
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

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        if hub == "ms":
            from funasr import AutoModel

            model = AutoModel(model=audio_encoder, model_revision="master")
            # frontend = model.kwargs.get("frontend")
            audio_encoder_output_size = model.model.encoder_output_size

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

        if freeze:
            for name, param in audio_encoder.named_parameters():
                if freeze_layer_num > 0:
                    idx = re.search(r"\.\d+\.", name)
                    if idx is not None:
                        beg, end = idx.regs[0]
                        layer_id = int(name[beg + 1 : end - 1])
                        if layer_id < freeze_layer_num:
                            param.requires_grad = False
                    elif "ln_post." not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
        )
        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = llm_dim
        audio_adaptor = adaptor_class(**audio_adaptor_conf)
        init_param_path = audio_adaptor_conf.get("init_param_path", None)
        if init_param_path is not None:
            src_state = torch.load(init_param_path, map_location="cpu")
            flag = audio_adaptor.load_state_dict(src_state, strict=False)
            logging.info(f"Loading audio_adaptor ckpt: {init_param_path}, status: {flag}")

        self.audio_adaptor = audio_adaptor

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

    def forward(
        self,
        speech: torch.Tensor = None,
        speech_lengths: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels_ids: torch.Tensor = None,
        fbank_beg: torch.Tensor = None,
        fbank_mask: torch.Tensor = None,
        **kwargs,
    ):
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb
        #
        # pdb.set_trace()
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        if speech is not None:
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape
            batch_size, token_num = input_ids.shape

            with torch.cuda.amp.autocast(enabled=False):
                # audio encoder
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

                # audio_adaptor
                encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

            batch_size, token_num, dims = inputs_embeds.shape
            fake_token_len = kwargs.get("fake_token_len")
            fake_token_len[fake_token_len < 0] = 0
            fbank_beg[fbank_beg < 0] = 0

            speech_idx = 0
            for batch_idx in range(batch_size):

                for turn_id in range(fbank_beg.shape[1]):
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    if fbank_beg_idx > 0:
                        speech_token_len = fake_token_len[batch_idx, turn_id]
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]

                        try:
                            inputs_embeds[
                                batch_idx, fbank_beg_idx : fbank_beg_idx + speech_token_len, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            # import pdb;
                            # pdb.set_trace()
                            speech_token_len = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :speech_token_len, :]
                            inputs_embeds[
                                batch_idx, fbank_beg_idx : fbank_beg_idx + speech_token_len, :
                            ] = speech_token

                        speech_idx += 1

        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids == -1] = -100
            attention_mask[attention_mask < 0] = 0
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
                labels=labels_ids,
            )
            loss = model_outputs.loss

        stats = {}
        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size
        stats["batch_size_speech"] = batch_size_speech
        stats["batch_size_x_frames"] = frames * batch_size_speech
        stats["batch_size_real_frames"] = speech_lengths.sum().item()
        stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]
        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        dialog_turns = (fbank_beg > 0).sum(-1)
        dialog_turns_max = torch.max(dialog_turns).int().item()
        dialog_turns_avg = dialog_turns.sum().item() / batch_size
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                if "audio" in item:
                    audio = item["audio"]
                    content = [content, audio]
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):

        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):

                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

            splits = pattern.split(source_input)
            source_ids = []
            fbank_i = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            fbank_lens_i = []
            speech, speech_lengths = [], []
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):  # !!: audio sample point
                            sub_str = audio
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(sub_str, fs=frontend.fs)
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as e:
                            logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,
                            is_final=True,
                        )  # speech: [b, T, d]

                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = (
                            speech_lengths.sum().item()
                            * frontend.frame_shift
                            * frontend.lfr_n
                            / 1000
                        )

                        if kwargs.get("permute", True):
                            speech = speech.permute(0, 2, 1)
                        if speech_lengths > kwargs.get("max_source_length", 5500):
                            # logging.info(
                            #     f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                            # )
                            badcase_flag = True

                        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                        olens = 1 + (olens - 3 + 2 * 1) // 2
                        fake_token_len_i = (olens - 1) // 2 + 1
                        fake_token = [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids)
                        source_ids += fake_token
                        fbank_mask_i += [1] * len(fake_token)

            fbank_beg += [fbank_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            source_mask = [-100] * len(source_ids)
            target_out = f"{target_out}<|im_end|>"
            target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        # fbank = speech[0, :, :]
        # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
            speech_lengths = torch.nn.utils.rnn.pad_sequence(
                fbank_lens, batch_first=True, padding_value=-1
            )
        else:
            speech = []
            speech_lengths = []
        output = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None,],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None,],
            "attention_mask": attention_mask[None,],
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def inference_prepare(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        meta_data = {}
        prompt = kwargs.get("prompt", None)

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        batch = to_device(output, kwargs["device"])

        # audio encoder
        speech = batch["speech"]
        if len(speech) > 0:
            speech_lengths = batch["speech_lengths"][:, 0]
            # fp16
            if kwargs.get("fp16", False):
                speech = speech.to(torch.float16)
            elif kwargs.get("bf16", False):
                speech = speech.to(torch.bfloat16)
            # audio encoder
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids

        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape

        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        speech_idx = 0
        for batch_idx in range(batch_size):

            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = encoder_out[speech_idx, :speech_token_len, :]

                    try:
                        inputs_embeds[
                            batch_idx, fbank_beg_idx : fbank_beg_idx + speech_token_len, :
                        ] = speech_token
                    except Exception as e:
                        #
                        logging.error(f"{str(e)}, {traceback.format_exc()}")
                        logging.info(
                            f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                        )
                        # import pdb;
                        # pdb.set_trace()
                        speech_token_len = encoder_out_lens[speech_idx].item()
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx, fbank_beg_idx : fbank_beg_idx + speech_token_len, :
                        ] = speech_token

                    speech_idx += 1
        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.cuda.amp.autocast(
            enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            label = contents["assistant"][-1]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])

            if not kwargs.get("tearchforing", False):

                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds, max_new_tokens=kwargs.get("max_length", 512)
                )
                # generated_ids = [
                #     output_ids[len(input_id) :]
                #     for input_id, output_ids in zip(input_ids, generated_ids)
                # ]
                response = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=kwargs.get("skip_special_tokens", True)
                )[0]

                loss = None
            else:

                labels_ids = batch["labels_ids"]
                labels_ids[labels_ids == -1] = -100
                attention_mask = batch.get("attention_mask", None)
                # attention_mask = attention_mask.to(dtype_map[llm_dtype])
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels_ids
                )

                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1] :]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                loss = model_outputs.loss.item()

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {"key": key[0], "text": response, "text_tn": response_clean, "label": label}
        if loss is not None:
            result_i["loss"] = loss
        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


@tables.register("model_classes", "LLMASR5")
class LLMASR5(nn.Module):
    """ """

    def __init__(
        self,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        audio_decoder: str = None,
        audio_decoder_conf: dict = None,
        **kwargs,
    ):

        super().__init__()

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        if hub == "ms":
            from funasr import AutoModel

            model = AutoModel(model=audio_encoder, model_revision="master")
            # frontend = model.kwargs.get("frontend")
            audio_encoder_output_size = model.model.encoder_output_size

            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )

            # self.frontend = frontend

        elif hub == "hf":
            pass
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))
        # if freeze_layer_num > 0:
        #     freeze_layer_num = range(freeze_layer_num)

        if freeze:
            for name, param in audio_encoder.named_parameters():
                if freeze_layer_num > 0:
                    idx = re.search(r"\.\d+\.", name)
                    if idx is not None:
                        beg, end = idx.regs[0]
                        layer_id = int(name[beg + 1 : end - 1])
                        if layer_id < freeze_layer_num:
                            param.requires_grad = False
                    elif "ln_post." not in name:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

            audio_encoder.eval()

        self.audio_encoder = audio_encoder

        # llm
        self.llm = None

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        init_param_path = llm_conf.get("init_param_path", "vicuna-7b-v1.5")

        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
            output_hidden_states=True,
        )
        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = llm_dim
        audio_adaptor = adaptor_class(**audio_adaptor_conf)
        init_param_path = audio_adaptor_conf.get("init_param_path", None)
        if init_param_path is not None:
            src_state = torch.load(init_param_path, map_location="cpu")
            flag = audio_adaptor.load_state_dict(src_state, strict=False)
            logging.info(f"Loading audio_adaptor ckpt: {init_param_path}, status: {flag}")

        self.audio_adaptor = audio_adaptor

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

        self.eos = kwargs.get("eos", 151645)

        # audio decoder related

        self.codebook_dim = audio_decoder_conf.get("codebook_dim", 1024)
        self.codebook_size = audio_decoder_conf.get("codebook_size", 4096)
        self.lm_out_voc_size = self.codebook_size + 1
        self.audio_decoder = self.build_audio_decoder(name=audio_decoder, conf=audio_decoder_conf)
        self.concat_emb_hidden = audio_decoder_conf.get("concat_emb_hidden", False)
        self.concat_emb_hidden_norm = audio_decoder_conf.get("concat_emb_hidden_norm", False)
        if self.concat_emb_hidden_norm:
            self.hidden_norm = LayerNorm(llm_dim)
            self.fusion_dropout = nn.Dropout(audio_decoder_conf.get("fusion_drop_rate", 0.0))
            self.emb_norm = LayerNorm(llm_dim)
            self.fusion_norm = LayerNorm(self.audio_decoder.embed_unit)
            self.fusion_act = Swish()
        audio_decoder_in_proj_dim = llm_dim * 2 if self.concat_emb_hidden else llm_dim
        self.audio_decoder_in_proj = torch.nn.Linear(
            audio_decoder_in_proj_dim, self.audio_decoder.embed_unit
        )
        self.codec_embedder = torch.nn.Embedding(self.codebook_size, self.codebook_dim)
        self.audio_decoder_embedding = torch.nn.Embedding(2, self.audio_decoder.embed_unit)
        self.ad_sos_eos = 0
        self.ad_task_id = 1
        self.ad_ignore_id = -1
        self.predict_nq = 1

        from .label_smoothing_loss import LabelSmoothingLoss

        self.criterion_ce = LabelSmoothingLoss(
            size=self.lm_out_voc_size // self.predict_nq,
            padding_idx=self.ad_ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
            reduction=False,
        )

        mel_decoder_name = kwargs.get("mel_decoder", None)
        mel_decoder_conf = kwargs.get("mel_decoder_conf", None)
        self.mel_decoder = self.build_mel_decoder(name=mel_decoder_name, conf=mel_decoder_conf)
        vocoder_name = kwargs.get("vocoder", None)
        vocoder_conf = kwargs.get("vocoder_conf", None)
        self.vocoder = self.build_vocoder(name=vocoder_name, conf=vocoder_conf)

    def build_mel_decoder(self, name: str, conf: dict):
        if name is None or conf is None:
            return None
        if name == "MaskedDiffWithXvec":
            from funasr.models.llm_asr.flow_matching import MaskedDiffWithXvec
            return MaskedDiffWithXvec(**conf)
        return None

    def build_vocoder(self, name: str, conf: dict):
        if name is None or conf is None:
            return None
        if name == "HifiGAN":
            from funasr.models.llm_asr.hifigan import HifiGan
            return HifiGan(**conf)
        return None

    def build_audio_decoder(self, name, conf):
        if name == "transformer":
            from funasr.models.llm_asr.transformer_lm import TransformerEmbedLM

            if "text_vocab_size" in conf:
                lm_model = TransformerEmbedLM(vocab_size=self.lm_out_voc_size, **conf)
            else:
                lm_model = TransformerEmbedLM(
                    vocab_size=self.lm_out_voc_size, text_vocab_size=self.lm_out_voc_size, **conf
                )
        else:
            raise TypeError(f"Unknown codec decoder type {name}")

        return lm_model

    def calc_dense_vector(self, codec, codec_lengths):
        """
        Args:
            codec: (B, T, Nq)
            codec_lengths: (B, )
        """
        mask = codec != self.ad_ignore_id
        return self.codec_embedder(codec * mask).sum(dim=-2) * mask

    def prepare_audio_decoder_io(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        codec: Optional[torch.Tensor] = None,
        codec_lengths: Optional[torch.Tensor] = None,
        need_targets: bool = True,
    ):
        """build inputs and targets for language model

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length, Dim)
            text_lengths: (Batch,)
            codec: (Batch, Length)
            codec_lengths: (Batch,)
            need_targets: bool, whether provide targets
        """

        if need_targets:
            assert (
                codec is not None and codec_lengths is not None
            ), "need_target=True, but codec or codec_length is None"

        sos_eos_emb = self.audio_decoder_embedding(
            torch.tensor([self.ad_sos_eos], dtype=torch.int64, device=text.device)
        )
        task_id_emb = self.audio_decoder_embedding(
            torch.tensor([self.ad_task_id], dtype=torch.int64, device=text.device)
        )
        codec_emb = None
        if codec is not None and codec_lengths is not None:
            codec_emb = self.calc_dense_vector(codec, codec_lengths)
        inputs_list = []
        for i, text_len in enumerate(text_lengths):
            one_input = [sos_eos_emb, text[i, :text_len], task_id_emb]
            if codec_emb is not None:
                one_input.append(codec_emb[i, : codec_lengths[i]])
            inputs_list.append(torch.cat(one_input, dim=0))
        llm_inputs = pad_list(inputs_list, 0.0)
        llm_lengths = text_lengths + 2
        if codec_emb is not None:
            llm_lengths = llm_lengths + codec_lengths

        if not need_targets:
            return llm_inputs, llm_lengths

        bb, tt = text.shape[0], codec_lengths.max() + 1
        llm_targets = -1 * torch.ones(
            [bb, tt, self.predict_nq], dtype=torch.int64, device=text.device
        )
        for i, codec_len in enumerate(codec_lengths):
            llm_targets[i, :codec_len] = codec[i, :codec_len]
            llm_targets[i, codec_len] = self.codebook_size + self.ad_sos_eos

        return (llm_inputs, llm_targets), (llm_lengths, codec_lengths + 1)

    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        codec: Optional[torch.Tensor] = None,
        codec_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length, Dim)
            text_lengths: (Batch,)
            codec: (Batch, Length)
            codec_lengths: (Batch,)
        """
        batch_size = text.size(0)
        # For data parallel
        text = text[:, : text_lengths.max()]
        codec = codec[:, : codec_lengths.max()]
        # text = self.audio_decoder_in_proj(text)

        # build inputs and targets for language model
        with autocast(False):
            (sequence, target), (x_lengths, y_lengths) = self.prepare_audio_decoder_io(
                text, text_lengths, codec, codec_lengths, need_targets=True
            )

        # 2a. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        sequence = sequence[:, : x_lengths.max()]
        target = target[:, : y_lengths.max()]
        y, _ = self.audio_decoder(sequence, x_lengths, text_lengths + 1)
        bb, tt = y.shape[0], y.shape[1]
        y = y.reshape(bb, tt, self.predict_nq, -1)
        # 2b. Extract real logits
        logits_list = []
        for i, (text_len, codec_len) in enumerate(zip(text_lengths, codec_lengths)):
            logits_list.append(y[i, text_len + 1 : text_len + 2 + codec_len])
        logits = pad_list(logits_list, 0.0)

        # 3. Calc negative log likelihood
        tt = logits.shape[1]
        nll = self.criterion_ce(
            logits.reshape(bb, tt * self.predict_nq, -1), target.reshape(bb, tt * self.predict_nq)
        )
        nll = nll.sum(-1)
        # nll: (BxL,) -> (BxL,)
        nll.masked_fill_(make_pad_mask(y_lengths * self.predict_nq).to(nll.device).view(-1), 0.0)
        # nll: (BxL,) -> (B, L)
        nll = nll.reshape(batch_size, -1).reshape(batch_size, tt, self.predict_nq)

        return nll, logits, target, codec_lengths + 1

    def forward(
        self,
        speech: torch.Tensor = None,
        speech_lengths: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels_ids: torch.Tensor = None,
        fbank_beg: torch.Tensor = None,
        fbank_mask: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        # import pdb
        #
        # pdb.set_trace()
        stats = {}
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        batch_size, token_num, dims = inputs_embeds.shape
        if speech is not None:
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            with torch.cuda.amp.autocast(enabled=False):
                # audio encoder
                encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

                # audio_adaptor
                encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

            fake_token_len = kwargs.get("fake_token_len")
            fake_token_len[fake_token_len < 0] = 0
            fbank_beg[fbank_beg < 0] = 0

            speech_idx = 0
            for batch_idx in range(batch_size):

                for turn_id in range(fbank_beg.shape[1]):
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    if fbank_beg_idx > 0:
                        speech_token_len = fake_token_len[batch_idx, turn_id]
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]

                        try:
                            inputs_embeds[
                                batch_idx, fbank_beg_idx : fbank_beg_idx + speech_token_len, :
                            ] = speech_token
                        except Exception as e:
                            #
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            # import pdb;
                            # pdb.set_trace()
                            speech_token_len = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :speech_token_len, :]
                            inputs_embeds[
                                batch_idx, fbank_beg_idx : fbank_beg_idx + speech_token_len, :
                            ] = speech_token

                        speech_idx += 1

            stats["batch_size_speech"] = batch_size_speech
            stats["batch_size_x_frames"] = frames * batch_size_speech
            stats["batch_size_real_frames"] = speech_lengths.sum().item()
            stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids == -1] = -100
            attention_mask[attention_mask < 0] = 0
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
                labels=labels_ids,
            )
            loss = model_outputs.loss

        codec = kwargs.get("codec")
        # codec_len = kwargs.get("codec_len")
        # if len(codec_len.size()) > 1:
        #     codec_len = codec_len[:, 0]
        codec_len = (codec > 0).sum(-1)
        hidden_states = model_outputs.hidden_states[-1].float()

        target_ids = []
        target_ids_len = []
        hidden_states_select = []
        for batch_idx in range(labels_ids.shape[0]):
            beg_i = 0
            end_i = 0
            for token_idx in range(labels_ids.shape[1]):
                token_int = labels_ids[batch_idx, token_idx].item()
                if token_int == self.eos:
                    target_ids_i = labels_ids[batch_idx, beg_i:end_i]
                    target_ids_len_i = end_i - beg_i
                    target_ids_len.append(target_ids_len_i)
                    target_ids.append(target_ids_i)
                    hidden_states_i = hidden_states[batch_idx, beg_i - 1 : end_i - 1, :]
                    hidden_states_select.append(hidden_states_i)
                    end_i += 1
                    beg_i = end_i
                    continue

                end_i += 1
                if token_int <= 0:
                    beg_i += 1

        target_ids = torch.nn.utils.rnn.pad_sequence(
            target_ids, batch_first=True, padding_value=-100
        )
        hidden_states_select = torch.nn.utils.rnn.pad_sequence(
            hidden_states_select, batch_first=True, padding_value=0.0
        )
        target_ids_len = torch.tensor(target_ids_len, dtype=torch.int32, device=input_ids.device)
        target_ids = target_ids.to(device=input_ids.device)
        target_ids[target_ids < 0] = 0
        target_emb = self.llm.model.get_input_embeddings()(target_ids)
        hidden_states_select = hidden_states_select.to(device=input_ids.device)
        if self.concat_emb_hidden:
            if not self.concat_emb_hidden_norm:
                hidden_states_select = torch.concat((hidden_states_select, target_emb), dim=-1)
                hidden_states_select = self.audio_decoder_in_proj(hidden_states_select)
            else:
                outs = self.hidden_norm(hidden_states_select)
                outs = self.fusion_dropout(self.fusion_act(outs))
                # emb = model_outputs.hidden_states[0]
                emb = self.fusion_dropout(self.fusion_act(self.emb_norm(target_emb)))
                outs = self.audio_decoder_in_proj(torch.cat([outs, emb], dim=-1))
                hidden_states_select = self.fusion_act(self.fusion_norm(outs))

        nll, logits, target, target_lengths = self.nll(
            hidden_states_select, target_ids_len, codec[:, :, None], codec_len
        )
        output_mask = (
            ~make_pad_mask(target_lengths, maxlen=target_lengths.max())
            .to(hidden_states_select.device)
            .unsqueeze(-1)
        )
        total, batch_size = output_mask.sum() * self.predict_nq, nll.shape[0] * self.predict_nq
        denom = total if self.length_normalized_loss else batch_size
        loss = (nll * output_mask).sum() / denom

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

            cc = logits.shape[-1]
            for i in range(self.predict_nq):
                acc = th_accuracy(
                    logits[:, :, i, :].reshape(-1, cc), target[:, :, i], self.ad_ignore_id
                )
                stats[f"codec_acc_{i + 1}"] = acc

        stats["loss"] = torch.clone(loss.detach())
        stats["batch_size"] = batch_size
        stats["batch_size_x_tokens"] = token_num * batch_size
        stats["batch_size_real_tokens"] = attention_mask.sum().item()
        stats["padding_tokens"] = stats["batch_size_x_tokens"] - stats["batch_size_real_tokens"]

        dialog_turns = (fbank_beg > 0).sum(-1)
        dialog_turns_max = torch.max(dialog_turns).int().item()
        dialog_turns_avg = dialog_turns.sum().item() / batch_size
        stats["dialog_turns_max"] = dialog_turns_max
        stats["dialog_turns_avg"] = dialog_turns_avg

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        if self.length_normalized_loss:
            batch_size = int((labels_ids > 0 + 1).sum())
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def data_template(self, data):
        system, user, assistant = [], [], []
        for i, item in enumerate(data):
            role = item["role"]
            content = item["content"]
            if role == "system":
                system.append(content)
            elif role == "user":
                user.append(content)
            elif role == "assistant":
                assistant.append(content)

        system = system * len(user)

        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        return contents

    def data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):

        system = contents["system"]
        user = contents["user"]
        assistant = contents["assistant"]
        pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")

        input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):
                break

            if i == 0:
                source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                source_input = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

            splits = pattern.split(source_input)
            source_ids = []
            fbank_i = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
            fbank_lens_i = []
            speech, speech_lengths = [], []
            for k, sub_str in enumerate(splits):
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_token = tokenizer.encode(sub_str)
                    source_ids += sub_token
                    fbank_mask_i += [0] * len(sub_token)
                else:
                    sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                        "<|endofspeech|>", ""
                    )
                    if sub_str.startswith("!"):
                        sub_str = sub_str[1:]
                        if sub_str.startswith("!"):  # !!bytes
                            sub_str = eval(sub_str[1:])
                        try:
                            time1 = time.perf_counter()
                            data_src = load_audio_text_image_video(sub_str, fs=frontend.fs)
                            time2 = time.perf_counter()
                            meta_data["load_data"] = f"{time2 - time1:0.3f}"
                        except Exception as e:
                            logging.error(f"Loading wav failed! {str(e)}, {traceback.format_exc()}")

                        speech, speech_lengths = extract_fbank(
                            data_src,
                            data_type=kwargs.get("data_type", "sound"),
                            frontend=frontend,
                            is_final=True,
                        )  # speech: [b, T, d]

                        time3 = time.perf_counter()
                        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
                        meta_data["batch_data_time"] = (
                            speech_lengths.sum().item()
                            * frontend.frame_shift
                            * frontend.lfr_n
                            / 1000
                        )

                        if kwargs.get("permute", True):
                            speech = speech.permute(0, 2, 1)
                        if speech_lengths > kwargs.get("max_source_length", 5500):
                            # logging.info(
                            #     f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                            # )
                            badcase_flag = True

                        olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                        olens = 1 + (olens - 3 + 2 * 1) // 2
                        fake_token_len_i = (olens - 1) // 2 + 1
                        fake_token = [0] * fake_token_len_i
                        fbank_beg_i = len(source_ids)
                        source_ids += fake_token
                        fbank_mask_i += [1] * len(fake_token)

            fbank_beg += [fbank_beg_i + len(input_ids)]
            fake_token_len += [fake_token_len_i]
            source_mask = [-100] * len(source_ids)
            splits = pattern.split(target_out)
            for k, sub_str in enumerate(splits):
                if len(sub_str) < 1:
                    continue
                if not sub_str.startswith("<|startofspeech|>"):
                    sub_str = f"{sub_str}<|im_end|>"
                    sub_token = tokenizer.encode(sub_str)
            target_ids = sub_token
            # target_out = f"{target_out}<|im_end|>"
            # target_ids = tokenizer.encode(target_out)
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

        # fbank = speech[0, :, :]
        # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
        fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
        fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
        fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)
        source_ids = torch.tensor(input_source_ids, dtype=torch.int64)
        target_ids = torch.tensor(target_ids, dtype=torch.int64)

        if len(fbank) > 0:
            speech = torch.nn.utils.rnn.pad_sequence(fbank, batch_first=True, padding_value=0.0)
            speech_lengths = torch.nn.utils.rnn.pad_sequence(
                fbank_lens, batch_first=True, padding_value=-1
            )
        else:
            speech = []
            speech_lengths = []
        output = {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "fbank_mask": fbank_mask[None, :],
            "fbank_beg": fbank_beg[None,],
            "fake_token_len": fake_token_len[None, :],
            "input_ids": input_ids[None,],
            "attention_mask": attention_mask[None,],
            "labels_ids": labels,
            "source_ids": source_ids[None, :],
            "target_ids": target_ids[None, :],
        }

        return output

    def inference_prepare(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        meta_data = {}
        prompt = kwargs.get("prompt", None)

        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        contents = self.data_template(data_in[0])
        output = self.data_load_speech(contents, tokenizer, frontend, meta_data=meta_data, **kwargs)
        batch = to_device(output, kwargs["device"])

        # audio encoder
        speech = batch["speech"]
        if len(speech) > 0:
            speech_lengths = batch["speech_lengths"][:, 0]
            # fp16
            if kwargs.get("fp16", False):
                speech = speech.to(torch.float16)
            elif kwargs.get("bf16", False):
                speech = speech.to(torch.bfloat16)
            # audio encoder
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

            # audio_adaptor
            encoder_out, encoder_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("tearchforing", False):
            input_ids = source_ids

        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)

        batch_size, token_num, dims = inputs_embeds.shape

        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        speech_idx = 0
        for batch_idx in range(batch_size):

            for turn_id in range(fbank_beg.shape[1]):
                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    speech_token_len = fake_token_len[batch_idx, turn_id]
                    speech_token = encoder_out[speech_idx, :speech_token_len, :]

                    try:
                        inputs_embeds[
                            batch_idx, fbank_beg_idx : fbank_beg_idx + speech_token_len, :
                        ] = speech_token
                    except Exception as e:
                        #
                        logging.error(f"{str(e)}, {traceback.format_exc()}")
                        logging.info(
                            f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                        )
                        # import pdb;
                        # pdb.set_trace()
                        speech_token_len = encoder_out_lens[speech_idx].item()
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx, fbank_beg_idx : fbank_beg_idx + speech_token_len, :
                        ] = speech_token

                    speech_idx += 1
        return inputs_embeds, contents, batch, source_ids, meta_data

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        inputs_embeds, contents, batch, source_ids, meta_data = self.inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.cuda.amp.autocast(
            enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            label = contents["assistant"][-1]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])

            generated_ids = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=kwargs.get("max_length", 512),
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            hidden_states = generated_ids[
                "hidden_states"
            ]  # hidden_states: (t1, t2, ..., tn, ..., tN), tn=(l1, l2, ..., ln, ..., lN), ln: shape: 1x1x3584

            token_num = len(hidden_states)
            hidden_states_select = torch.zeros((1, token_num, 3584), dtype=torch.float32).to(
                inputs_embeds.device
            )
            hidden_states_out_len = torch.tensor(
                [
                    token_num,
                ],
                dtype=torch.int32,
            ).to(inputs_embeds.device)
            for i in range(token_num):
                hidden_states_select[0, i, :] = hidden_states[i][-1][0, 0, :].to(torch.float32)

            target_ids = generated_ids["sequences"]
            target_emb = self.llm.model.get_input_embeddings()(target_ids)
            if self.concat_emb_hidden:
                if not self.concat_emb_hidden_norm:
                    hidden_states_select = torch.concat((hidden_states_select, target_emb), dim=-1)
                    hidden_states_select = self.audio_decoder_in_proj(hidden_states_select)
                else:
                    outs = self.hidden_norm(hidden_states_select)
                    outs = self.fusion_dropout(self.fusion_act(outs))
                    # emb = model_outputs.hidden_states[0]
                    emb = self.fusion_dropout(self.fusion_act(self.emb_norm(target_emb)))
                    outs = self.audio_decoder_in_proj(torch.cat([outs, emb], dim=-1))
                    hidden_states_select = self.fusion_act(self.fusion_norm(outs))

            speech_tokens = self.audio_decode(hidden_states_select, hidden_states_out_len)[
                :, :, 0
            ]  # 1xlx1: 2,10,1023

            # generated_ids = [
            #     output_ids[len(input_id) :]
            #     for input_id, output_ids in zip(input_ids, generated_ids)
            # ]
            response = tokenizer.batch_decode(
                target_ids, skip_special_tokens=kwargs.get("skip_special_tokens", True)
            )[0]

            loss = None

        # synthesize waveforms
        spk_emb = kwargs.get("spk_emb", None)
        feat, wav = self.synthesize_waveform(speech_tokens, spk_emb, inputs_embeds.device)

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

            self.write_mel_wav(kwargs.get("output_dir"), feat, wav, key[0])

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {
            "key": key[0],
            "text": response,
            "text_tn": response_clean,
            "label": label,
            "speech_tokens": speech_tokens,
        }
        if loss is not None:
            result_i["loss"] = loss
        results.append(result_i)

        speech_tokens_out = "<|startofspeech|>"
        for i in range(speech_tokens.shape[-1]):
            tmp = speech_tokens[0, i].item()
            speech_tokens_out += f"<|c{tmp}|>"
        speech_tokens_out += "<|endofspeech|><|im_end|>"
        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean
            ibest_writer["speech_tokens"][key[0]] = speech_tokens_out

        return results, meta_data

    def write_mel_wav(self, output_dir, feat, wav, key):
        out_dir = os.path.join(output_dir, "1best_recog", "mels")
        os.makedirs(out_dir, exist_ok=True)
        if feat is not None:
            feat = feat.cpu().numpy()[0]
            np.save(os.path.join(out_dir, f"{key}.npy"), feat)

        out_dir = os.path.join(output_dir, "1best_recog", "wavs")
        os.makedirs(out_dir, exist_ok=True)
        if wav is not None:
            path = os.path.join(out_dir, f"{key}.wav")
            torchaudio.save(
                path, wav[0], sample_rate=self.vocoder.sample_rate,
                encoding='PCM_S', bits_per_sample=16
            )

    def synthesize_waveform(self, speech_tokens, spk_emb, device):
        mel_feat, wav = None, None
        if self.mel_decoder is not None and spk_emb is not None:
            # mel_feat in BxCxT
            mel_feat = self.token2mel(speech_tokens, spk_emb, device)
            if self.vocoder is not None:
                wav = self.vocoder.inference(mel_feat.transpose(1, 2))

        return mel_feat, wav

    def token2mel(self, tokens: torch.Tensor, xvec: torch.Tensor, device: torch.device):
        xvec = torch.tensor(xvec).to(device).unsqueeze(0)
        xvec_lens = torch.tensor([xvec.shape[1]], device=device, dtype=torch.int64)
        token_lens = torch.tensor([tokens.shape[1]], device=device, dtype=torch.int64)
        feat = self.mel_decoder.inference(
            tokens, token_lens,
            xvec, xvec_lens,
            diff_steps=10,
            temperature=1.0,
            prompt=dict(
                prompt_text=(None, None),
                prompt_audio=(None, None)
            )
        )
        return feat

    def audio_decode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        min_length=None,
        max_length: int = 30 * 25,
        infer_cfg_ratio=None,
        decoding_length=None,
    ):
        # 1. encode text
        # text = self.audio_decoder_in_proj(text)
        device = text.device
        sos_eos_emb = self.audio_decoder_embedding(
            torch.tensor([[self.ad_sos_eos]], dtype=torch.int64, device=device)
        )
        task_id_emb = self.audio_decoder_embedding(
            torch.tensor([[self.ad_task_id]], dtype=torch.int64, device=device)
        )
        prompt = torch.cat([sos_eos_emb, text, task_id_emb], dim=1)
        seq_input = torch.zeros(
            [1, prompt.shape[1] + max_length, prompt.shape[2]],
            dtype=torch.float32, device=device
        )
        seq_input[:, :prompt.shape[1], :] = prompt
        out_tokens = torch.zeros([1, max_length, 1], dtype=torch.int64, device=device)
        out_token_len = 0
        prompt_len = prompt.shape[1]
        state, hit_eos = None, False
        for i in range(max_length):
            # use state for speedup
            pred, (state, _) = self.audio_decoder.score(seq_input[0, :prompt_len+out_token_len], state, prompt[0])

            # sampling all `nq` token ids
            pred = pred.reshape(self.predict_nq, -1)
            # normalize scores
            pred = torch.log_softmax(pred, dim=-1)
            if min_length is not None and i < min_length:
                pred[:, self.codebook_size + self.ad_sos_eos] = float(np.finfo(np.float32).min)
            top_ids = self.ras_sampling(pred[0], out_tokens[0, :out_token_len, 0])

            if torch.any(top_ids == (self.codebook_size + self.ad_sos_eos)):
                hit_eos = True
                break

            out_tokens[0, out_token_len, 0] = top_ids[0]
            seq_input[0, prompt_len + out_token_len, :] = self.codec_embedder(top_ids)[0]
            out_token_len += 1

        if decoding_length is None:
            return out_tokens
        else:
            return out_tokens, hit_eos

    # Repetition Aware Sampling in VALL-E 2
    def ras_sampling(
        self, weighted_scores, decoded_tokens, *, top_p=0.8, top_k=25, win_size=10, tau_r=0.1
    ):
        top_ids = self.nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
        rep_num = torch.sum(decoded_tokens[-win_size:] == top_ids).item()
        if rep_num >= win_size * tau_r:
            top_ids = self.random_sampling(weighted_scores)

        return top_ids

    def nucleus_sampling(self, weighted_scores, top_p=0.8, top_k=25):
        cum_prob = 0.0
        sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
        i = len(sorted_idx)
        for i in range(len(sorted_idx)):
            # sampling both top-p and numbers.
            if cum_prob < top_p and i < top_k:
                cum_prob += sorted_value[i]
            else:
                break
        prob = sorted_value[:i]
        indices = sorted_idx[:i]
        sampling_ids = prob.multinomial(1, replacement=True)
        top_ids = indices[sampling_ids]
        return top_ids

    def random_sampling(self, weighted_scores):
        top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
        return top_ids
