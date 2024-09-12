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
import math
from torch.nn import CrossEntropyLoss

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
from funasr.train_utils.set_all_random_seed import set_all_random_seed
import traceback
from pydub import AudioSegment
from io import BytesIO

try:
    import numpy as np
    from scipy.io import savemat
except:
    pass

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
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )
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
        llm_load_kwargs = llm_conf.get("load_kwargs", {})

        if not llm_conf.get("low_cpu", False):
            model = AutoModelForCausalLM.from_pretrained(
                init_param_path,
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
                **llm_load_kwargs,
            )
        else:
            import os

            if int(os.environ.get("RANK", 0)) == 0:
                model = AutoModelForCausalLM.from_pretrained(
                    init_param_path,
                    load_in_8bit=None,
                    device_map="cpu",
                    use_cache=None,
                    **llm_load_kwargs,
                )
            else:
                llm_config = AutoConfig.from_pretrained(init_param_path)
                model = AutoModelForCausalLM.from_config(llm_config)

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        logging.info(f"use_lora: {llm_conf.get('use_lora', False)}")
        if llm_conf.get("use_lora", False):
            from omegaconf import OmegaConf, DictConfig

            lora_conf = llm_conf.get("lora_conf", {})
            if isinstance(lora_conf, (OmegaConf, DictConfig)):
                lora_conf = OmegaConf.to_container(lora_conf, resolve=True)
            from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel

            lora_init_param_path = lora_conf.get("init_param_path", None)
            if lora_init_param_path is not None:
                logging.info(f"lora_init_param_path: {lora_init_param_path}")
                model = PeftModel.from_pretrained(model, lora_init_param_path)
                for name, param in model.named_parameters():
                    if not lora_conf.get("freeze_lora", False):
                        if "lora_" in name:
                            param.requires_grad = True
            else:
                peft_config = LoraConfig(**lora_conf)
                model = get_peft_model(model, peft_config)

            model.print_trainable_parameters()

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

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
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for name, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()

        self.audio_adaptor = audio_adaptor

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

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
        batch_size, token_num = input_ids.shape
        stats = {}
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        if speech is not None:
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
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

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

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
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

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
            llm_kwargs = kwargs.get("llm_kwargs", {})
            if not kwargs.get("tearchforing", False):

                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_length", 512),
                    **llm_kwargs,
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
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    **llm_kwargs,
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


@tables.register("model_classes", "LLMASR4_extract_kv")
class LLMASR4_extract_kv(nn.Module):
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
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )
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

        if not llm_conf.get("low_cpu", False):
            model = AutoModelForCausalLM.from_pretrained(
                init_param_path,
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
                output_hidden_states=llm_conf.get("output_hidden_states", True),
            )
        else:
            import os

            if int(os.environ.get("RANK", 0)) == 0:
                model = AutoModelForCausalLM.from_pretrained(
                    init_param_path,
                    load_in_8bit=None,
                    device_map="cpu",
                    use_cache=None,
                )
            else:
                llm_config = AutoConfig.from_pretrained(init_param_path)
                model = AutoModelForCausalLM.from_config(llm_config)

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        logging.info(f"use_lora: {llm_conf.get('use_lora', False)}")
        if llm_conf.get("use_lora", False):
            from omegaconf import OmegaConf, DictConfig

            lora_conf = llm_conf.get("lora_conf", {})
            if isinstance(lora_conf, (OmegaConf, DictConfig)):
                lora_conf = OmegaConf.to_container(lora_conf, resolve=True)
            from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel

            lora_init_param_path = lora_conf.get("init_param_path", None)
            if lora_init_param_path is not None:
                model = PeftModel.from_pretrained(model, lora_init_param_path)
            else:
                peft_config = LoraConfig(**lora_conf)
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]
        self.kv_cache_outdir = llm_conf.get("kv_cache_outdir", None)
        if self.kv_cache_outdir is not None:
            import os

            os.makedirs(self.kv_cache_outdir, exist_ok=True)
            os.makedirs(f"{self.kv_cache_outdir}/mat", exist_ok=True)
            os.makedirs(f"{self.kv_cache_outdir}/inputs_embeds", exist_ok=True)
            os.makedirs(f"{self.kv_cache_outdir}/txt", exist_ok=True)

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
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for name, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()

        self.audio_adaptor = audio_adaptor

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None
        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")
        self.fo = open(f"{self.kv_cache_outdir}/txt/{rank}.txt", "w")

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
        stats = {}
        input_ids[input_ids < 0] = 0
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        batch_size, token_num = input_ids.shape
        if speech is not None:
            if len(speech_lengths.size()) > 1:
                speech_lengths = speech_lengths[:, 0]

            batch_size_speech, frames, _ = speech.shape

            # with torch.cuda.amp.autocast(enabled=False):
            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
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

        input_mask_beg = kwargs.get("input_mask_beg")
        input_mask_beg[input_mask_beg < 0] = 0
        input_mask = kwargs.get("input_mask")
        input_mask[input_mask < 0] = 0

        hidden_states = model_outputs.hidden_states[-1].float()
        key = kwargs.get("key")[0]
        kv_cache_outdir = self.kv_cache_outdir
        mat_file = f"{kv_cache_outdir}/mat/{key}.mat"
        savemat(mat_file, {"kv_cache": hidden_states[0].cpu()})

        mat_file = f"{kv_cache_outdir}/inputs_embeds/{key}.mat"
        savemat(mat_file, {"inputs_embeds": inputs_embeds[0].float().cpu()})

        for turn_id_cum in range(input_mask.shape[0]):
            beg = input_mask_beg[turn_id_cum].sum(-1)
            end = input_mask[turn_id_cum].sum(-1)
            uttid = f"{key}_assistant_{turn_id_cum:02d}"
            line = f"{uttid} {mat_file} {beg} {end}\n"
            self.fo.write(line)
            self.fo.flush()

        with torch.no_grad():
            preds = torch.argmax(model_outputs.logits, -1)
            acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
            stats["acc"] = acc_att

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
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

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
            llm_kwargs = kwargs.get("llm_kwargs", {})
            if not kwargs.get("tearchforing", False):

                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_length", 512),
                    **llm_kwargs,
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
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    **llm_kwargs,
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


@tables.register("model_classes", "LLMASRXvecSlotTTS")
class LLMASRXvecSlotTTS(nn.Module):
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
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )
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

        if not llm_conf.get("low_cpu", False):
            model = AutoModelForCausalLM.from_pretrained(
                init_param_path,
                load_in_8bit=None,
                device_map=None,
                use_cache=None,
                output_hidden_states=True,
            )
        else:
            import os

            if int(os.environ.get("RANK", 0)) == 0:
                model = AutoModelForCausalLM.from_pretrained(
                    init_param_path,
                    load_in_8bit=None,
                    device_map="cpu",
                    use_cache=None,
                )
            else:
                llm_config = AutoConfig.from_pretrained(init_param_path)
                model = AutoModelForCausalLM.from_config(llm_config)

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        logging.info(f"use_lora: {llm_conf.get('use_lora', False)}")
        if llm_conf.get("use_lora", False):
            from omegaconf import OmegaConf, DictConfig

            lora_conf = llm_conf.get("lora_conf", {})
            if isinstance(lora_conf, (OmegaConf, DictConfig)):
                lora_conf = OmegaConf.to_container(lora_conf, resolve=True)
            from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel

            lora_init_param_path = lora_conf.get("init_param_path", None)
            if lora_init_param_path is not None:
                model = PeftModel.from_pretrained(model, lora_init_param_path)
            else:
                peft_config = LoraConfig(**lora_conf)
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

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
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for name, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()

        self.audio_adaptor = audio_adaptor

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

        # tts text tokenizer related
        audio_decoder_conf = kwargs.get("audio_decoder_conf", {})
        tts_token_type = audio_decoder_conf.get("tts_token_type", "whisper_rich_ttsfrd")
        ttsfrd_res_dir = audio_decoder_conf.get("ttsfrd_res_dir", "./ttsfrd/9.5.5")
        from funasr.models.llm_asr.tts_text_tokenizer.build_tokenizer import build_tokenizer

        self.tts_text_tokenizer = build_tokenizer(
            tts_token_type,
            bpemodel=ttsfrd_res_dir,
            p_word2phn=1.0,
        )
        # e2e tts model related
        from funasr.models.llm_asr.tts_models.e2e_model import UCTDXvecSlotModel

        self.tts_model = UCTDXvecSlotModel(**kwargs.get("tts_model_conf", {}))
        # vocoder related
        vocoder_name = kwargs.get("vocoder", None)
        vocoder_conf = kwargs.get("vocoder_conf", None)
        self.vocoder = self.build_vocoder(name=vocoder_name, conf=vocoder_conf)

        import os

        rank = int(os.environ.get("RANK", 0))
        logging.info(f"rank: {rank}, model is builded.")

    def build_vocoder(self, name: str, conf: dict):
        if name is None or conf is None:
            return None
        if name == "HifiGAN":
            from funasr.models.llm_asr.hifigan import HifiGan

            return HifiGan(**conf)
        return None

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

            # with torch.cuda.amp.autocast(enabled=False):
            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
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

        # tts model training related
        # audio sampling point
        audio = kwargs.get("audio")
        audio_len = kwargs.get("audio_len")

        # codec
        codec = kwargs.get("codec")
        codec_len = (codec > 0).sum(-1)

        input_mask = kwargs.get("input_mask")
        input_mask[input_mask < 0] = 0

        hidden_states = model_outputs.hidden_states[-1].float()
        hidden_states_his_select = []

        # target, str
        target_ids = []
        target_ids_len = []
        turn_id_cum = 0
        for batch_idx in range(labels_ids.shape[0]):

            for turn_id in range(fbank_beg.shape[1]):
                beg = 0
                end = input_mask[turn_id_cum].sum(-1)
                print(f"beg: {beg}, end: {end}")
                hidden_states_his_i = hidden_states[batch_idx, beg:end, :]
                hidden_states_his_select.append(hidden_states_his_i)

                turn_id_cum += 1

            beg_i = 0
            end_i = 0
            for token_idx in range(labels_ids.shape[1]):
                token_int = labels_ids[batch_idx, token_idx].item()
                if token_int == self.eos:
                    target_ids_i = labels_ids[batch_idx, beg_i:end_i]
                    target_ids_len_i = end_i - beg_i
                    target_ids_len.append(target_ids_len_i)
                    target = self.tokenizer.decode(target_ids_i)
                    target_ids.append(target)

                    end_i += 1
                    beg_i = end_i
                    continue

                end_i += 1
                if token_int <= 0:
                    beg_i += 1

        # hidden_states_his_select
        hidden_states_his_select = torch.nn.utils.rnn.pad_sequence(
            hidden_states_his_select, batch_first=True, padding_value=0.0
        )
        hidden_states_his_select = hidden_states_his_select.to(device=input_ids.device)
        hidden_states_his_select_len = input_mask.sum(-1)

        # nar tts model related
        device = hidden_states_his_select.device
        text = [self.tts_text_tokenizer.text2tokens(x) for x in target_ids]
        text_lengths = [len(x) for x in text]
        text = pad_list(text, pad_value=-1).long().to(device)
        text_lengths = torch.tensor(text_lengths).to(audio_len)
        # mute the "da" noise.
        # TODO: make sure the sample rate is 22050.
        audio[:, : int(0.02 * 22050)] = 0
        hidden_states_his_select = self.tts_dim_proj(hidden_states_his_select)
        tts_loss, tts_states, tts_weight = self.tts_model.forward(
            text=text,
            text_lengths=text_lengths,
            speech_token=codec,
            speech_token_lengths=codec_len,
            audio=audio,
            audio_lengths=audio_len,
            prompt=hidden_states_his_select,
            prompt_len=hidden_states_his_select_len,
        )
        loss = loss + tts_loss
        for key, value in tts_states.items():
            stats[f"tts_{key}"] = value

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

        (
            input_ids,
            labels,
            fbank,
            fbank_lens,
            fbank_mask,
            fbank_beg,
            fake_token_len,
            input_mask,
            input_mask_beg,
        ) = ([], [], [], [], [], [], [], [], [])
        input_source_ids = []
        for i, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
            if i >= kwargs.get("multiturn_num_max", 5):
                break
            if len(input_ids) > kwargs.get("max_token_length", 1500):

                break
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

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

            if i == 0:
                sys_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                sys_prompt_len = tokenizer.encode(sys_prompt)
                input_mask_i = (
                    [1] * len(sys_prompt_len) + [0] * len(source_ids) + [0] * len(target_ids)
                )
            else:
                input_mask_i = [1] * len(input_ids) + [0] * len(source_ids)
            input_mask_i = torch.tensor(input_mask_i, dtype=torch.int64)
            input_mask_beg.append(input_mask_i)

            input_mask_i = [1] * len(input_ids) + [1] * len(source_ids)
            input_mask_i = torch.tensor(input_mask_i, dtype=torch.int64)
            input_mask.append(input_mask_i)

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
        if len(input_mask) > 0:
            output["input_mask"] = input_mask
            output["input_mask_beg"] = input_mask_beg
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
        rand_seed = kwargs.get("rand_seed", 0)

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        with torch.cuda.amp.autocast(
            enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            label = contents["assistant"][-1]
            self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})
            if not kwargs.get("tearchforing", False):

                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_length", 512),
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **llm_kwargs,
                )

                # TODO: get llm_cur_kv_cache

                target_ids = generated_ids["sequences"]
                response = tokenizer.batch_decode(
                    target_ids, skip_special_tokens=kwargs.get("skip_special_tokens", True)
                )[0]

                loss = None
            else:

                labels_ids = batch["labels_ids"]
                labels_ids[labels_ids == -1] = -100
                attention_mask = batch.get("attention_mask", None)
                # attention_mask = attention_mask.to(dtype_map[llm_dtype])
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    **llm_kwargs,
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

        # tts related inference, require the kv cache of llm last layer for only the current inputs
        # TODO: select kv cache of the current turn inputs
        attention_mask = batch.get("attention_mask", None)
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            labels=None,
            **llm_kwargs,
        )
        hidden_states = model_outputs.hidden_states[-1].float()
        # hidden_states = generated_ids[
        #     "hidden_states"
        # ]  # hidden_states: (t1, t2, ..., tn, ..., tN), tn=(l1, l2, ..., ln, ..., lN), ln: shape: 1x1x3584

        # token_num = len(hidden_states)
        # hidden_states_select = torch.zeros((1, token_num, 3584), dtype=torch.float32).to(
        #     inputs_embeds.device
        # )
        #
        # for i in range(token_num):
        #     hidden_states_select[0, i, :] = hidden_states[i][-1][0, 0, :].to(torch.float32)

        llm_cur_kv_cache, llm_cur_kv_cache_len = None, None

        input_mask_beg = batch.get("input_mask_beg")[-1][None, :]
        input_mask_beg[input_mask_beg < 0] = 0
        input_mask = batch.get("input_mask")[-1][None, :]
        input_mask[input_mask < 0] = 0

        for turn_id_cum in range(input_mask.shape[0]):
            beg = input_mask_beg[turn_id_cum].sum(-1)
            end = input_mask[turn_id_cum].sum(-1)
            llm_cur_kv_cache = hidden_states[:, beg:end, :]
            llm_cur_kv_cache_len = torch.tensor(
                [
                    end - beg,
                ],
                dtype=torch.int32,
            ).to(inputs_embeds.device)
        # Generative quality is sensitive to dtype, FM requires fp32
        tts_dtype = "fp32"
        with torch.cuda.amp.autocast(
            enabled=True if tts_dtype != "fp32" else False, dtype=dtype_map[tts_dtype]
        ):
            assert llm_cur_kv_cache is not None
            set_all_random_seed(rand_seed)
            # speech_tokens, mel, wav = self.generate_speech(
            #     response, llm_cur_kv_cache, llm_cur_kv_cache_len, dtype_map[tts_dtype]
            # )
            speech_tokens, mel, wav, mp3 = self.simulate_streaming_generate_speech(
                target_ids, llm_cur_kv_cache, llm_cur_kv_cache_len, dtype_map[tts_dtype], tokenizer
            )
            self.write_mel_wav(kwargs.get("output_dir"), mel, wav, mp3, key[0])

        return results, meta_data

    def generate_speech(self, text, llm_cur_kv_cache, llm_cur_kv_cache_len, llm_dtype):
        # self.tts_text_tokenizer = self.tts_text_tokenizer
        self.vocoder.to(llm_dtype)
        device = llm_cur_kv_cache.device
        # tokenize text
        text_token = self.tts_text_tokenizer.text2tokens(f"<|endofprompt|><|sil|>{text}<|sil|>")
        text_token = torch.tensor([text_token], dtype=torch.long, device=device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.long, device=device)
        # e2e tts model forward
        self.tts_model.to(llm_dtype)
        speech_tokens, mel_feats = self.tts_model.inference(
            text_token,
            text_token_len,
            None,
            None,
            outside_prompt=llm_cur_kv_cache,
            outside_prompt_lengths=llm_cur_kv_cache_len,
            sampling="threshold_1e-6",
        )
        # vocoder forward
        wav = self.vocoder.inference(mel_feats.transpose(1, 2))

        return speech_tokens, mel_feats, wav

    def split_characters_and_words(self, input_string):
        # 定义正则表达式模式
        pattern = r"[\u4e00-\u9fff]|[\w]+|[^\w\s]"
        # 使用 re.findall 找到所有匹配的字符和单词
        results = re.findall(pattern, input_string)
        return results

    def tts_tokenizer_warpper(self, text):
        text_token = self.tts_text_tokenizer.text2tokens(text)
        # remove the added pouc by ttsfrd.
        if text[-1] != "。" and text_token[-1] == 1542:
            text_token = text_token[:-1]
        return text_token

    def generate_speech_one_step(
        self,
        text: str,
        last_t_size,
        llm_cur_kv_cache,
        llm_cur_kv_cache_len,
        prompt_token,
        prompt_audio,
        tts_text_chunk_size,
        chunk_idx,
        is_last,
        para_len=30,
    ):
        device = llm_cur_kv_cache.device
        pounc = ["。", "？", "！", "；", "：", ".", "?", "!", ";", "\n"]

        # remove duplicated pounctuations
        normed_text = []
        for i, c in enumerate(text):
            if i > 0 and text[i - 1] in pounc and text[i] in pounc:
                continue
            normed_text.append(c)
        text = "".join(normed_text)

        cur_token, feat, wav = None, None, None
        _text = f"<|endofprompt|><|sil|>{text}" + ("<|sil|>" if is_last else "")
        text_token = self.tts_tokenizer_warpper(_text)
        t_size = len(text_token)
        if (t_size - last_t_size) >= tts_text_chunk_size or is_last:
            text_token = torch.tensor([text_token], dtype=torch.long, device=device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.long, device=device)
            cur_token, feat = self.tts_model.streaming_one_step(
                text_token,
                text_token_len,
                xvec=None,
                xvec_lengths=None,
                prompt_dict={
                    "prompt_token": prompt_token,
                    "prompt_audio": prompt_audio,
                },
                outside_prompt=llm_cur_kv_cache,
                outside_prompt_lengths=llm_cur_kv_cache_len,
                sampling="threshold_1e-6",
                chunk_idx=chunk_idx,
            )
            if cur_token is not None and cur_token.shape[1] > 0 and feat.shape[2] > 0:
                # process first package, token in B,T,D, feat in B,F,T
                if prompt_token[0] is None:
                    prompt_token = [
                        cur_token,
                        torch.tensor([cur_token.shape[1]], dtype=torch.long, device=device),
                    ]
                    prompt_audio = [
                        feat.transpose(1, 2),
                        torch.tensor([feat.shape[2]], dtype=torch.long, device=device),
                    ]
                else:
                    prompt_token[1] = prompt_token[1] + cur_token.shape[1]
                    prompt_token[0] = torch.concat([prompt_token[0], cur_token], dim=1)
                    prompt_audio[1] = prompt_audio[1] + feat.shape[2]
                    prompt_audio[0] = torch.concat([prompt_audio[0], feat.transpose(1, 2)], dim=1)
                wav = self.vocoder.inference(feat.transpose(1, 2))
                chunk_idx += 1
            else:
                cur_token, feat, wav = None, None, None

            # post process
            last_t_size = t_size
            # restart a new paragraph
            # char_words = self.split_characters_and_words(text)
            # if len(char_words) > para_len:
            #     # find the last pounc to split paragraph
            #     idx = -1
            #     for i in range(len(char_words)-1, -1, -1):
            #         if char_words[i] in pounc:
            #             idx = i
            #             break
            #     if idx > 0:
            #         text = text[idx+1:]
            #         last_t_size = len(self.tts_tokenizer_warpper(text))

        return ((cur_token, feat, wav), (text, last_t_size, prompt_token, prompt_audio, chunk_idx))

    def convert_wav_to_mp3(self, wav: torch.Tensor):
        wav = wav.detach().cpu().numpy()
        wav = (wav * (2**16-1)).astype(np.int16)
        mp3 = AudioSegment.from_raw(
            wav.tobytes(),
            sample_width=16 // 8,  # Sample width in bytes
            frame_rate=22050,
            channels=1
        )
        mp3_buffer = BytesIO()
        mp3.export(mp3_buffer, format="mp3", bitrate="48k")
        # we should return this to web page.
        mp3_bytes_data = mp3_buffer.getvalue()

        return mp3_bytes_data

    def simulate_streaming_generate_speech(
        self, preds, llm_cur_kv_cache, llm_cur_kv_cache_len, llm_dtype, llm_tokenizer
    ):
        # self.tts_text_tokenizer = self.tts_text_tokenizer
        self.vocoder.to(llm_dtype)
        self.tts_model.to(llm_dtype)
        llm_token_num_per_call = 3
        text_chunk_size = 8
        given_rtf = 0.5

        token_list, feat_list, wav_list, mp3_list = [], [], [], []
        prompt_token, prompt_audio = [None, None], [None, None]
        new_text, last_t_size, chunk_idx = "", 0, 0
        st, count = 0, 0
        while st < preds.shape[1]:
            chunk_size = int(llm_token_num_per_call / (given_rtf ** min(count, 2)))
            _resp = llm_tokenizer.batch_decode(
                preds[:, st : st + chunk_size],
                add_special_tokens=False,
                skip_special_tokens=True,
            )[0]
            is_last = st + chunk_size >= preds.shape[1]

            new_text = new_text + _resp
            rt_value, states = self.generate_speech_one_step(
                new_text,
                last_t_size,
                llm_cur_kv_cache,
                llm_cur_kv_cache_len,
                prompt_token,
                prompt_audio,
                text_chunk_size,
                chunk_idx,
                is_last,
            )
            cur_token, feat, wav = rt_value
            new_text, last_t_size, prompt_token, prompt_audio, chunk_idx = states
            # save results
            if cur_token is not None and feat is not None and wav is not None:
                token_list.append(cur_token)
                feat_list.append(feat)
                # we should return this data to web page for playing.
                mp3_data = self.convert_wav_to_mp3(wav)
                wav_list.append(wav)
                mp3_list.append(mp3_data)

            st += chunk_size
            count += 1

        speech_tokens = torch.cat(token_list, dim=1)
        mel_feats = torch.cat(feat_list, dim=2)
        wav = torch.cat(wav_list, dim=1)
        mp3 = b''.join(mp3_list)
        return speech_tokens, mel_feats, wav, mp3

    def write_mel_wav(self, output_dir, feat, wav, mp3, key):
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
                path,
                wav.cpu(),
                sample_rate=self.vocoder.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )
        if mp3 is not None:
            path = os.path.join(out_dir, f"{key}.mp3")
            fd = open(path, "wb")
            fd.write(mp3)
            fd.close()


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
        rand_seed = kwargs.get("rand_seed", 0)

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

            # set random seed for reproduce
            set_all_random_seed(rand_seed)
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

            # set random seed for reproduce
            set_all_random_seed(rand_seed)
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
            "wav": wav,
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
                path,
                wav.cpu(),
                sample_rate=self.vocoder.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
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
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            diff_steps=10,
            temperature=1.0,
            prompt=dict(prompt_text=(None, None), prompt_audio=(None, None)),
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
            [1, prompt.shape[1] + max_length, prompt.shape[2]], dtype=torch.float32, device=device
        )
        seq_input[:, : prompt.shape[1], :] = prompt
        out_tokens = torch.zeros([1, max_length, 1], dtype=torch.int64, device=device)
        out_token_len = 0
        prompt_len = prompt.shape[1]
        state, hit_eos = None, False
        for i in range(max_length):
            # use state for speedup
            pred, (state, _) = self.audio_decoder.score(
                seq_input[0, : prompt_len + out_token_len], state, prompt[0]
            )

            # sampling all `nq` token ids
            pred = pred.reshape(self.predict_nq, -1)
            # normalize scores
            pred = torch.log_softmax(pred, dim=-1)
            if min_length is not None and i < min_length:
                pred[:, self.codebook_size + self.ad_sos_eos] = float(np.finfo(np.float32).min)
            top_ids = self.ras_sampling(pred[0], out_tokens[0, :out_token_len, 0])

            if torch.any(top_ids == (self.codebook_size + self.ad_sos_eos)):
                hit_eos = True
                out_tokens = out_tokens[:, :out_token_len, :]
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


@tables.register("model_classes", "LLMASR6")
class LLMASR6(nn.Module):
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
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )

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
        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(init_param_path)

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

        # tts text tokenizer related
        tts_token_type = audio_decoder_conf.get("tts_token_type", "whisper_rich_ttsfrd")
        ttsfrd_res_dir = audio_decoder_conf.get("ttsfrd_res_dir", "./ttsfrd/9.5.5")
        from funasr.models.llm_asr.tts_text_tokenizer.build_tokenizer import build_tokenizer

        self.tts_text_tokenizer = build_tokenizer(
            tts_token_type,
            bpemodel=ttsfrd_res_dir,
            p_word2phn=1.0,
        )
        from funasr.models.llm_asr.tts_models.e2e_model import UCTDXvecSlotModel

        from omegaconf import OmegaConf, DictConfig

        tts_model_conf = kwargs.get("tts_model_conf", {})
        if isinstance(tts_model_conf, DictConfig):
            tts_model_conf = OmegaConf.to_container(tts_model_conf, resolve=True)
        self.tts_model = UCTDXvecSlotModel(**tts_model_conf)
        self.tts_dim_proj = nn.Linear(llm_dim, self.tts_model.output_size)

        # self.codebook_dim = audio_decoder_conf.get("codebook_dim", 1024)
        # self.codebook_size = audio_decoder_conf.get("codebook_size", 4096)
        # self.lm_out_voc_size = self.codebook_size + 1
        # self.audio_decoder = self.build_audio_decoder(name=audio_decoder, conf=audio_decoder_conf)
        # self.concat_emb_hidden = audio_decoder_conf.get("concat_emb_hidden", False)
        # self.concat_emb_hidden_norm = audio_decoder_conf.get("concat_emb_hidden_norm", False)
        # if self.concat_emb_hidden_norm:
        #     self.hidden_norm = LayerNorm(llm_dim)
        #     self.fusion_dropout = nn.Dropout(audio_decoder_conf.get("fusion_drop_rate", 0.0))
        #     self.emb_norm = LayerNorm(llm_dim)
        #     self.fusion_norm = LayerNorm(self.audio_decoder.embed_unit)
        #     self.fusion_act = Swish()
        # audio_decoder_in_proj_dim = llm_dim * 2 if self.concat_emb_hidden else llm_dim
        # self.audio_decoder_in_proj = torch.nn.Linear(
        #     audio_decoder_in_proj_dim, self.audio_decoder.embed_unit
        # )
        # self.codec_embedder = torch.nn.Embedding(self.codebook_size, self.codebook_dim)
        # self.audio_decoder_embedding = torch.nn.Embedding(2, self.audio_decoder.embed_unit)
        # self.ad_sos_eos = 0
        # self.ad_task_id = 1
        # self.ad_ignore_id = -1
        # self.predict_nq = 1
        #
        # from .label_smoothing_loss import LabelSmoothingLoss
        #
        # self.criterion_ce = LabelSmoothingLoss(
        #     size=self.lm_out_voc_size // self.predict_nq,
        #     padding_idx=self.ad_ignore_id,
        #     smoothing=lsm_weight,
        #     normalize_length=length_normalized_loss,
        #     reduction=False,
        # )
        #
        # mel_decoder_name = kwargs.get("mel_decoder", None)
        # mel_decoder_conf = kwargs.get("mel_decoder_conf", None)
        # self.mel_decoder = self.build_mel_decoder(name=mel_decoder_name, conf=mel_decoder_conf)
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

            # with torch.cuda.amp.autocast(enabled=False):
            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
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

        # audio sampling point
        audio = kwargs.get("audio")
        audio_len = kwargs.get("audio_len")

        # codec
        codec = kwargs.get("codec")
        codec_len = (codec > 0).sum(-1)

        input_mask = kwargs.get("input_mask")
        input_mask[input_mask < 0] = 0

        hidden_states = model_outputs.hidden_states[-1].float()
        hidden_states_his_select = []

        # target, str
        target_ids = []
        target_ids_len = []
        turn_id_cum = 0
        for batch_idx in range(labels_ids.shape[0]):

            try:
                for turn_id in range(fbank_beg.shape[1]):

                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    if fbank_beg_idx > 0:
                        beg = 0
                        end = input_mask[turn_id_cum].sum(-1)
                        # print(f"beg: {beg}, end: {end}")
                        hidden_states_his_i = hidden_states[batch_idx, beg:end, :]
                        hidden_states_his_select.append(hidden_states_his_i)
                        turn_id_cum += 1
            except:
                import pdb

                pdb.set_trace()
            beg_i = 0
            end_i = 0
            for token_idx in range(labels_ids.shape[1]):
                token_int = labels_ids[batch_idx, token_idx].item()
                if token_int == self.eos:
                    target_ids_i = labels_ids[batch_idx, beg_i:end_i]
                    target_ids_len_i = end_i - beg_i
                    target_ids_len.append(target_ids_len_i)
                    target = self.tokenizer.decode(target_ids_i)
                    target_ids.append(target)

                    end_i += 1
                    beg_i = end_i
                    continue

                end_i += 1
                if token_int <= 0:
                    beg_i += 1

        # hidden_states_his_select
        hidden_states_his_select = torch.nn.utils.rnn.pad_sequence(
            hidden_states_his_select, batch_first=True, padding_value=0.0
        )
        hidden_states_his_select = hidden_states_his_select.to(device=input_ids.device)
        hidden_states_his_select_len = input_mask.sum(-1)

        # import pdb
        #
        # pdb.set_trace()

        # if self.concat_emb_hidden:
        #     if not self.concat_emb_hidden_norm:
        #         hidden_states_select = torch.concat((hidden_states_select, target_emb), dim=-1)
        #         hidden_states_select = self.audio_decoder_in_proj(hidden_states_select)
        #     else:
        #         outs = self.hidden_norm(hidden_states_select)
        #         outs = self.fusion_dropout(self.fusion_act(outs))
        #         # emb = model_outputs.hidden_states[0]
        #         emb = self.fusion_dropout(self.fusion_act(self.emb_norm(target_emb)))
        #         outs = self.audio_decoder_in_proj(torch.cat([outs, emb], dim=-1))
        #         hidden_states_select = self.fusion_act(self.fusion_norm(outs))
        #
        # nll, logits, target, target_lengths = self.nll(
        #     hidden_states_select, target_ids_len, codec[:, :, None], codec_len
        # )
        # output_mask = (
        #     ~make_pad_mask(target_lengths, maxlen=target_lengths.max())
        #     .to(hidden_states_select.device)
        #     .unsqueeze(-1)
        # )
        # total, batch_size = output_mask.sum() * self.predict_nq, nll.shape[0] * self.predict_nq
        # denom = total if self.length_normalized_loss else batch_size
        # loss = (nll * output_mask).sum() / denom
        #
        # with torch.no_grad():
        #     preds = torch.argmax(model_outputs.logits, -1)
        #     acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
        #     stats["acc"] = acc_att
        #
        #     cc = logits.shape[-1]
        #     for i in range(self.predict_nq):
        #         acc = th_accuracy(
        #             logits[:, :, i, :].reshape(-1, cc), target[:, :, i], self.ad_ignore_id
        #         )
        #         stats[f"codec_acc_{i + 1}"] = acc

        # nar tts model related
        # import pdb; pdb.set_trace()
        device = hidden_states_his_select.device
        text = [
            torch.tensor(self.tts_text_tokenizer.text2tokens(x), dtype=torch.int64).to(device)
            for x in target_ids
        ]
        text_lengths = [len(x) for x in text]
        text = pad_list(text, pad_value=-1).long().to(device)
        audio_len = torch.tensor(audio_len, dtype=torch.int64).to(device)
        text_lengths = torch.tensor(text_lengths, dtype=torch.int64).to(device)
        # mute the "da" noise.
        # TODO: make sure the sample rate is 22050.
        audio[:, : int(0.02 * 22050)] = 0
        hidden_states_his_select = self.tts_dim_proj(hidden_states_his_select)
        tts_loss, tts_states, tts_weight = self.tts_model.forward(
            text=text,
            text_lengths=text_lengths,
            speech_token=codec,
            speech_token_lengths=codec_len,
            audio=audio,
            audio_lengths=audio_len,
            prompt=hidden_states_his_select,
            prompt_len=hidden_states_his_select_len,
        )
        loss = loss + tts_loss
        for key, value in tts_states.items():
            stats[f"tts_{key}"] = value

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
        rand_seed = kwargs.get("rand_seed", 0)

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

            # set random seed for reproduce
            set_all_random_seed(rand_seed)
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

            # set random seed for reproduce
            set_all_random_seed(rand_seed)
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
            "wav": wav,
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
                path,
                wav.cpu(),
                sample_rate=self.vocoder.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
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
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            diff_steps=10,
            temperature=1.0,
            prompt=dict(prompt_text=(None, None), prompt_audio=(None, None)),
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
            [1, prompt.shape[1] + max_length, prompt.shape[2]], dtype=torch.float32, device=device
        )
        seq_input[:, : prompt.shape[1], :] = prompt
        out_tokens = torch.zeros([1, max_length, 1], dtype=torch.int64, device=device)
        out_token_len = 0
        prompt_len = prompt.shape[1]
        state, hit_eos = None, False
        for i in range(max_length):
            # use state for speedup
            pred, (state, _) = self.audio_decoder.score(
                seq_input[0, : prompt_len + out_token_len], state, prompt[0]
            )

            # sampling all `nq` token ids
            pred = pred.reshape(self.predict_nq, -1)
            # normalize scores
            pred = torch.log_softmax(pred, dim=-1)
            if min_length is not None and i < min_length:
                pred[:, self.codebook_size + self.ad_sos_eos] = float(np.finfo(np.float32).min)
            top_ids = self.ras_sampling(pred[0], out_tokens[0, :out_token_len, 0])

            if torch.any(top_ids == (self.codebook_size + self.ad_sos_eos)):
                hit_eos = True
                out_tokens = out_tokens[:, :out_token_len, :]
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


@tables.register("model_classes", "LLMASR7")
class LLMASR7(nn.Module):
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
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )

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
        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()
        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(init_param_path)

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

        # tts text tokenizer related
        tts_token_type = audio_decoder_conf.get("tts_token_type", "whisper_rich_ttsfrd")
        ttsfrd_res_dir = audio_decoder_conf.get("ttsfrd_res_dir", "./ttsfrd/9.5.5")
        from funasr.models.llm_asr.tts_text_tokenizer.build_tokenizer import build_tokenizer

        self.tts_text_tokenizer = build_tokenizer(
            tts_token_type,
            bpemodel=ttsfrd_res_dir,
            p_word2phn=1.0,
        )
        from funasr.models.llm_asr.tts_models.e2e_model import UCTDXvecSlotModel

        from omegaconf import OmegaConf, DictConfig

        tts_model_conf = kwargs.get("tts_model_conf", {})
        if isinstance(tts_model_conf, DictConfig):
            tts_model_conf = OmegaConf.to_container(tts_model_conf, resolve=True)
        self.tts_model = UCTDXvecSlotModel(**tts_model_conf)
        self.tts_dim_proj = nn.Linear(llm_dim, self.tts_model.output_size)

        # self.codebook_dim = audio_decoder_conf.get("codebook_dim", 1024)
        # self.codebook_size = audio_decoder_conf.get("codebook_size", 4096)
        # self.lm_out_voc_size = self.codebook_size + 1
        # self.audio_decoder = self.build_audio_decoder(name=audio_decoder, conf=audio_decoder_conf)
        # self.concat_emb_hidden = audio_decoder_conf.get("concat_emb_hidden", False)
        # self.concat_emb_hidden_norm = audio_decoder_conf.get("concat_emb_hidden_norm", False)
        # if self.concat_emb_hidden_norm:
        #     self.hidden_norm = LayerNorm(llm_dim)
        #     self.fusion_dropout = nn.Dropout(audio_decoder_conf.get("fusion_drop_rate", 0.0))
        #     self.emb_norm = LayerNorm(llm_dim)
        #     self.fusion_norm = LayerNorm(self.audio_decoder.embed_unit)
        #     self.fusion_act = Swish()
        # audio_decoder_in_proj_dim = llm_dim * 2 if self.concat_emb_hidden else llm_dim
        # self.audio_decoder_in_proj = torch.nn.Linear(
        #     audio_decoder_in_proj_dim, self.audio_decoder.embed_unit
        # )
        # self.codec_embedder = torch.nn.Embedding(self.codebook_size, self.codebook_dim)
        # self.audio_decoder_embedding = torch.nn.Embedding(2, self.audio_decoder.embed_unit)
        # self.ad_sos_eos = 0
        # self.ad_task_id = 1
        # self.ad_ignore_id = -1
        # self.predict_nq = 1
        #
        # from .label_smoothing_loss import LabelSmoothingLoss
        #
        # self.criterion_ce = LabelSmoothingLoss(
        #     size=self.lm_out_voc_size // self.predict_nq,
        #     padding_idx=self.ad_ignore_id,
        #     smoothing=lsm_weight,
        #     normalize_length=length_normalized_loss,
        #     reduction=False,
        # )
        #
        # mel_decoder_name = kwargs.get("mel_decoder", None)
        # mel_decoder_conf = kwargs.get("mel_decoder_conf", None)
        # self.mel_decoder = self.build_mel_decoder(name=mel_decoder_name, conf=mel_decoder_conf)
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

            # with torch.cuda.amp.autocast(enabled=False):
            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
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

        # audio sampling point
        audio = kwargs.get("audio")
        audio_len = kwargs.get("audio_len")
        audio_len = audio_len[-1:]
        audio = audio[-1:, : audio_len[0]]

        # codec
        codec = kwargs.get("codec")
        codec_len = (codec > 0).sum(-1)
        codec_len = codec_len[-1:]
        codec = codec[-1:, : codec_len[0]]

        input_mask = kwargs.get("input_mask")
        input_mask[input_mask < 0] = 0

        hidden_states = model_outputs.hidden_states[-1].float()
        hidden_states_his_select = []

        # target, str
        # target_ids = []
        # target_ids_len = []
        turn_id_cum = 0
        for batch_idx in range(labels_ids.shape[0]):

            for turn_id in range(fbank_beg.shape[1]):

                fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                if fbank_beg_idx > 0:
                    beg = 0
                    end = input_mask[turn_id_cum].sum(-1)
                    # print(f"beg: {beg}, end: {end}")
                    hidden_states_his_i = hidden_states[batch_idx, beg:end, :]
                    hidden_states_his_select.append(hidden_states_his_i)
                    turn_id_cum += 1

        target = kwargs.get("turn_targets")
        target_ids = target[-1:]

        # hidden_states_his_select
        hidden_states_his_select = torch.nn.utils.rnn.pad_sequence(
            hidden_states_his_select[-1:], batch_first=True, padding_value=0.0
        )
        hidden_states_his_select = hidden_states_his_select.to(device=input_ids.device)
        hidden_states_his_select_len = torch.tensor(
            [hidden_states_his_select.shape[1]], dtype=torch.int64
        ).to(hidden_states_his_select.device)

        device = hidden_states_his_select.device
        text = [
            torch.tensor(self.tts_text_tokenizer.text2tokens(x), dtype=torch.int64).to(device)
            for x in target_ids
        ]
        text_lengths = [len(x) for x in text]
        text = pad_list(text, pad_value=-1).long().to(device)
        audio_len = torch.tensor(audio_len, dtype=torch.int64).to(device)
        text_lengths = torch.tensor(text_lengths, dtype=torch.int64).to(device)
        # mute the "da" noise.
        # TODO: make sure the sample rate is 22050.
        audio[:, : int(0.02 * 22050)] = 0
        hidden_states_his_select = self.tts_dim_proj(hidden_states_his_select)
        tts_loss, tts_states, tts_weight = self.tts_model.forward(
            text=text,
            text_lengths=text_lengths,
            speech_token=codec,
            speech_token_lengths=codec_len,
            audio=audio,
            audio_lengths=audio_len,
            prompt=hidden_states_his_select,
            prompt_len=hidden_states_his_select_len,
        )
        loss = tts_loss  # loss + tts_loss
        for key, value in tts_states.items():
            stats[f"tts_{key}"] = value

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
        rand_seed = kwargs.get("rand_seed", 0)

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

            # set random seed for reproduce
            set_all_random_seed(rand_seed)
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

            # set random seed for reproduce
            set_all_random_seed(rand_seed)
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
            "wav": wav,
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
                path,
                wav.cpu(),
                sample_rate=self.vocoder.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
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
            tokens,
            token_lens,
            xvec,
            xvec_lens,
            diff_steps=10,
            temperature=1.0,
            prompt=dict(prompt_text=(None, None), prompt_audio=(None, None)),
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
            [1, prompt.shape[1] + max_length, prompt.shape[2]], dtype=torch.float32, device=device
        )
        seq_input[:, : prompt.shape[1], :] = prompt
        out_tokens = torch.zeros([1, max_length, 1], dtype=torch.int64, device=device)
        out_token_len = 0
        prompt_len = prompt.shape[1]
        state, hit_eos = None, False
        for i in range(max_length):
            # use state for speedup
            pred, (state, _) = self.audio_decoder.score(
                seq_input[0, : prompt_len + out_token_len], state, prompt[0]
            )

            # sampling all `nq` token ids
            pred = pred.reshape(self.predict_nq, -1)
            # normalize scores
            pred = torch.log_softmax(pred, dim=-1)
            if min_length is not None and i < min_length:
                pred[:, self.codebook_size + self.ad_sos_eos] = float(np.finfo(np.float32).min)
            top_ids = self.ras_sampling(pred[0], out_tokens[0, :out_token_len, 0])

            if torch.any(top_ids == (self.codebook_size + self.ad_sos_eos)):
                hit_eos = True
                out_tokens = out_tokens[:, :out_token_len, :]
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


@tables.register("model_classes", "LLMVAD")
class LLMVAD(nn.Module):
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
        length_normalized_loss: bool = False,
        **kwargs,
    ):

        super().__init__()

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get(
            "activation_checkpoint", False
        )
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
        logging.info(f"Loading llm ckpt: {init_param_path}")
        model = AutoModelForCausalLM.from_pretrained(
            init_param_path,
            load_in_8bit=None,
            device_map=None,
            use_cache=None,
        )
        logging.info(f"llm ckpt loaded: {init_param_path}")

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        logging.info(f"use_lora: {llm_conf.get('use_lora', False)}")
        if llm_conf.get("use_lora", False):
            from omegaconf import OmegaConf, DictConfig

            lora_conf = llm_conf.get("lora_conf", {})
            if isinstance(lora_conf, (OmegaConf, DictConfig)):
                lora_conf = OmegaConf.to_container(lora_conf, resolve=True)
            from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel

            lora_init_param_path = lora_conf.get("init_param_path", None)
            if lora_init_param_path is not None:
                model = PeftModel.from_pretrained(model, lora_init_param_path)
            else:
                peft_config = LoraConfig(**lora_conf)
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

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
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for name, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()

        self.audio_adaptor = audio_adaptor

        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

        self.loss_fct = CrossEntropyLoss()

        print("self.llm.config:", self.llm.config)
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        from copy import deepcopy

        self.task_decoder_layer_config = deepcopy(self.llm.config)
        self.task_decoder_layer_config.hidden_size = self.llm.config.hidden_size // 4
        self.task_decoder_layer_config.intermediate_size = self.llm.config.intermediate_size // 4
        self.task_decoder_layer_config.num_attention_heads = (
            self.llm.config.num_attention_heads // 4
        )
        self.task_decoder_layer_config.num_key_value_heads = (
            self.llm.config.num_key_value_heads // 4
        )
        print("self.task_decoder_layer_config:", self.task_decoder_layer_config)
        self.down_proj = nn.Linear(
            self.llm.config.hidden_size, self.task_decoder_layer_config.hidden_size, bias=False
        ).to(dtype_map[self.llm_dtype])
        self.task_decoder_layer = Qwen2DecoderLayer(
            self.task_decoder_layer_config, self.llm.config.num_hidden_layers
        ).to(dtype_map[self.llm_dtype])
        if getattr(self.llm.config, "classifier_dropout", None) is not None:
            classifier_dropout = self.llm.config.classifier_dropout
        elif getattr(self.llm.config, "hidden_dropout", None) is not None:
            classifier_dropout = self.llm.config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.barge_in_num_labels = 2
        self.turn_taking_num_labels = 2
        self.barge_in_score = nn.Linear(
            self.task_decoder_layer_config.hidden_size, self.barge_in_num_labels
        ).to(dtype_map[self.llm_dtype])
        self.turn_taking_score = nn.Linear(
            self.task_decoder_layer_config.hidden_size, self.turn_taking_num_labels
        ).to(dtype_map[self.llm_dtype])

    def forward(
        self,
        speech: torch.Tensor = None,
        speech_lengths: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels_ids: torch.Tensor = None,
        fbank_beg: torch.Tensor = None,
        fbank_mask: torch.Tensor = None,
        turn_taking_labels: torch.Tensor = None,
        barge_in_labels: torch.Tensor = None,
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

            # with torch.cuda.amp.autocast(enabled=False):
            # audio encoder
            if self.audio_encoder_activation_checkpoint:
                from torch.utils.checkpoint import checkpoint

                encoder_out, encoder_out_lens = checkpoint(
                    self.encode, speech, speech_lengths, use_reentrant=False
                )
            else:
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
                output_hidden_states=True,
            )
            output_attentions = kwargs.get("output_attentions", None)
            past_key_values = kwargs.get("past_key_values", None)
            past_key_values_length = kwargs.get("past_key_values_length", 0)
            position_ids = kwargs.get("position_ids", None)
            use_cache = kwargs.get("use_cache", None)
            seq_length = token_num
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()
            from transformers.modeling_attn_mask_utils import (
                _prepare_4d_causal_attention_mask,
                _prepare_4d_causal_attention_mask_for_sdpa,
            )

            if self.llm.config._attn_implementation == "flash_attention_2":
                # 2d mask is passed through the layers
                causal_attention_mask = (
                    attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                )
            elif self.llm.config._attn_implementation == "sdpa" and not output_attentions:
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                causal_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.llm.config.sliding_window,
                )
            else:
                # 4d mask is passed through the layers
                causal_attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.llm.config.sliding_window,
                )

            sequence_output = model_outputs.hidden_states[-1]
            sequence_output = self.down_proj(sequence_output)
            if self.llm.model.gradient_checkpointing and self.llm.model.training:
                layer_outputs = self.llm._gradient_checkpointing_func(
                    self.task_decoder_layer.__call__,
                    sequence_output,
                    causal_attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = self.task_decoder_layer(
                    sequence_output,
                    attention_mask=causal_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            sequence_output = layer_outputs[0]

            sequence_output = self.dropout(sequence_output)
            turn_taking_logits = self.turn_taking_score(sequence_output)
            barge_in_logits = self.barge_in_score(sequence_output)

        loss = None
        if barge_in_labels is not None:
            barge_in_labels[barge_in_labels == -1] = -100
            barge_in_loss = self.loss_fct(
                barge_in_logits.view(-1, self.barge_in_num_labels), barge_in_labels.view(-1)
            )
            loss = barge_in_loss
        if turn_taking_labels is not None:
            turn_taking_labels[turn_taking_labels == -1] = -100
            turn_taking_loss = self.loss_fct(
                turn_taking_logits.view(-1, self.turn_taking_num_labels),
                turn_taking_labels.view(-1),
            )
            loss = turn_taking_loss if loss is None else loss + turn_taking_loss

        stats = {}
        # with torch.no_grad():
        #     preds = torch.argmax(model_outputs.logits, -1)
        #     acc_att = compute_accuracy(preds[:, :-1], labels_ids[:, 1:], ignore_label=-100)
        #     stats["acc"] = acc_att
        if turn_taking_labels is not None:
            stats["turn_taking_loss"] = torch.clone(turn_taking_loss.detach())
            with torch.no_grad():
                turn_taking_preds = torch.argmax(turn_taking_logits, -1)
                turn_taking_acc = compute_accuracy(
                    turn_taking_preds, turn_taking_labels, ignore_label=-100
                )
                stats["turn_taking_acc"] = turn_taking_acc
        if barge_in_labels is not None:
            stats["barge_in_loss"] = torch.clone(barge_in_loss.detach())
            with torch.no_grad():
                barge_in_preds = torch.argmax(barge_in_logits, -1)
                barge_in_acc = compute_accuracy(barge_in_preds, barge_in_labels, ignore_label=-100)
                stats["barge_in_acc"] = barge_in_acc
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

    def vad_inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        inputs_embeds, contents, batch, source_ids, meta_data = self.vad_inference_prepare(
            data_in, data_lengths, key, tokenizer, frontend, **kwargs
        )
        task = contents.get("task", "vad")
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]
        fbank_mask = batch["fbank_mask"]
        batch_size, token_num, dims = inputs_embeds.shape
        fake_token_len[fake_token_len < 0] = 0
        fbank_beg[fbank_beg < 0] = 0

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        stats = {
            "turn_taking_preds": [],
            "barge_in_preds": [],
            "turn_taking_labels": [],
            "barge_in_labels": [],
            "task": task,
        }
        with torch.cuda.amp.autocast(
            enabled=True if llm_dtype != "fp32" else False, dtype=dtype_map[llm_dtype]
        ):
            self.llm = self.llm.to(dtype_map[llm_dtype])
            self.down_proj = self.down_proj.to(dtype_map[llm_dtype])
            self.task_decoder_layer = self.task_decoder_layer.to(dtype_map[llm_dtype])
            self.turn_taking_score = self.turn_taking_score.to(dtype_map[llm_dtype])
            self.barge_in_score = self.barge_in_score.to(dtype_map[llm_dtype])

            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})

            attention_mask = batch.get("attention_mask", None)
            # attention_mask = attention_mask.to(dtype_map[llm_dtype])
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=None,
                output_hidden_states=True,
                **llm_kwargs,
            )
            output_attentions = llm_kwargs.get("output_attentions", None)
            past_key_values = llm_kwargs.get("past_key_values", None)
            past_key_values_length = llm_kwargs.get("past_key_values_length", 0)
            position_ids = llm_kwargs.get("position_ids", None)
            use_cache = llm_kwargs.get("use_cache", None)
            seq_length = token_num
            if position_ids is None:
                device = inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()

            from transformers.modeling_attn_mask_utils import (
                _prepare_4d_causal_attention_mask,
                _prepare_4d_causal_attention_mask_for_sdpa,
            )

            if self.llm.config._attn_implementation == "flash_attention_2":
                # 2d mask is passed through the layers
                attention_mask = (
                    attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                )
            elif self.llm.config._attn_implementation == "sdpa" and not output_attentions:
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.llm.config.sliding_window,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                    sliding_window=self.llm.config.sliding_window,
                )

            sequence_output = model_outputs.hidden_states[-1]
            sequence_output = self.down_proj(sequence_output)

            layer_outputs = self.task_decoder_layer(
                sequence_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            sequence_output = layer_outputs[0]

            sequence_output = self.dropout(sequence_output)
            turn_taking_logits = self.turn_taking_score(sequence_output)
            barge_in_logits = self.barge_in_score(sequence_output)

            turn_taking_labels = batch.get("turn_taking_labels", None)
            barge_in_labels = batch.get("barge_in_labels", None)
            # print(f'batch: {batch}')
            # print(f"fake_token_len: {fake_token_len}")
            # print(f"turn taking labels: {turn_taking_labels}")
            # print(f"barge in labels: {barge_in_labels}")
            turn_taking_preds_res = []
            barge_in_preds_res = []
            turn_taking_labels_res = []
            barge_in_labels_res = []
            with torch.no_grad():
                turn_taking_preds = torch.argmax(turn_taking_logits, -1)
                barge_in_preds = torch.argmax(barge_in_logits, -1)
                for batch_idx in range(batch_size):
                    fbank_begin_index = fbank_beg[batch_idx, -1].item()
                    fbank_end_index = fbank_begin_index + fake_token_len[batch_idx, -1].item()
                    turn_taking_preds_last = (
                        turn_taking_preds[batch_idx, fbank_begin_index:fbank_end_index]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    turn_taking_preds_res.append(turn_taking_preds_last)
                    # print(f"turn_taking_labels: {turn_taking_labels}")
                    turn_taking_labels_last = (
                        turn_taking_labels[batch_idx, fbank_begin_index:fbank_end_index]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    turn_taking_labels_res.append(turn_taking_labels_last)
                    # print(f"turn_taking_preds: {turn_taking_preds_last}")
                    barge_in_preds_last = (
                        barge_in_preds[batch_idx, fbank_begin_index:fbank_end_index]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    barge_in_preds_res.append(barge_in_preds_last)
                    # print(f"barge_in_labels: {barge_in_labels}")
                    barge_in_labels_last = (
                        barge_in_labels[batch_idx, fbank_begin_index:fbank_end_index]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    barge_in_labels_res.append(barge_in_labels_last)

                turn_taking_acc = compute_accuracy(
                    turn_taking_preds, turn_taking_labels, ignore_label=-100
                )
                stats["turn_taking_acc"] = turn_taking_acc.item()

                barge_in_acc = compute_accuracy(barge_in_preds, barge_in_labels, ignore_label=-100)
                stats["barge_in_acc"] = barge_in_acc.item()
            stats["turn_taking_preds"].append(turn_taking_preds_res)
            stats["barge_in_preds"].append(barge_in_preds_res)
            stats["turn_taking_labels"].append(turn_taking_labels_res)
            stats["barge_in_labels"].append(barge_in_labels_res)
        return turn_taking_logits, barge_in_logits, meta_data, stats

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech.permute(0, 2, 1), speech_lengths)

        return encoder_out, encoder_out_lens

    def vad_data_template(self, sample):
        data = sample["messages"]
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
        assistant.append("")
        contents = {
            "system": system,
            "user": user,
            "assistant": assistant,
        }

        if "task" in sample:
            task = sample["task"]
            last_total_time = data[-1]["end_time"] - data[-1]["start_time"]
            if task == "turn-taking":
                true_time_span = data[-1]["turn-taking-gap_time-added"]
            elif task == "barge-in":
                true_time_span = last_total_time - data[-1]["barge-in-0"]
            else:
                raise ValueError("task must be turn-taking or barge-in")
            contents["true_time_span"] = true_time_span
            contents["last_total_time"] = last_total_time
            contents["task"] = sample["task"]
        return contents

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

    def vad_data_load_speech(self, contents: dict, tokenizer, frontend, meta_data={}, **kwargs):

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
            if isinstance(user_prompt, (list, tuple)):
                user_prompt, audio = user_prompt
            if i == 0:
                source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
            elif i == len(system) - 1:
                source_input = f"<|im_start|>user\n{user_prompt}"
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
            # target_out = f"{target_out}<|im_end|>"
            # target_ids = tokenizer.encode(target_out)
            target_ids = []
            input_source_ids = input_ids + source_ids
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            fbank_mask += fbank_mask_i
            if len(speech) > 0:
                fbank.append(speech[0, :, :])
                fbank_lens.append(speech_lengths)

        turn_taking_labels = [-100] * len(labels)
        barge_in_labels = [-100] * len(labels)
        last_vad = [0] * fake_token_len[-1]
        if "true_time_span" in contents:
            true_time_span = contents["true_time_span"]
            last_time_span = contents["last_total_time"]
            pos_vad = math.ceil(fake_token_len[-1] * (true_time_span / last_time_span))
            assert pos_vad <= fake_token_len[-1]
            if pos_vad > 0:
                last_vad[-pos_vad:] = [1] * pos_vad
        turn_taking_labels[-fake_token_len[-1] :] = last_vad
        barge_in_labels[-fake_token_len[-1] :] = last_vad

        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
        attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
        labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]
        turn_taking_labels = torch.tensor(
            [turn_taking_labels], dtype=torch.int64
        )  # [: self.max_token_length]
        barge_in_labels = torch.tensor(
            [barge_in_labels], dtype=torch.int64
        )  # [: self.max_token_length]

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
            "turn_taking_labels": turn_taking_labels,
            "barge_in_labels": barge_in_labels,
        }

        return output

    def vad_inference_prepare(
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

        contents = self.vad_data_template(data_in[0])
        output = self.vad_data_load_speech(
            contents, tokenizer, frontend, meta_data=meta_data, **kwargs
        )
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
            llm_kwargs = kwargs.get("llm_kwargs", {})
            if not kwargs.get("tearchforing", False):

                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=kwargs.get("max_length", 512),
                    **llm_kwargs,
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
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    **llm_kwargs,
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
