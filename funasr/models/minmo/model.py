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
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


try:
    from scipy.io import savemat
except:
    pass

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


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


@tables.register("model_classes", "LLMASR4")
@tables.register("model_classes", "MinMo_S2T")
class MinMo_S2T(nn.Module):
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
            audio_encoder_output_size = (
                model.model.encoder_output_size
                if hasattr(model.model, "encoder_output_size")
                else -1
            )

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
        if audio_encoder_output_size > 0:
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
                            data_src = load_audio_text_image_video(
                                sub_str, fs=frontend.fs, **kwargs
                            )
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
            if "audio_embedding" in kwargs and "audio_embedding_lens" in kwargs:
                encoder_out = kwargs["audio_embedding"]
                encoder_out_lens = kwargs["audio_embedding_lens"]
            else:
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

                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

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


@tables.register("model_classes", "MinMo_T2S")
class MinMo_T2S(nn.Module):
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

        # # adaptor
        # adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        # if audio_encoder_output_size > 0:
        #     audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        # audio_adaptor_conf["llm_dim"] = llm_dim
        # audio_adaptor = adaptor_class(**audio_adaptor_conf)
        # init_param_path = audio_adaptor_conf.get("init_param_path", None)
        # if init_param_path is not None:
        #     src_state = torch.load(init_param_path, map_location="cpu")
        #     flag = audio_adaptor.load_state_dict(src_state, strict=False)
        #     logging.info(f"Loading audio_adaptor ckpt: {init_param_path}, status: {flag}")
        # freeze = audio_adaptor_conf.get("freeze", False)
        # if freeze:
        #     for name, param in audio_adaptor.named_parameters():
        #         param.requires_grad = False
        #     audio_adaptor.eval()
        #
        # self.audio_adaptor = audio_adaptor
        del self.llm.lm_head
        self.codec_unit = kwargs.get("codec_unit", 4097)
        self.codec_embed = nn.Embedding(self.codec_unit, llm_dim, 0)
        self.codec_head = nn.Linear(llm_dim, self.codec_unit, bias=False)
        self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
        self.beam_search = None

        self.ignore_id = kwargs.get("ignore_id", -100)
        # self.criterion_ce = LabelSmoothingLoss(
        #     size=self.codec_unit,
        #     padding_idx=self.ignore_id,
        #     smoothing=kwargs.get("lsm_weight", 0.0),
        #     normalize_length=self.length_normalized_loss,
        #     # reduction=False,
        # )
        # self.criterion_ce = CrossEntropyLoss(ignore_index=-100)

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

        codec = kwargs.get("codec")
        codec[codec < 0] = 0
        codec_len = kwargs.get("codec_len")
        codec_len[codec_len < 0] = 0
        codec_beg = kwargs.get("codec_beg")
        codec_beg[codec_beg < 0] = 0
        fake_codec_len = kwargs.get("fake_codec_len")
        fake_codec_len[fake_codec_len < 0] = 0

        audio = kwargs.get("audio")
        audio_len = kwargs.get("audio_len")
        spk_emb = kwargs.get("spk_emb")
        spk_emb_len = kwargs.get("spk_emb_len")

        batch_size, token_num, dims = inputs_embeds.shape

        codec_emb = self.codec_embed(codec)
        idx = 0
        for i, batch_idx in enumerate(range(batch_size)):

            for j, turn_id in enumerate(range(codec_beg.shape[1])):
                codec_beg_idx = codec_beg[batch_idx, turn_id].item()
                if codec_beg_idx > 0:
                    codec_len_i = codec_len[batch_idx, turn_id]
                    codec_i = codec_emb[idx, :codec_len_i, :]

                    inputs_embeds[batch_idx, codec_beg_idx : codec_beg_idx + codec_len_i, :] = (
                        codec_i
                    )
                    idx += 1

        with torch.cuda.amp.autocast(
            enabled=True if self.llm_dtype != "fp32" else False, dtype=dtype_map[self.llm_dtype]
        ):
            labels_ids[labels_ids < 0] = self.ignore_id
            attention_mask[attention_mask < 0] = 0
            outputs = self.llm.model(
                inputs_embeds=inputs_embeds.to(dtype_map[self.llm_dtype]),
                attention_mask=attention_mask,
            )
            labels = labels_ids
            # loss = model_outputs.loss
            hidden_states = outputs[0]
            logits = self.codec_head(hidden_states)
            logits = logits.float()

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # loss_att = self.criterion_ce(shift_logits, shift_labels)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_id)
            shift_logits = shift_logits.view(-1, self.codec_unit)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        with torch.no_grad():
            preds = torch.argmax(logits, -1)
            acc_att = compute_accuracy(
                preds[:, :-1], labels_ids[:, 1:], ignore_label=self.ignore_id
            )
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
            if "audio_embedding" in kwargs and "audio_embedding_lens" in kwargs:
                encoder_out = kwargs["audio_embedding"]
                encoder_out_lens = kwargs["audio_embedding_lens"]
            else:
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

                meta_data["audio_adaptor_out"] = encoder_out
                meta_data["audio_adaptor_out_lens"] = encoder_out_lens

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
