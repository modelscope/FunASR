import logging
import os
import random
import re
import string
import time
import traceback
from typing import Union

import torch
import torch.nn as nn

from funasr.metrics.compute_acc import compute_accuracy
from funasr.register import tables
from funasr.train_utils.device_funcs import force_gatherable, to_device
from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video
from transformers import AutoConfig, AutoModelForCausalLM

from ctc import CTC
from tools.utils import forced_align

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@tables.register("model_classes", "FunASRNano")
class FunASRNano(nn.Module):
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
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)

        if freeze:
            for _, param in audio_encoder.named_parameters():
                param.requires_grad = False
            audio_encoder.eval()
        self.audio_encoder = audio_encoder

        # llm
        self.llm = None
        init_param_path = llm_conf.get("init_param_path", None)
        llm_dim = None

        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        config = AutoConfig.from_pretrained(init_param_path)
        model = AutoModelForCausalLM.from_config(config, **llm_load_kwargs)

        freeze = llm_conf.get("freeze", True)
        if freeze:
            for _, param in model.named_parameters():
                param.requires_grad = False
            model.eval()
        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()

        self.llm_dtype = llm_conf.get("llm_dtype", "fp32")
        self.llm = model.to(dtype_map[self.llm_dtype])
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        if audio_encoder_output_size > 0:
            audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = (
            llm_dim if llm_dim is not None else audio_adaptor_conf["llm_dim"]
        )
        audio_adaptor = adaptor_class(**audio_adaptor_conf)
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for _, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()
        self.audio_adaptor = audio_adaptor
        self.use_low_frame_rate = audio_adaptor_conf.get("use_low_frame_rate", False)

        # ctc decoder
        self.ctc_decoder = None
        # TODO: fix table name
        ctc_decoder_class = tables.adaptor_classes.get(kwargs.get("ctc_decoder", None))
        if ctc_decoder_class is not None:
            ctc_tokenizer = (
                kwargs.get("ctc_tokenizer", None)
                if "ctc_tokenizer" in kwargs
                else kwargs["dataset_conf"]["ctc_tokenizer"]
            )
            ctc_tokenizer_conf = (
                kwargs.get("ctc_tokenizer_conf", None)
                if "ctc_tokenizer_conf" in kwargs
                else kwargs["dataset_conf"]["ctc_tokenizer_conf"]
            )
            if ctc_tokenizer is not None and ctc_tokenizer_conf is not None:
                ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer)
                ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)
                self.ctc_tokenizer = ctc_tokenizer
            assert ctc_tokenizer is not None, f"ctc_tokenizer must be set"

            ctc_vocab_size = kwargs.get("ctc_vocab_size", 60515)
            ctc_decoder_conf = kwargs.get("ctc_decoder_conf", {})
            if audio_encoder_output_size > 0:
                ctc_decoder_conf["encoder_dim"] = audio_encoder_output_size
            self.ctc_decoder = ctc_decoder_class(**ctc_decoder_conf)
            init_param_path = ctc_decoder_conf.get("init_param_path", None)
            if init_param_path is not None:
                src_state = torch.load(init_param_path, map_location="cpu")
                flag = self.ctc_decoder.load_state_dict(src_state, strict=False)
                logging.info(f"Loading ctc_decoder ckpt: {init_param_path}, status: {flag}")
            freeze = ctc_decoder_conf.get("freeze", False)
            if freeze:
                for _, param in self.ctc_decoder.named_parameters():
                    param.requires_grad = False
                self.ctc_decoder.eval()

            ctc_conf = kwargs.get("ctc_conf", {})
            self.blank_id = ctc_conf.get("blank_id", ctc_vocab_size - 1)
            self.ctc_weight = kwargs.get("ctc_weight", 0.3)
            self.ctc = CTC(
                odim=ctc_vocab_size,
                encoder_output_size=audio_encoder_output_size,
                blank_id=self.blank_id,
                **ctc_conf,
            )
            self.detach_ctc_decoder = kwargs.get("detach_ctc_decoder", True)
            self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss
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
                                batch_idx,
                                fbank_beg_idx : fbank_beg_idx + speech_token_len,
                                :,
                            ] = speech_token
                        except Exception as e:
                            logging.error(f"{str(e)}, {traceback.format_exc()}")
                            logging.info(
                                f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, encoder_out: {encoder_out.shape}, encoder_out_lens: {encoder_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                            )
                            speech_token_len = encoder_out_lens[speech_idx].item()
                            speech_token = encoder_out[speech_idx, :speech_token_len, :]
                            inputs_embeds[
                                batch_idx,
                                fbank_beg_idx : fbank_beg_idx + speech_token_len,
                                :,
                            ] = speech_token

                        speech_idx += 1

            stats["batch_size_speech"] = batch_size_speech
            stats["batch_size_x_frames"] = frames * batch_size_speech
            stats["batch_size_real_frames"] = speech_lengths.sum().item()
            stats["padding_frames"] = stats["batch_size_x_frames"] - stats["batch_size_real_frames"]

        device_type = next(self.parameters()).device.type
        with torch.autocast(
            device_type=device_type if device_type in ["cuda", "xpu", "mps"] else "cpu",
            enabled=True if self.llm_dtype != "fp32" else False,
            dtype=dtype_map[self.llm_dtype],
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

    def forward_export(self, speech, speech_lengths, **kwargs):
        x, olens = self.audio_encoder(speech, speech_lengths)
        encoder_out, encoder_out_lens = self.audio_adaptor(x, olens)
        return encoder_out, encoder_out_lens

    def encode(self, speech, speech_lengths):
        # audio encoder
        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)

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
        do_think = True
        sys_prompt = True
        if "dataset_conf" in kwargs:
            do_think = kwargs["dataset_conf"].get("do_think", True)
            sys_prompt = kwargs["dataset_conf"].get("sys_prompt", True)

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
                    if not sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not sys_prompt:
                        source_input = (
                            f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                        )
            else:
                if kwargs.get("infer_with_assistant_input", False):
                    source_input = f"<|im_start|>user\n{user_prompt}"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
            if not do_think:
                source_input += "<think>\n\n</think>\n\n"
            if kwargs.get("prev_text", None) is not None:
                source_input += kwargs["prev_text"]

            splits = pattern.split(source_input)
            source_ids = []
            fbank_mask_i = []
            fake_token_len_i = 0
            fbank_beg_i = -1
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

                        if self.use_low_frame_rate:
                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                        else:
                            fake_token_len_i = speech_lengths[0].item()
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
                adaptor_out, adaptor_out_lens = self.audio_adaptor(encoder_out, encoder_out_lens)
                meta_data["encoder_out"] = encoder_out
                meta_data["encoder_out_lens"] = encoder_out_lens
                meta_data["audio_adaptor_out"] = adaptor_out
                meta_data["audio_adaptor_out_lens"] = adaptor_out_lens

        input_ids = batch["input_ids"]
        source_ids = batch["source_ids"]
        fbank_beg = batch["fbank_beg"]
        fake_token_len = batch["fake_token_len"]

        if not kwargs.get("teacherforcing", False):
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
                    speech_token = adaptor_out[speech_idx, :speech_token_len, :]

                    try:
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token
                    except Exception as e:
                        #
                        logging.error(f"{str(e)}, {traceback.format_exc()}")
                        logging.info(
                            f"batch_idx: {batch_idx}, inputs_embeds: {inputs_embeds.shape}, fbank_beg_idx: {fbank_beg_idx}, speech_token_len: {speech_token_len}, adaptor_out: {adaptor_out.shape}, adaptor_out_lens: {adaptor_out_lens}, fake_token_len: {fake_token_len}, speech_lengths: {speech_lengths}"
                        )
                        speech_token_len = adaptor_out_lens[speech_idx].item()
                        speech_token = adaptor_out[speech_idx, :speech_token_len, :]
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token

                    speech_idx += 1
        return inputs_embeds, contents, batch, source_ids, meta_data

    def get_prompt(self, hotwords: list[str], language: str = None, itn: bool = True):
        if len(hotwords) > 0:
            hotwords = ", ".join(hotwords)
            prompt = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{hotwords}]\n"
        else:
            prompt = ""
        if language is None:
            prompt += "语音转写"
        else:
            prompt += f"语音转写成{language}"
        if not itn:
            prompt += "，不进行文本规整"
        return prompt + "："

    def generate_chatml(self, prompt: str, data: Union[str, torch.Tensor]):
        if isinstance(data, str):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}<|startofspeech|>!{data}<|endofspeech|>"},
                {"role": "assistant", "content": "null"},
            ]
        elif isinstance(data, torch.Tensor):
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"{prompt}<|startofspeech|>!!<|endofspeech|>",
                    "audio": data,
                },
                {"role": "assistant", "content": "null"},
            ]

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        prompt = self.get_prompt(
            kwargs.get("hotwords", []), kwargs.get("language", None), kwargs.get("itn", True)
        )
        data_in = [self.generate_chatml(prompt, data) for data in data_in]

        if key is None:
            key = []
            for _ in data_in:
                chars = string.ascii_letters + string.digits
                key.append("rand_key_" + "".join(random.choice(chars) for _ in range(13)))

        return self.inference_llm(
            data_in,
            data_lengths=data_lengths,
            key=key,
            tokenizer=tokenizer,
            frontend=frontend,
            **kwargs,
        )

    def inference_llm(
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

        ctc_results = []
        if self.ctc_decoder is not None:
            encoder_out = meta_data["encoder_out"]
            encoder_out_lens = meta_data["encoder_out_lens"]
            decoder_out, decoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)
            ctc_logits = self.ctc.log_softmax(decoder_out)

            b, n, d = encoder_out.size()
            if isinstance(key[0], (list, tuple)):
                key = key[0]
            if len(key) < b:
                key = key * b
            for i in range(b):
                x = ctc_logits[i, : encoder_out_lens[i].item(), :]
                yseq = x.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)
                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                # Change integer-ids to tokens
                text = self.ctc_tokenizer.decode(token_int)
                ctc_results.append({"key": key[i], "text": text, "ctc_logits": x})

        llm_dtype = kwargs.get("llm_dtype", "fp32")
        if llm_dtype == "fp32":
            llm_dtype = "fp16" if kwargs.get("fp16", False) else llm_dtype
            llm_dtype = "bf16" if kwargs.get("bf16", False) else llm_dtype

        device_type = torch.device(kwargs.get("device", "cuda")).type
        with torch.autocast(
            device_type=device_type if device_type in ["cuda", "xpu", "mps"] else "cpu",
            enabled=True if llm_dtype != "fp32" else False,
            dtype=dtype_map[llm_dtype],
        ):
            label = contents["assistant"][-1]
            self.llm = self.llm.to(dtype_map[llm_dtype])
            inputs_embeds = inputs_embeds.to(dtype_map[llm_dtype])
            llm_kwargs = kwargs.get("llm_kwargs", {})
            if not kwargs.get("teacherforcing", False):
                attention_mask = batch.get("attention_mask", None)
                generated_ids = self.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=kwargs.get("max_length", 512),
                    pad_token_id=self.llm.config.pad_token_id or self.llm.config.eos_token_id,
                    **llm_kwargs,
                )

                response = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]

                loss = None
            else:
                labels_ids = batch["labels_ids"]
                labels_ids[labels_ids == -1] = -100
                attention_mask = batch.get("attention_mask", None)
                model_outputs = self.llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels_ids,
                    pad_token_id=self.llm.config.pad_token_id or self.llm.config.eos_token_id,
                    **llm_kwargs,
                )

                preds = torch.argmax(model_outputs.logits, -1)[:, source_ids.shape[1] :]
                response = tokenizer.batch_decode(
                    preds,
                    add_special_tokens=False,
                    skip_special_tokens=kwargs.get("skip_special_tokens", True),
                )[0]
                loss = model_outputs.loss.item()
        response = kwargs.get("prev_text", "") + response

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{0 + 1}best_recog"]

        results = []
        response_clean = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        result_i = {
            "key": key[0],
            "text": re.sub(r"\s+", " ", response.replace("/sil", " ")),
            "text_tn": response_clean,
            "label": label,
        }
        if loss is not None:
            result_i["loss"] = loss
        results.append(result_i)

        for ctc_result, result in zip(ctc_results, results):
            result["ctc_text"] = ctc_result["text"].replace("<|nospeech|>", "")
            target_ids = torch.tensor(
                self.ctc_tokenizer.encode(result["ctc_text"]), dtype=torch.int64
            )
            result["ctc_timestamps"] = forced_align(
                ctc_result["ctc_logits"], target_ids, self.blank_id
            )
            target_ids = torch.tensor(self.ctc_tokenizer.encode(result["text"]), dtype=torch.int64)
            result["timestamps"] = forced_align(ctc_result["ctc_logits"], target_ids, self.blank_id)
            for timestamps in [result["timestamps"], result["ctc_timestamps"]]:
                for timestamp in timestamps:
                    timestamp["token"] = self.ctc_tokenizer.decode([timestamp["token"]])
                    timestamp["start_time"] = timestamp["start_time"] * 6 * 10 / 1000
                    timestamp["end_time"] = timestamp["end_time"] * 6 * 10 / 1000

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = response.replace("\n", " ")
            ibest_writer["label"][key[0]] = label.replace("\n", " ")
            ibest_writer["text_tn"][key[0]] = response_clean

        return results, meta_data

    @staticmethod
    def from_pretrained(model: str = None, **kwargs):
        from funasr import AutoModel

        model, kwargs = AutoModel.build_model(model=model, trust_remote_code=True, **kwargs)

        return model, kwargs
