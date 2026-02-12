import json
import logging

import re
import torch
import random
import traceback
import numpy as np
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "FunASR")
class FunASR(torch.utils.data.Dataset):
    """
    FunASR dataset
    """

    def __init__(
            self,
            path,
            index_ds: str = None,
            frontend=None,
            tokenizer=None,
            int_pad_value: int = -1,
            float_pad_value: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf")
            )
        self.preprocessor_speech = preprocessor_speech

        preprocessor_noise = kwargs.get("preprocessor_noise", None)
        if preprocessor_noise:
            preprocessor_noise_class = tables.preprocessor_classes.get(preprocessor_noise)
            preprocessor_noise = preprocessor_noise_class(**kwargs.get("preprocessor_noise_conf"))
        self.preprocessor_noise = preprocessor_noise

        prompt_classes_text = kwargs.get("prompt_classes", None)
        if prompt_classes_text is not None:
            prompt_classes = tables.prompt_classes.get(prompt_classes_text)
            prompt_classes = prompt_classes(**kwargs.get("prompt_conf"))
        else:
            prompt_classes = None
        self.prompt_classes = prompt_classes

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.int_pad_value = int_pad_value
        self.float_pad_value = float_pad_value
        self.sos = kwargs.get("sos", "<|startoftranscript|>")
        self.eos = kwargs.get("eos", "<|endoftext|>")
        self.batch_size = kwargs.get("batch_size")
        self.batch_type = kwargs.get("batch_type")
        self.prompt_ids_len = 0
        self.retry = kwargs.get("retry", 100)

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)
        self.max_target_length = kwargs.get("max_target_length", 1024)
        self.do_think = kwargs.get("do_think", True)
        self.sys_prompt = kwargs.get("sys_prompt", True)

        # used for dynamic output alignment
        self.use_dynamic_output_ratio = kwargs.get("use_dynamic_output_ratio", 0.0)
        self.min_output_mask_token_len = kwargs.get("min_mask_token_len", 1)
        self.min_output_non_mask_token_len = kwargs.get("min_non_mask_token_len", 6)  # [eos]

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def get_random_user_prompt(self, item, user_prompt):
        tasks = ["语音转写：", "Speech transcription:"]
        language = item.get("language", None)
        # LID in distill data is fake
        language = None
        if language is not None:
            if language.lower() == "zh":
                tasks.append("语音转写成中文：")
                tasks.append("Transcribe speech into Chinese:")
            elif language.lower() == "en":
                tasks.append("语音转写成英文：")
                tasks.append("Transcribe speech into English:")
        if len(tasks) == 2:
            task = random.choice(tasks)
        elif len(tasks) == 4:
            task = random.choices(tasks, weights=[0.4, 0.4, 0.1, 0.1])[0]
        if "语音转写：<|startofspeech|>" in user_prompt:
            user_prompt = user_prompt.replace("语音转写：<|startofspeech|>", task + "<|startofspeech|>")
        elif "Speech transcription:<|startofspeech|>" in user_prompt:
            user_prompt = user_prompt.replace("Speech transcription:<|startofspeech|>", task + "<|startofspeech|>")
        return user_prompt

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        output = None

        for idx in range(self.retry):
            if idx > 0:
                logging.info(f"retry: {idx}")
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]
            is_noised = item.get("noised", False)
            if len(user) < 1 or len(assistant) < 1:
                logging.warning(f"item is error: {item}")
                continue
            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg, fake_token_len = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                    zip(system, user, assistant)
            ):
                if i >= self.multiturn_num_max:
                    break
                if len(input_ids) > self.max_token_length:
                    logging.info(
                        f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                    )
                    break

                if self.prompt_classes is not None:
                    asr_prompt = user_prompt.split("<|startofspeech|>")[0]
                    language = self.prompt_classes.detect_language(asr_prompt)
                    user_prompt_all_context = self.prompt_classes.get_prompt(item, language)
                else:
                    user_prompt_all_context = ""

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt_all_context}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    if not self.sys_prompt:
                        source_input = f"<|im_start|>user\n{user_prompt_all_context}{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )
                if not self.do_think:
                    source_input += "<think>\n\n</think>\n\n"
                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                speech = []
                speech_lengths = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                                if self.preprocessor_noise is not None and not is_noised:
                                    try:
                                        data_src = self.preprocessor_noise(data_src.numpy())
                                    except Exception as e:
                                        logging.error(f"Generate noise audio failed: {e}")

                                speech, speech_lengths = extract_fbank(
                                    data_src,
                                    data_type=self.data_type,
                                    frontend=self.frontend,
                                    is_final=True,
                                )  # speech: [b, T, d]
                                if speech_lengths > self.max_source_length:
                                    logging.info(
                                        f"speech_lengths > max_source_length: {speech_lengths}>{self.max_source_length}, {item}"
                                    )
                                    badcase_flag = True
                            except Exception as e:
                                logging.warning(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}\n{item}"
                                )
                                badcase_flag = True
                                continue
                            if True:
                                olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                                olens = 1 + (olens - 3 + 2 * 1) // 2
                                fake_token_len_i = (olens - 1) // 2 + 1
                            else:
                                fake_token_len_i = speech_lengths[0].item()
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token

                if badcase_flag:
                    continue
                if fbank_beg_i > 0:
                    fbank_beg += [fbank_beg_i + len(input_ids)]
                    fake_token_len += [fake_token_len_i]
                else:
                    fbank_beg += [-1]
                    fake_token_len += [0]

                if target_out is not None and any(
                        isinstance(item, dict) and "prev_content" in item for item in target_out
                ):
                    prev_value = next(
                        (
                            item["prev_content"]
                            for item in target_out
                            if isinstance(item, dict) and "prev_content" in item
                        ),
                        None,
                    )
                    source_ids += self.tokenizer.encode(prev_value)
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out[0]}<|im_end|>"
                else:
                    source_mask = [-100] * len(source_ids)
                    target_out = f"{target_out}<|im_end|>"
                target_ids = self.tokenizer.encode(target_out)

                if len(target_ids) > self.max_target_length:
                    logging.info(
                        f"text_length: {len(target_ids)} > {self.max_target_length}, drop it: {item}"
                    )
                #  simulate prev-token fixed output
                target_labels = target_ids.copy()
                if np.random.rand() < self.use_dynamic_output_ratio:
                    max_len = len(target_labels)
                    min_output_mask_token_len = min(self.min_output_mask_token_len, max_len)
                    min_output_non_mask_token_len = min(self.min_output_non_mask_token_len, max_len)
                    if max_len - min_output_non_mask_token_len > min_output_mask_token_len:
                        end_index = np.random.randint(min_output_mask_token_len,
                                                      max_len - min_output_non_mask_token_len)
                    else:
                        end_index = max_len - min_output_non_mask_token_len
                    if end_index > 0:
                        target_labels[:end_index] = [-100] * end_index

                input_ids += source_ids + target_ids
                labels += source_mask + target_labels
                if len(speech) > 0:
                    fbank.append(speech[0, :, :])
                    fbank_lens.append(speech_lengths)
            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
            output["item"] = item
            if len(fbank) > 0:
                output["speech"] = fbank
                output["speech_lengths"] = fbank_lens
            if len(input_ids) > self.max_token_length:
                logging.warning(
                    f"len(input_ids): {len(input_ids)} > max_token_length: {self.max_token_length}, item: {item}"
                )
                continue

            break

        return output

    def collator(self, samples: list = None):

        for idx in range(self.retry):
            badcase_flag = False

            outputs = {}
            for sample in samples:
                if sample is None:
                    continue
                for key in sample.keys():
                    if key not in outputs:
                        outputs[key] = []
                    if isinstance(sample[key], (list, tuple)):
                        outputs[key].extend(sample[key])
                    else:
                        outputs[key].append(sample[key])

            for key, data_list in outputs.items():
                if isinstance(data_list[0], torch.Tensor):
                    if data_list[0].dtype == torch.int64 or data_list[0].dtype == torch.int32:

                        pad_value = self.int_pad_value
                    else:
                        pad_value = self.float_pad_value

                    outputs[key] = torch.nn.utils.rnn.pad_sequence(
                        data_list, batch_first=True, padding_value=pad_value
                    )

            if self.batch_type != "example":
                b, t = outputs["input_ids"].shape
                if b > 1 and b * t > self.batch_size_token_max:
                    logging.info(
                        f"Warning, {idx}th, b*t: {b}*{t}={b * t} > batch_size_sample_max: {self.batch_size_token_max}, drop last data"
                    )
                    samples = samples[:-1]
                    continue

            break

        return outputs


@tables.register("index_ds_classes", "FunASR")
class FunASR(torch.utils.data.Dataset):  # torch.utils.data.Dataset

    def __init__(self, path: str, **kwargs):
        super().__init__()

        self.max_source_length = kwargs.get("max_source_length", 8000)
        self.min_source_length = kwargs.get("min_source_length", 10)
        self.max_target_length = kwargs.get("max_target_length", 2048)
        self.min_target_length = kwargs.get("min_target_length", 0)
        # self.max_token_length = kwargs.get("max_token_length", 2200)+
        audio_downsample_rate = int(kwargs.get("audio_downsample_rate", 8))

        is_training = kwargs.get("is_training", True)
        if not (path.endswith(".jsonl") or path.endswith(".json")):
            # jsonl list file
            data_split_num = kwargs.get("data_split_num", 1)
            data_split_i = kwargs.get("data_split_i", 0)

            if not is_training:
                data_split_num = 1
                data_split_i = 0
            with open(path, encoding="utf-8") as fin:
                file_list_all = fin.readlines()

                num_per_slice = (len(file_list_all) - 1) // data_split_num + 1  # 16
                file_list = file_list_all[
                            data_split_i * num_per_slice: (data_split_i + 1) * num_per_slice
                            ]
                logging.info(
                    f"is_training: {is_training}, data_split_num: {data_split_num}, data_split_i: {data_split_i}, \nfile_list: {file_list}, \nfile_list_all: {file_list_all}"
                )

        else:
            file_list = [path]

        contents = []
        total_whrs = 0.0
        total_token_for_llm_B = 0.0
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    try:
                        data_dict = json.loads(line.strip())
                    except Exception as e:
                        logging.error(
                            f"drop it, json error: {e}, line: {line}, file_json: {file_json}"
                        )
                        continue

                    data = data_dict["messages"]
                    if isinstance(data_dict.get("speech_length", 0), (list, tuple)):
                        speech_length = int(data_dict.get("speech_length", [0])[0])
                        text_length = int(data_dict.get("text_length", 0)[0])
                    else:
                        speech_length = int(data_dict.get("speech_length", 0))
                        text_length = int(data_dict.get("text_length", 0))
                    speech_length = int(speech_length)
                    text_length = int(text_length)
                    if speech_length > 0 and speech_length < 1:
                        continue
                    if text_length < 1:
                        logging.warning(
                            f"speech_length: {speech_length}, text_length: {text_length}, data: {data}, file_json: {file_json}"
                        )
                        if len(data) > 2:
                            text_length = len(data[2]['content'])
                        continue
                    if speech_length > self.max_source_length:
                        continue
                    if speech_length < self.min_source_length:
                        continue
                    if text_length > self.max_target_length:
                        continue

                    system, user, assistant = [], [], []
                    for i, item in enumerate(data):
                        try:
                            role = item["role"]
                            content = item["content"]
                        except KeyError:
                            logging.error(
                                f"drop it, KeyError: {item}, file_json: {file_json}"
                            )
                            continue

                        if role == "system":
                            system.append(content)
                        elif role == "user":
                            user.append(content)
                        elif role == "assistant":
                            if "prev_content" in item:
                                prev_content = item["prev_content"]
                                assistant.append([content, {"prev_content": prev_content}])
                            else:
                                assistant.append(content)
                    if len(system) == 0:
                        system = ["You are a helpful assistant."]
                    system = system * len(user)

                    contents_i = {
                        "system": system,
                        "user": user,
                        "assistant": assistant,
                        "source_len": speech_length + text_length,
                    }
                    if "key" in data_dict:
                        contents_i["key"] = data_dict["key"] if not isinstance(data_dict.get("key", "key_01234"),
                                                                               (list, tuple)) else data_dict["key"][0]

                    if "hist_context" in data_dict:
                        contents_i["hist_context"] = data_dict["hist_context"]
                    if "hotwords" in data_dict:
                        contents_i["hotwords"] = data_dict["hotwords"]
                    if "asr_hotwords" in data_dict:
                        contents_i["asr_hotwords"] = data_dict["asr_hotwords"]
                    if "vad_segs" in data_dict:
                        contents_i["vad_segs"] = data_dict["vad_segs"]
                    if "word_list" in data_dict:
                        contents_i["word_list"] = data_dict["word_list"]
                    if "one_pass_result" in data_dict:
                        contents_i["one_pass_result"] = data_dict["one_pass_result"]
                    if "one_pass_wer" in data_dict:
                        contents_i["one_pass_wer"] = data_dict["one_pass_wer"]
                    if "noised" in data_dict:
                        contents_i["noised"] = data_dict["noised"]

                    if kwargs.get("save_meta", False):
                        contents_i["meta"] = data_dict

                    total_whrs += speech_length / 100.0 / 3600 / 10000 * audio_downsample_rate
                    total_token_for_llm_B += (text_length + speech_length / 8) / 1000 / 1000 / 1000
                    contents.append(contents_i)

        self.contents = contents

        logging.info(
            f"\n\ntotal_num of samplers: {len(self.contents)}, total_whrs: {total_whrs:.5f}, total_token_for_llm_B: {total_token_for_llm_B:.5g}, {path}, {file_list}\n\n")

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):

        data = self.contents[index]

        return data

    def get_source_len(self, data_dict):
        source_len = data_dict.get("source_len", -1)
        if source_len < 0:
            source_len = len(data_dict["system"]) + len(data_dict["user"])
        return source_len

    def get_target_len(self, data_dict):

        return 0
