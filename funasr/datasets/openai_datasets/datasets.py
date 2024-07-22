import logging
import re
import torch
import random
import traceback
from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "OpenAIDataset")
class OpenAIDataset(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
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
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

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

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1024)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.audio_adaptor_downsample_rate = kwargs.get("audio_adaptor_downsample_rate", 2)
        self.audio_encoder_downsample_rate = kwargs.get("audio_encoder_downsample_rate", 4)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb;
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]

            input_ids, labels, fbank, fbank_lens, fbank_mask, fbank_beg = [], [], [], [], [], []

            for i, (system_prompt, user_prompt, target_out) in enumerate(
                zip(system, user, assistant)
            ):

                source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_mask_i = []
                fbank_beg_i = []
                fbank_lens_i = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.error(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue
                            if self.audio_encoder_downsample_rate == 4:
                                olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                                olens = 1 + (olens - 3 + 2 * 1) // 2
                            elif self.audio_encoder_downsample_rate == 1:
                                olens = speech_lengths[0].item()

                            sub_token_len = (olens - 1) // self.audio_adaptor_downsample_rate + 1
                            sub_token = [0] * sub_token_len
                            fbank_beg_i = [len(source_ids)]
                            source_ids += sub_token
                            fbank_mask_i += [1] * len(sub_token)

                if badcase_flag:
                    continue
                source_mask = [-100] * len(source_ids)
                target_out = f"{target_out}<|im_end|>"
                target_ids = self.tokenizer.encode(target_out)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                fbank_mask += fbank_mask_i
                fbank_beg.append(fbank_beg_i)

            if len(input_ids) > self.max_token_length:
                logging.info(
                    f"input_ids > max_token_length: {len(input_ids)}>{self.max_token_length}, {item}"
                )
                badcase_flag = True
            if badcase_flag:
                continue
            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            fbank = speech[0, :, :]
            fbank_lens = speech_lengths
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)

            output = {
                "speech": fbank,
                "speech_lengths": fbank_lens,
                "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
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


@tables.register("dataset_classes", "OpenAIDatasetMultiTurn")
class OpenAIDatasetMultiTurn(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
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
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf"))
        self.preprocessor_text = preprocessor_text

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

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")
        # self.kwargs = kwargs
        self.max_token_length = kwargs.get("max_token_length", 1500)
        self.batch_size_scale_ratio_max = kwargs.get("batch_size_scale_ratio_max", 1.5)
        self.batch_size_token_max = kwargs.get("batch_size_token_max", 2500)
        self.multiturn_num_max = kwargs.get("multiturn_num_max", 5)
        self.max_source_length = kwargs.get("max_source_length", 3000)

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        # import pdb
        #
        # pdb.set_trace()

        output = None

        for idx in range(self.retry):
            badcase_flag = False
            if idx == 0:
                index_cur = index
            else:
                index_cur = torch.randint(0, len(self.index_ds), ()).item()

            item = self.index_ds[index_cur]

            system = item["system"]
            user = item["user"]
            assistant = item["assistant"]

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

                if i == 0:
                    source_input = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    source_input = (
                        f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    )

                splits = self.pattern.split(source_input)
                source_ids = []
                fbank_i = []
                fbank_mask_i = []
                fake_token_len_i = 0
                fbank_beg_i = -1
                fbank_lens_i = []
                for k, sub_str in enumerate(splits):
                    if not sub_str.startswith("<|startofspeech|>"):
                        sub_token = self.tokenizer.encode(sub_str)
                        source_ids += sub_token
                        fbank_mask_i += [0] * len(sub_token)
                    else:
                        sub_str = sub_str.replace("<|startofspeech|>", "").replace(
                            "<|endofspeech|>", ""
                        )
                        if sub_str.startswith("!"):
                            try:
                                data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)
                            except Exception as e:
                                logging.error(
                                    f"Loading wav failed! {str(e)}, {traceback.format_exc()}"
                                )
                                badcase_flag = True
                                continue
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
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            # if speech_lengths > self.batch_size:
                            #     continue

                            olens = 1 + (speech_lengths[0].item() - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            fake_token_len_i = (olens - 1) // 2 + 1
                            fake_token = [0] * fake_token_len_i
                            fbank_beg_i = len(source_ids)
                            source_ids += fake_token
                            fbank_mask_i += [1] * len(fake_token)

                if badcase_flag:
                    continue

                fbank_beg += [fbank_beg_i + len(input_ids)]
                fake_token_len += [fake_token_len_i]
                source_mask = [-100] * len(source_ids)
                target_out = f"{target_out}<|im_end|>"
                target_ids = self.tokenizer.encode(target_out)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                fbank.append(speech[0, :, :])
                fbank_mask += fbank_mask_i
                fbank_lens.append(speech_lengths)

            if badcase_flag:
                continue

            input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [: self.max_token_length]
            attention_mask = torch.tensor([1] * len(input_ids), dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)  # [: self.max_token_length]

            # fbank = speech[0, :, :]
            # fbank_lens = torch.tensor(fbank_lens, dtype=torch.int32)
            fbank_mask = torch.tensor(fbank_mask, dtype=torch.float32)
            fbank_beg = torch.tensor(fbank_beg, dtype=torch.int32)
            fake_token_len = torch.tensor(fake_token_len, dtype=torch.int32)

            output = {
                "speech": fbank,
                "speech_lengths": fbank_lens,
                "fbank_mask": fbank_mask,
                "fbank_beg": fbank_beg,
                "fake_token_len": fake_token_len,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels_ids": labels,
            }
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
