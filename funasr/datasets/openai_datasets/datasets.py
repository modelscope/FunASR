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
        self.retry = kwargs.get("retry", 5)

        self.permute = False
        from funasr.frontends.whisper_frontend import WhisperFrontend

        if isinstance(self.frontend, WhisperFrontend):
            self.permute = True

        self.pattern = re.compile(r"(<\|startofspeech\|>.*?<\|endofspeech\|>)")

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

                            data_src = load_audio_text_image_video(sub_str[1:], fs=self.fs)

                            speech, speech_lengths = extract_fbank(
                                data_src,
                                data_type=self.data_type,
                                frontend=self.frontend,
                                is_final=True,
                            )  # speech: [b, T, d]
                            if self.permute:
                                speech = speech.permute(0, 2, 1)
                            if speech_lengths > self.batch_size:
                                continue

                            fbank_lens = speech_lengths[0].item()
                            olens = 1 + (fbanks_len - 3 + 2 * 1) // 2
                            olens = 1 + (olens - 3 + 2 * 1) // 2
                            sub_token_len = (olens - 1) // 2 + 1
                            sub_token = [0] * sub_token_len[0]
                            fbank_beg_i = [len(source_ids)]
                            source_ids += sub_token
                            fbank_mask_i += [1] * len(sub_token)

                source_mask = [-100] * len(source_ids)
                target_out = f"{target_out}<|im_end|>"
                target_ids = tokenizer.encode(target_out)
                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
                fbank_mask += fbank_mask_i
                fbank_beg.append(fbank_beg_i)

            input_ids = torch.tensor(input_ids, dtype=torch.int64)
            attention_mask = torch.tensor([len(input_ids)], dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)

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
            for i in range(10):
                outputs = self._filter_badcase(outputs, i=i)

        return outputs

    def _filter_badcase(self, outputs, i=0):
        b, t, _ = outputs["speech"].shape

        if b * t > self.batch_size * 1.25:
            beg = torch.randint(0, 2, ()).item()
            if b < 2:
                beg = 0
            logging.info(
                f"Warning, b * t: {b * t} > {self.batch_size}, drop half data {i}th, beg:{beg}"
            )
            for key, data_list in outputs.items():
                outputs[key] = outputs[key][beg : beg + b : 2]

            speech_lengths_max = outputs["speech_lengths"].max().item()
            outputs["speech"] = outputs["speech"][:, :speech_lengths_max, :]
            text_lengths_max = outputs["text_lengths"].max().item()
            outputs["text"] = outputs["text"][:, :text_lengths_max]
            target_mask_lengths_max = outputs["target_mask_lengths"].max().item()
            outputs["target_mask"] = outputs["target_mask"][:, :target_mask_lengths_max]

        return outputs
