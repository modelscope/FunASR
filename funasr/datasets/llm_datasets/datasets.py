import torch
import copy

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "AudioLLMNARDataset")
class AudioLLMNARDataset(torch.utils.data.Dataset):
    """
    AudioLLMDataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf", {})
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf", {}))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.float_pad_value = float_pad_value
        self.prompt = kwargs.get("prompt", "Please copy the following text.")
        self.prompt_pre = "USER: \nINSTRUCTION: {}\nINPUT: ".format(
            self.prompt
        )  # "USER: \nINSTRUCTION: {}\nINPUT: {}\nASSISTANT: "
        self.prompt_af = ""
        self.IGNORE_INDEX = kwargs.get("IGNORE_INDEX", -100)
        self.int_pad_value = self.IGNORE_INDEX

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        item = self.index_ds[index]
        # import pdb;
        # pdb.set_trace()
        source = item["source"]
        data_src = load_audio_text_image_video(source, fs=self.fs)
        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=self.fs)
        speech, speech_lengths = extract_fbank(
            data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
        )  # speech: [b, T, d]
        speech = speech.squeeze(0)

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)

        prompt_ids_pre = self.tokenizer.encode(self.prompt_pre)  # [bos,prompt]
        prompt_ids_length = len(prompt_ids_pre)

        # bos prompt audio bos target
        # prompt_input = "{}{}".format(self.prompt_pre, target)
        # prompt_input_ids = self.tokenizer.encode(prompt_input) #[bos, prompt, input]
        # audio_length = len(prompt_input_ids) - prompt_ids_length
        target_ids = self.tokenizer.encode(target)
        if target_ids[0] == self.tokenizer.bos_token_id:
            target_ids = target_ids[1:]
        target_ids_length = len(target_ids)
        audio_length = target_ids_length
        input_ids = (
            prompt_ids_pre + target_ids + [self.tokenizer.pad_token_id] + target_ids
        )  # [bos, prompt, input, pad, target]
        input_ids = torch.tensor(
            copy.deepcopy(input_ids), dtype=torch.int64
        )  # [bos, prompt, input, pad, target]
        input_ids[prompt_ids_length : prompt_ids_length + audio_length] = (
            -1
        )  # [bos, prompt,-1, pad, target] # it is no need, only for check
        attention_mask = input_ids.ge(-1)  # [true, true, true, true, true], length mask

        # bos prompt audio target eos
        # prompt_answer = "{}{}".format(self.prompt_pre, target)
        # prompt_answer_ids = self.tokenizer.encode(prompt_answer) #[bos, prompt, input]
        # answer_length = len(prompt_answer_ids) - prompt_ids_length
        target_ids = self.tokenizer.encode(target)
        if target_ids[0] == self.tokenizer.bos_token_id:
            target_ids = target_ids[1:]
        # target_ids_length = len(target_ids)
        labels_ids = (
            prompt_ids_pre + target_ids + target_ids + [self.tokenizer.eos_token_id]
        )  # [bos, prompt, input, target, eos]
        labels_ids = torch.tensor(
            copy.deepcopy(labels_ids), dtype=torch.int64
        )  # [bos, prompt, input, target, eos]
        labels_ids[:prompt_ids_length] = -1  # [-1, -1, input, target, eos]
        label_mask = labels_ids.ge(0)  # [false, false, true, true, true], length mask
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-1, -1, input, target, eos]

        audio_mask = (
            [0] * prompt_ids_length + [1] * audio_length + [0] * target_ids_length + [0]
        )  # [0, 0, 1, 0, 0]
        audio_mask = torch.tensor(audio_mask, dtype=torch.float32)

        ids = target_ids  # self.tokenizer.encode(target) # token ids is different from labels_ids
        text = torch.tensor(ids, dtype=torch.int64)
        text_lengths = torch.tensor([len(ids)], dtype=torch.int32)

        prompt_bos_length = torch.tensor([len(prompt_ids_pre)], dtype=torch.int32)

        return {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_ids": labels_ids,
            "label_mask": label_mask,
            "audio_mask": audio_mask,
            "prompt_bos_length": prompt_bos_length,
        }

    def collator(self, samples: list = None):
        outputs = {}
        for sample in samples:
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
        return outputs


@tables.register("dataset_classes", "AudioLLMDataset")
class AudioLLMDataset(torch.utils.data.Dataset):
    """
    AudioLLMDataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf", {})
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf", {}))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.float_pad_value = float_pad_value
        self.prompt = kwargs.get("prompt", "Transcribe speech to text.")
        self.prompt_pre = "USER: \nINSTRUCTION: {}\nINPUT: ".format(
            self.prompt
        )  # "USER: \nINSTRUCTION: {}\nnINPUT: {}\nASSISTANT: "
        self.prompt_af = ""
        self.IGNORE_INDEX = kwargs.get("IGNORE_INDEX", -100)
        self.int_pad_value = self.IGNORE_INDEX

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        item = self.index_ds[index]
        # import pdb;
        # pdb.set_trace()
        source = item["source"]
        data_src = load_audio_text_image_video(source, fs=self.fs)
        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=self.fs)
        speech, speech_lengths = extract_fbank(
            data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
        )  # speech: [b, T, d]
        speech = speech.squeeze(0)

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)

        prompt_ids_pre = self.tokenizer.encode(self.prompt_pre)  # [bos,prompt]
        prompt_ids_length = len(prompt_ids_pre)

        prompt_input = "{}{}".format(self.prompt_pre, target)
        prompt_input_ids = self.tokenizer.encode(prompt_input)
        audio_length = len(prompt_input_ids) - prompt_ids_length
        input_ids = prompt_input_ids + [self.tokenizer.pad_token_id]
        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [bos, prompt, input, pad]
        input_ids[prompt_ids_length:] = -1  # [bos, prompt,-1,-1]
        attention_mask = input_ids.ge(-1)  # [true, true, true, true], length mask

        prompt_answer = "{}{}".format(self.prompt_pre, target)
        prompt_answer_ids = self.tokenizer.encode(prompt_answer)
        answer_length = len(prompt_answer_ids) - prompt_ids_length
        labels_ids = copy.deepcopy(prompt_input_ids) + [self.tokenizer.eos_token_id]
        labels_ids = torch.tensor(labels_ids, dtype=torch.int64)  # [bos, prompt, input, eos]
        labels_ids[:prompt_ids_length] = -1  # [-1, -1, input, eos]
        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,input,eos]

        audio_mask = [0] * prompt_ids_length + [1] * audio_length + [0]
        audio_mask = torch.tensor(audio_mask, dtype=torch.float32)

        ids = self.tokenizer.encode(target)  # token ids is different from labels_ids
        text = torch.tensor(ids, dtype=torch.int64)
        text_lengths = torch.tensor([len(ids)], dtype=torch.int32)

        return {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_ids": labels_ids,
            "label_mask": label_mask,
            "audio_mask": audio_mask,
        }

    def collator(self, samples: list = None):
        outputs = {}
        for sample in samples:
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
        return outputs


@tables.register("dataset_classes", "AudioLLMARDataset")
class AudioLLMARDataset(torch.utils.data.Dataset):
    """
    AudioLLMDataset
    """

    def __init__(
        self,
        path,
        index_ds: str = None,
        frontend=None,
        tokenizer=None,
        int_pad_value: int = -1,
        float_pad_value: float = 0.0,
        **kwargs
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(
                **kwargs.get("preprocessor_speech_conf", {})
            )
        self.preprocessor_speech = preprocessor_speech
        preprocessor_text = kwargs.get("preprocessor_text", None)
        if preprocessor_text:
            preprocessor_text_class = tables.preprocessor_classes.get(preprocessor_text)
            preprocessor_text = preprocessor_text_class(**kwargs.get("preprocessor_text_conf", {}))
        self.preprocessor_text = preprocessor_text

        self.frontend = frontend
        self.fs = 16000 if frontend is None else frontend.fs
        self.data_type = "sound"
        self.tokenizer = tokenizer

        self.float_pad_value = float_pad_value
        self.prompt = kwargs.get("prompt", "Transcribe speech to text.")
        self.prompt_pre = "USER: \nINSTRUCTION: {}\nINPUT: ".format(
            self.prompt
        )  # "USER: \nINSTRUCTION: {}\nnINPUT: {}\nASSISTANT: "
        self.prompt_af = ""
        self.IGNORE_INDEX = kwargs.get("IGNORE_INDEX", -100)
        self.int_pad_value = self.IGNORE_INDEX

    def get_source_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_source_len(item)

    def get_target_len(self, index):
        item = self.index_ds[index]
        return self.index_ds.get_target_len(item)

    def __len__(self):
        return len(self.index_ds)

    def __getitem__(self, index):
        item = self.index_ds[index]
        # import pdb;
        # pdb.set_trace()
        source = item["source"]
        data_src = load_audio_text_image_video(source, fs=self.fs)
        if self.preprocessor_speech:
            data_src = self.preprocessor_speech(data_src, fs=self.fs)
        speech, speech_lengths = extract_fbank(
            data_src, data_type=self.data_type, frontend=self.frontend, is_final=True
        )  # speech: [b, T, d]
        speech = speech.squeeze(0)

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)

        prompt_ids_pre = self.tokenizer.encode(self.prompt_pre)  # [bos,prompt]
        prompt_ids_length = len(prompt_ids_pre)

        prompt_input = "{}{}".format(self.prompt_pre, target)
        prompt_input_ids = self.tokenizer.encode(prompt_input)
        audio_length = len(prompt_input_ids) - prompt_ids_length
        input_ids = prompt_input_ids + [self.tokenizer.pad_token_id]
        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [bos, prompt, input, pad]
        input_ids[prompt_ids_length:] = -1  # [bos, prompt,-1,-1]
        attention_mask = input_ids.ge(-1)  # [true, true, true, true], length mask

        prompt_answer = "{}{}".format(self.prompt_pre, target)
        prompt_answer_ids = self.tokenizer.encode(prompt_answer)
        answer_length = len(prompt_answer_ids) - prompt_ids_length
        labels_ids = copy.deepcopy(prompt_input_ids) + [self.tokenizer.eos_token_id]
        labels_ids = torch.tensor(labels_ids, dtype=torch.int64)  # [bos, prompt, input, eos]
        labels_ids[:prompt_ids_length] = -1  # [-1, -1, input, eos]
        label_mask = labels_ids.ge(0)  # [False,False,True,True]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100,-100,input,eos]

        audio_mask = [0] * prompt_ids_length + [1] * audio_length + [0]
        audio_mask = torch.tensor(audio_mask, dtype=torch.float32)

        ids = self.tokenizer.encode(target)  # token ids is different from labels_ids
        text = torch.tensor(ids, dtype=torch.int64)
        text_lengths = torch.tensor([len(ids)], dtype=torch.int32)

        return {
            "speech": speech,
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_ids": labels_ids,
            "label_mask": label_mask,
            "audio_mask": audio_mask,
        }

    def collator(self, samples: list = None):
        outputs = {}
        for sample in samples:
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
        return outputs
