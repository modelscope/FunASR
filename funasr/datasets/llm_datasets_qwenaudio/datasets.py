import torch
import copy

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "AudioLLMQwenAudioDataset")
class AudioLLMQwenAudioDataset(torch.utils.data.Dataset):
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
        # self.prompt_pre = "USER: \nINSTRUCTION: {}\nINPUT: ".format(self.prompt)  # "USER: \nINSTRUCTION: {}\nnINPUT: {}\nASSISTANT: "
        self.prompt_af = ""
        self.IGNORE_INDEX = kwargs.get("IGNORE_INDEX", -100)
        self.int_pad_value = self.IGNORE_INDEX
        self.audio_adaptor_downsample_rate = kwargs.get("audio_adaptor_downsample_rate", 5)
        self.audio_encoder_downsample_rate = kwargs.get("audio_encoder_downsample_rate", 2)
        self.prompt_template = "{}"
        self.answer_template = "{}"

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

        audio_pseudo_length = (
            (speech.shape[0] + 1)
            // self.audio_adaptor_downsample_rate
            // self.audio_encoder_downsample_rate
        )
        audio_pseudo = torch.full((audio_pseudo_length,), -1)  # placeholder

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)

        self.prompt_pre = self.prompt_template.format(self.prompt)
        prompt_ids_pre = self.tokenizer.encode(self.prompt_pre)  # [bos,prompt]
        prompt_pre_length = len(prompt_ids_pre)

        # input
        input = self.answer_template.format(target.lower())
        prompt_input = "{}{}".format(self.prompt_pre, input)
        prompt_input_ids = self.tokenizer.encode(prompt_input)  # [bos, prompt, input]
        # audio_length = len(prompt_input_ids) - prompt_pre_length
        input_ids = prompt_input_ids + [self.tokenizer.pad_token_id]  # [bos, prompt, input, pad]
        input_ids_length = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.int64)  # [bos, prompt, input, pad]
        input_ids = torch.cat((audio_pseudo, input_ids))  # [audio, bos, prompt, input, pad]
        # input_ids[:audio_pseudo_length] = -1 # [-1, bos, prompt, input, pad]
        attention_mask = input_ids.ge(-1)  # [true, true, true, true, true], length mask
        # input_ids[prompt_pre_length:] = -1  # [bos, prompt,-1,-1]
        # attention_mask = input_ids.ge(-1)  # [true, true, true, true], length mask

        # label
        answer = self.answer_template.format(target.lower())
        prompt_answer = "{}{}".format(self.prompt_pre, answer)
        prompt_answer_ids = self.tokenizer.encode(prompt_answer)
        # answer_length = len(prompt_answer_ids) - prompt_pre_length
        labels_ids = copy.deepcopy(prompt_answer_ids) + [self.tokenizer.eos_token_id]
        labels_ids = torch.tensor(labels_ids, dtype=torch.int64)  # [bos, prompt, answer, eos]
        labels_ids = torch.cat((audio_pseudo, labels_ids))  # [audio, bos, prompt, answer, eos]
        labels_ids[: audio_pseudo_length + prompt_pre_length] = -1  # [-1, -1, -1, answer, eos]
        # labels_ids[:prompt_pre_length] = -1  # [-1, -1, input, eos]
        label_mask = labels_ids.ge(0)  # [false, false, false, true, true]
        labels_ids[~label_mask] = self.IGNORE_INDEX  # [-100, -100, -100, answer, eos]

        # audio_mask for input_ids
        audio_mask = [1] * audio_pseudo_length + [0] * input_ids_length
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
