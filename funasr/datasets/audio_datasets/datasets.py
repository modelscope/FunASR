import torch
import random

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "AudioDataset")
class AudioDataset(torch.utils.data.Dataset):
    """
    AudioDataset
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

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)
        if self.tokenizer:
            ids = self.tokenizer.encode(target)
            text = torch.tensor(ids, dtype=torch.int64)
        else:
            ids = target
            text = ids
        ids_lengths = len(ids)
        text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

        return {
            "speech": speech[0, :, :],
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
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


@tables.register("dataset_classes", "AudioDatasetHotword")
class AudioDatasetHotword(AudioDataset):
    # for finetuning contextual_paraformer and seaco_paraformer
    def __init__(
        self,
        *args,
        seaco_id: bool = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seaco_id = seaco_id

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

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)
        if self.tokenizer:
            ids = self.tokenizer.encode(target)
            text = torch.tensor(ids, dtype=torch.int64)
        else:
            ids = target
            text = ids
        ids_lengths = len(ids)
        text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

        def generate_index(
            length,
            hotword_min_length=2,
            hotword_max_length=8,
            sample_rate=0.75,
            double_rate=0.1,
            pre_prob=0.0,
            pre_index=None,
            pre_hwlist=None,
        ):
            if length < hotword_min_length:
                return [-1]
            if random.random() < sample_rate:
                if pre_prob > 0 and random.random() < pre_prob and pre_index is not None:
                    return pre_index
                if length == hotword_min_length:
                    return [0, length - 1]
                elif (
                    random.random() < double_rate
                    and length > hotword_max_length + hotword_min_length + 2
                ):
                    # sample two hotwords in a sentence
                    _max_hw_length = min(hotword_max_length, length // 2)
                    # first hotword
                    start1 = random.randint(0, length // 3)
                    end1 = random.randint(
                        start1 + hotword_min_length - 1, start1 + _max_hw_length - 1
                    )
                    # second hotword
                    start2 = random.randint(end1 + 1, length - hotword_min_length)
                    end2 = random.randint(
                        min(length - 1, start2 + hotword_min_length - 1),
                        min(length - 1, start2 + hotword_max_length - 1),
                    )
                    return [start1, end1, start2, end2]
                else:  # single hotword
                    start = random.randint(0, length - hotword_min_length)
                    end = random.randint(
                        min(length - 1, start + hotword_min_length - 1),
                        min(length - 1, start + hotword_max_length - 1),
                    )
                    return [start, end]
            else:
                return [-1]

        hotword_indx = generate_index(text_lengths[0])
        return {
            "speech": speech[0, :, :],
            "speech_lengths": speech_lengths,
            "text": text,
            "text_lengths": text_lengths,
            "hotword_indx": hotword_indx,
            "seaco_id": self.seaco_id,
        }

    def collator(self, samples: list = None):
        outputs = {}
        hotword_indxs = []
        seaco_id = samples[0]["seaco_id"]
        for sample in samples:
            for key in sample.keys():
                if key == "seaco_id":
                    continue
                elif key == "hotword_indx":
                    hotword_indxs.append(sample[key])
                else:
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

        hotword_list, hotword_lengths = [], []
        text = outputs["text"]
        seaco_label_pad = torch.ones_like(text) * -1 if seaco_id else None
        for b, (hotword_indx, one_text, length) in enumerate(
            zip(hotword_indxs, text, outputs["text_lengths"])
        ):
            length = length[0]
            if seaco_label_pad is not None:
                seaco_label_pad[b][:length] = seaco_id
            if hotword_indx[0] != -1:
                start, end = int(hotword_indx[0]), int(hotword_indx[1])
                hotword = one_text[start : end + 1]
                hotword_list.append(hotword)
                hotword_lengths.append(end - start + 1)
                if seaco_label_pad is not None:
                    seaco_label_pad[b][start : end + 1] = one_text[start : end + 1]
                if len(hotword_indx) == 4 and hotword_indx[2] != -1:
                    # the second hotword if exist
                    start, end = int(hotword_indx[2]), int(hotword_indx[3])
                    hotword_list.append(one_text[start : end + 1])
                    hotword_lengths.append(end - start + 1)
                    if seaco_label_pad is not None:
                        seaco_label_pad[b][start : end + 1] = one_text[start : end + 1]
        hotword_list.append(torch.tensor([1]))
        hotword_lengths.append(1)
        hotword_pad = torch.nn.utils.rnn.pad_sequence(
            hotword_list, batch_first=True, padding_value=0
        )
        outputs["hotword_pad"] = hotword_pad
        outputs["hotword_lengths"] = torch.tensor(hotword_lengths, dtype=torch.int32)
        if seaco_label_pad is not None:
            outputs["seaco_label_pad"] = seaco_label_pad
        return outputs
