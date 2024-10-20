import torch
import random


from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "KwsMTDataset")
class KwsMTDataset(torch.utils.data.Dataset):
    """
    KwsMTDataset, support multi tokenizers
    """
    def __init__(self,
                 path,
                 index_ds: str = None,
                 frontend=None,
                 tokenizer=None,
                 is_training: bool = True,
                 int_pad_value: int = -1,
                 float_pad_value: float = 0.0,
                 **kwargs,
    ):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)

        self.preprocessor_speech = None
        self.preprocessor_text = None

        if is_training:
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
        print(tokenizer)
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
        ) # speech: [b, T, d]

        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)

        if self.tokenizer[0]:
            ids = self.tokenizer[0].encode(target)
            text = torch.tensor(ids, dtype=torch.int64)
            # print("target: ", target,  ", ids: ", str(ids))
        else:
            ids = target
            text = ids

        if self.tokenizer[1]:
            ids2 = self.tokenizer[1].encode(target)
            text2 = torch.tensor(ids2, dtype=torch.int64)
            # print("target: ", target,  ", ids2: ", str(ids2))
        else:
            ids2 = target
            text2 = ids2

        ids_lengths = len(ids)
        text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

        ids2_lengths = len(ids2)
        text2_lengths = torch.tensor([ids2_lengths], dtype=torch.int32)

        return {"speech": speech[0, :, :],
                "speech_lengths": speech_lengths,
                "text": text,
                "text_lengths": text_lengths,
                "text2": text2,
                "text2_lengths": text2_lengths,
                }


    def collator(self, samples: list=None):
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
