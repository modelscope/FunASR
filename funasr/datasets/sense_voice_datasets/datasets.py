import torch
import random

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


@tables.register("dataset_classes", "SenseVoiceDataset")
class SenseVoiceDataset(torch.utils.data.Dataset):
    """
    SenseVoiceDataset
    """
    def __init__(self,
                 path,
                 index_ds: str = None,
                 frontend=None,
                 tokenizer=None,
                 int_pad_value: int = -1,
                 float_pad_value: float = 0.0,
                  **kwargs):
        super().__init__()
        index_ds_class = tables.index_ds_classes.get(index_ds)
        self.index_ds = index_ds_class(path, **kwargs)
        preprocessor_speech = kwargs.get("preprocessor_speech", None)
        if preprocessor_speech:
            preprocessor_speech_class = tables.preprocessor_classes.get(preprocessor_speech)
            preprocessor_speech = preprocessor_speech_class(**kwargs.get("preprocessor_speech_conf"))
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
        speech, speech_lengths = extract_fbank(data_src, data_type=self.data_type, frontend=self.frontend, is_final=True) # speech: [b, T, d]
        speech = speech.permute(0, 2, 1)
        target = item["target"]
        if self.preprocessor_text:
            target = self.preprocessor_text(target)
        
        task = item.get("prompt", "<|ASR|>")
        text_language = item.get("text_language", "<|zh|>")

        prompt = f"{self.sos}{task}{text_language}"
        prompt_ids = self.tokenizer.encode(prompt, allowed_special="all")
        prompt_ids_len = len(prompt_ids) - 1 # [sos, task]

        target_ids = self.tokenizer.encode(target, allowed_special="all")
        target_ids_len = len(target_ids) + 1 # [lid, text]
        
        eos = self.tokenizer.encode(self.eos, allowed_special="all") # [eos]
        
        ids = prompt_ids + target_ids + eos
        ids_lengths = len(ids)
        
        text = torch.tensor(ids, dtype=torch.int64)
        text_lengths = torch.tensor([ids_lengths], dtype=torch.int32)

        target_mask = [0] * (prompt_ids_len) + [1] * (target_ids_len) + [1]  # [sos, task, lid, text, eos]: [0, 0, 1, 1, 1]
        target_mask = torch.tensor(target_mask, dtype=torch.float32)

        return {"speech": speech[0, :, :],
                "speech_lengths": speech_lengths,
                "text": text,
                "text_lengths": text_lengths,
                "target_mask": target_mask,
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
                
                outputs[key] = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=pad_value)
        return outputs


