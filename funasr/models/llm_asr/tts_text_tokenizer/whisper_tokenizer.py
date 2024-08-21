import copy
import json
import os
import random
import re
from typing import Iterable, List, Union
import numpy as np


class WhisperRichTtsFrdTokenizer:
    def __init__(
            self,
            token_path: str,
            num_languages: int,
            task: str = None,
            language: str = None,
            ttsfrd_type: str = None,
            p_word2phn: float = 0.5,
            ttsfrd_model: str = None,
    ):
        import funasr.models.llm_asr.tts_text_tokenizer.voice_echo_rich_tokenizer as tokenizer

        self.token_path = token_path
        self.num_languages = num_languages
        self.language = language
        self.task = task
        self.ttsfrd_type = ttsfrd_type
        self.p_word2phn = p_word2phn
        # print('token_path:',token_path)
        if token_path == "whisper_en" or token_path == "whisper_gpt2" or token_path == "gpt2":
            self.tokenizer = tokenizer.get_tokenizer(multilingual=False, num_languages=num_languages)
        elif token_path == "whisper_multilingual" or token_path == "multilingual":
            self.tokenizer = tokenizer.get_tokenizer(
                multilingual=True, language=self.language, task=self.task, num_languages=num_languages
            )
        else:#
            self.tokenizer = tokenizer.get_tokenizer(
                multilingual=True, language=self.language, task=self.task, num_languages=num_languages,
                encoding_path=token_path, ttsfrd_name=ttsfrd_type
            )
        if ttsfrd_model is not None and os.path.isdir(ttsfrd_model):
            from funasr.models.llm_asr.tts_text_tokenizer.phoneme_tokenizer import TtsFrdRich
            self.ttsfrd_tokenizer = TtsFrdRich(remove_boundary=True, token_type="word2phn")
            self.ttsfrd_tokenizer.build(ttsfrd_model)
        else:
            self.ttsfrd_tokenizer = None
        # self.tokenizer = copy.deepcopy(self.tokenizer)


    def text_mixing(self, line: str) -> str:
        try:
            data_info = json.loads(line)
            # ttsfrd_word2phn info
            if isinstance(data_info, dict) and "raw" in data_info and "word2phn" in data_info:
                raw_text = data_info["raw"]
                ttsfrd_word2phn = data_info["word2phn"]
                if random.random() < self.p_word2phn:
                    ret_text = ""
                    for ttsfrd_word in ttsfrd_word2phn:
                        for word_str, phn_list in ttsfrd_word.items():
                            if phn_list is not None:
                                if random.random() < self.p_word2phn:
                                    ret_text = ret_text + "".join([f"<|@{p}|>" for p in phn_list])
                                else:
                                    ret_text += word_str
                            else:
                                ret_text += word_str
                else:
                    ret_text = raw_text
            else:
                ret_text = line
        except json.JSONDecodeError:
            ret_text = line
        return ret_text

    def get_num_vocabulary_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def text2ids(self, line: str, language: str) -> List[int]:
        language_tok = "<|" + language + "|>"
        assert language_tok in self.tokenizer.special_tokens, "Language token not found, lang: {}, line: {}".format(language_tok, line)
        # line = re.sub(r'<(\d+\.\d+)>', r'<|\1|>', line)
        pattern = re.compile(r'<|(\d+\.\d+)|>')
        with_timestamps = pattern.search(line)
        if with_timestamps:
            sot_tok = [self.tokenizer.special_tokens.get(language_tok), self.tokenizer.transcribe]
            allowed_special = set([f"<|{i * 0.02:.2f}|>" for i in range(1501)])
            encoded_line = self.tokenizer.encode(line, allowed_special=allowed_special)
        else:
            sot_tok = [self.tokenizer.special_tokens.get(language_tok), self.tokenizer.transcribe, self.tokenizer.no_timestamps]
            encoded_line = self.tokenizer.encode(line)
        return sot_tok + encoded_line

    def ids2text(self, integers: Union[np.ndarray, Iterable[int]]) -> str:
        return self.tokenizer.decode_with_timestamps(integers)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        return [self.tokenizer.decode_with_timestamps([i]) for i in integers]

    def text2tokens(self, line: str, endofprompt="<|endofprompt|>", sil="<|sil|>") -> List[str]:
        # keep prompt and sil unchanged
        prompt_text = ""
        st_sil, ed_sil = False, False
        if endofprompt in line:
            pos = line.find(endofprompt)
            prompt_text = line[:pos+len(endofprompt)]
            line = line[pos+len(endofprompt):]
        if line.startswith(sil):
            line = line[len(sil):]
            st_sil = True
        if line.endswith(sil):
            line = line[:-len(sil)]
            ed_sil = True

        # token to phone and mixup
        if self.ttsfrd_tokenizer is not None:
            line = self.ttsfrd_tokenizer(line)
        if self.ttsfrd_type is not None:
            line = self.text_mixing(line)

        # add prompt text and sil back
        if st_sil:
            line = sil + line
        if ed_sil:
            line = line + sil
        line = prompt_text + line

        return self.tokenizer.encode(line, allowed_special="all")

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.tokenizer.decode_with_timestamps(tokens)

    # def get_sot(self, sot_template: str, lang: str = None) -> List[int]:
    #     if lang is not None:
    #         lang = lang.replace("<", "").replace(">", "").replace("|", "")
    #         sot = sot_template.replace("LANG", lang)
    #     else:
    #         if "<|LANG|>" in sot_template:
    #             sot = sot_template.split("<|LANG|>", 1)[0]
    #         else:
    #             sot = sot_template
    #     sot_tok = self.tokenizer.encode(sot, allowed_special="all")
    #     return sot_tok

    def get_sot(self, language: str = None, with_timestamps: bool = False) -> List[int]:
        if language is not None:
            language_tok = "<|" + language + "|>"
            assert language_tok in self.tokenizer.special_tokens
            if with_timestamps:
                sot_tok = [self.tokenizer.sot, self.tokenizer.special_tokens.get(language_tok), self.tokenizer.transcribe]
            else:
                sot_tok = [self.tokenizer.sot, self.tokenizer.special_tokens.get(language_tok), self.tokenizer.transcribe, self.tokenizer.no_timestamps]
        else:
            sot_tok = [self.tokenizer.sot]
        return sot_tok

    def get_all_languages(self) -> List[str]:
        return list(self.tokenizer.all_language_codes)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model_type={self.token_path}, "
            f"language={self.language}, ttsfrd={self.ttsfrd_type})"
        )
