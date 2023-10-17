#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from typing import Optional
from typing import Union

import numpy as np
import torch
import os

from funasr.build_utils.build_model_from_file import build_model_from_file
from funasr.datasets.preprocessor import CodeMixTokenizerCommonPreprocessor
from funasr.datasets.preprocessor import split_to_mini_sentence
from funasr.torch_utils.device_funcs import to_device
from funasr.torch_utils.forward_adaptor import ForwardAdaptor


class Text2Punc:

    def __init__(
            self,
            train_config: Optional[str],
            model_file: Optional[str],
            device: str = "cpu",
            dtype: str = "float32",
    ):
        #  Build Model
        model, train_args = build_model_from_file(train_config, model_file, None, device, task_name="punc")
        self.device = device
        # Wrape model to make model.nll() data-parallel
        self.wrapped_model = ForwardAdaptor(model, "inference")
        self.wrapped_model.to(dtype=getattr(torch, dtype)).to(device=device).eval()
        # logging.info(f"Model:\n{model}")
        self.punc_list = train_args.punc_list
        self.period = 0
        for i in range(len(self.punc_list)):
            if self.punc_list[i] == ",":
                self.punc_list[i] = "，"
            elif self.punc_list[i] == "?":
                self.punc_list[i] = "？"
            elif self.punc_list[i] == "。":
                self.period = i
        self.seg_dict_file = None
        self.seg_jieba = False
        if "seg_jieba" in train_args:
            self.seg_jieba = train_args.seg_jieba
            self.seg_dict_file = os.path.dirname(model_file)+"/"+ "jieba_usr_dict"
        self.preprocessor = CodeMixTokenizerCommonPreprocessor(
            train=False,
            token_type=train_args.token_type,
            token_list=train_args.token_list,
            bpemodel=train_args.bpemodel,
            text_cleaner=train_args.cleaner,
            g2p_type=train_args.g2p,
            text_name="text",
            non_linguistic_symbols=train_args.non_linguistic_symbols,
            seg_jieba=self.seg_jieba,
            seg_dict_file=self.seg_dict_file
        )

    @torch.no_grad()
    def __call__(self, text: Union[list, str], split_size=20):
        data = {"text": text}
        result = self.preprocessor(data=data, uid="12938712838719")
        split_text = self.preprocessor.pop_split_text_data(result)
        mini_sentences = split_to_mini_sentence(split_text, split_size)
        mini_sentences_id = split_to_mini_sentence(data["text"], split_size)
        assert len(mini_sentences) == len(mini_sentences_id)
        cache_sent = []
        cache_sent_id = torch.from_numpy(np.array([], dtype='int32'))
        new_mini_sentence = ""
        new_mini_sentence_punc = []
        cache_pop_trigger_limit = 200
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.concatenate((cache_sent_id, mini_sentence_id), axis=0)
            data = {
                "text": torch.unsqueeze(torch.from_numpy(mini_sentence_id), 0),
                "text_lengths": torch.from_numpy(np.array([len(mini_sentence_id)], dtype='int32')),
            }
            data = to_device(data, self.device)
            y, _ = self.wrapped_model(**data)
            _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
            punctuations = indices
            if indices.size()[0] != 1:
                punctuations = torch.squeeze(indices)
            assert punctuations.size()[0] == len(mini_sentence)

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if self.punc_list[punctuations[i]] == "。" or self.punc_list[punctuations[i]] == "？":
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if sentenceEnd < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.period
                cache_sent = mini_sentence[sentenceEnd + 1:]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1:]
                mini_sentence = mini_sentence[0:sentenceEnd + 1]
                punctuations = punctuations[0:sentenceEnd + 1]

            # if len(punctuations) == 0:
            #    continue

            punctuations_np = punctuations.cpu().numpy()
            new_mini_sentence_punc += [int(x) for x in punctuations_np]
            words_with_punc = []
            for i in range(len(mini_sentence)):
                if (i==0 or self.punc_list[punctuations[i-1]] == "。" or self.punc_list[punctuations[i-1]] == "？") and len(mini_sentence[i][0].encode()) == 1:
                    mini_sentence[i] = mini_sentence[i].capitalize()
                if i == 0:
                    if len(mini_sentence[i][0].encode()) == 1:
                        mini_sentence[i] = " " + mini_sentence[i]
                if i > 0:
                    if len(mini_sentence[i][0].encode()) == 1 and len(mini_sentence[i - 1][0].encode()) == 1:
                        mini_sentence[i] = " " + mini_sentence[i]
                words_with_punc.append(mini_sentence[i])
                if self.punc_list[punctuations[i]] != "_":
                    punc_res = self.punc_list[punctuations[i]]
                    if len(mini_sentence[i][0].encode()) == 1:
                        if punc_res == "，":
                            punc_res = ","
                        elif punc_res == "。":
                            punc_res = "."
                        elif punc_res == "？":
                            punc_res = "?"
                    words_with_punc.append(punc_res)
            new_mini_sentence += "".join(words_with_punc)
            # Add Period for the end of the sentence
            new_mini_sentence_out = new_mini_sentence
            new_mini_sentence_punc_out = new_mini_sentence_punc
            if mini_sentence_i == len(mini_sentences) - 1:
                if new_mini_sentence[-1] == "，" or new_mini_sentence[-1] == "、":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
                elif new_mini_sentence[-1] == ",":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "."
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
                elif new_mini_sentence[-1] != "。" and new_mini_sentence[-1] != "？" and len(new_mini_sentence[-1].encode())==0:
                    new_mini_sentence_out = new_mini_sentence + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
                elif new_mini_sentence[-1] != "." and new_mini_sentence[-1] != "?" and len(new_mini_sentence[-1].encode())==1:
                    new_mini_sentence_out = new_mini_sentence + "."
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
        return new_mini_sentence_out, new_mini_sentence_punc_out


class Text2PuncVADRealtime:

    def __init__(
            self,
            train_config: Optional[str],
            model_file: Optional[str],
            device: str = "cpu",
            dtype: str = "float32",
    ):
        #  Build Model
        model, train_args = build_model_from_file(train_config, model_file, None, device, task_name="punc")
        self.device = device
        # Wrape model to make model.nll() data-parallel
        self.wrapped_model = ForwardAdaptor(model, "inference")
        self.wrapped_model.to(dtype=getattr(torch, dtype)).to(device=device).eval()
        # logging.info(f"Model:\n{model}")
        self.punc_list = train_args.punc_list
        self.period = 0
        for i in range(len(self.punc_list)):
            if self.punc_list[i] == ",":
                self.punc_list[i] = "，"
            elif self.punc_list[i] == "?":
                self.punc_list[i] = "？"
            elif self.punc_list[i] == "。":
                self.period = i
        self.preprocessor = CodeMixTokenizerCommonPreprocessor(
            train=False,
            token_type=train_args.token_type,
            token_list=train_args.token_list,
            bpemodel=train_args.bpemodel,
            text_cleaner=train_args.cleaner,
            g2p_type=train_args.g2p,
            text_name="text",
            non_linguistic_symbols=train_args.non_linguistic_symbols,
        )

    @torch.no_grad()
    def __call__(self, text: Union[list, str], cache: list, split_size=20):
        if cache is not None and len(cache) > 0:
            precache = "".join(cache)
        else:
            precache = ""
            cache = []
        data = {"text": precache + " " + text}
        result = self.preprocessor(data=data, uid="12938712838719")
        split_text = self.preprocessor.pop_split_text_data(result)
        mini_sentences = split_to_mini_sentence(split_text, split_size)
        mini_sentences_id = split_to_mini_sentence(data["text"], split_size)
        assert len(mini_sentences) == len(mini_sentences_id)
        cache_sent = []
        cache_sent_id = torch.from_numpy(np.array([], dtype='int32'))
        sentence_punc_list = []
        sentence_words_list = []
        cache_pop_trigger_limit = 200
        skip_num = 0
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.concatenate((cache_sent_id, mini_sentence_id), axis=0)
            data = {
                "text": torch.unsqueeze(torch.from_numpy(mini_sentence_id), 0),
                "text_lengths": torch.from_numpy(np.array([len(mini_sentence_id)], dtype='int32')),
                "vad_indexes": torch.from_numpy(np.array([len(cache)], dtype='int32')),
            }
            data = to_device(data, self.device)
            y, _ = self.wrapped_model(**data)
            _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
            punctuations = indices
            if indices.size()[0] != 1:
                punctuations = torch.squeeze(indices)
            assert punctuations.size()[0] == len(mini_sentence)

            # Search for the last Period/QuestionMark as cache
            if mini_sentence_i < len(mini_sentences) - 1:
                sentenceEnd = -1
                last_comma_index = -1
                for i in range(len(punctuations) - 2, 1, -1):
                    if self.punc_list[punctuations[i]] == "。" or self.punc_list[punctuations[i]] == "？":
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if sentenceEnd < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.period
                cache_sent = mini_sentence[sentenceEnd + 1:]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1:]
                mini_sentence = mini_sentence[0:sentenceEnd + 1]
                punctuations = punctuations[0:sentenceEnd + 1]

            punctuations_np = punctuations.cpu().numpy()
            sentence_punc_list += [self.punc_list[int(x)] for x in punctuations_np]
            sentence_words_list += mini_sentence

        assert len(sentence_punc_list) == len(sentence_words_list)
        words_with_punc = []
        sentence_punc_list_out = []
        for i in range(0, len(sentence_words_list)):
            if i > 0:
                if len(sentence_words_list[i][0].encode()) == 1 and len(sentence_words_list[i - 1][-1].encode()) == 1:
                    sentence_words_list[i] = " " + sentence_words_list[i]
            if skip_num < len(cache):
                skip_num += 1
            else:
                words_with_punc.append(sentence_words_list[i])
            if skip_num >= len(cache):
                sentence_punc_list_out.append(sentence_punc_list[i])
                if sentence_punc_list[i] != "_":
                    words_with_punc.append(sentence_punc_list[i])
        sentence_out = "".join(words_with_punc)

        sentenceEnd = -1
        for i in range(len(sentence_punc_list) - 2, 1, -1):
            if sentence_punc_list[i] == "。" or sentence_punc_list[i] == "？":
                sentenceEnd = i
                break
        cache_out = sentence_words_list[sentenceEnd + 1:]
        if sentence_out[-1] in self.punc_list:
            sentence_out = sentence_out[:-1]
            sentence_punc_list_out[-1] = "_"
        return sentence_out, sentence_punc_list_out, cache_out
