#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import torch
import numpy as np
from contextlib import contextmanager
from distutils.version import LooseVersion

from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
from funasr.models.ct_transformer.model import CTTransformer
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.models.ct_transformer.utils import split_to_mini_sentence, split_words


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "CTTransformerStreaming")
class CTTransformerStreaming(CTTransformer):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection
    https://arxiv.org/pdf/2003.01309.pdf
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def punc_forward(
        self, text: torch.Tensor, text_lengths: torch.Tensor, vad_indexes: torch.Tensor, **kwargs
    ):
        """Compute loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(text)
        # mask = self._target_mask(input)
        h, _, _ = self.encoder(x, text_lengths, vad_indexes=vad_indexes)
        y = self.decoder(h)
        return y, None

    def with_vad(self):
        return True

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        cache: dict = {},
        **kwargs,
    ):
        assert len(data_in) == 1

        if len(cache) == 0:
            cache["pre_text"] = []
        text = load_audio_text_image_video(data_in, data_type=kwargs.get("kwargs", "text"))[0]
        text = "".join(cache["pre_text"]) + " " + text

        split_size = kwargs.get("split_size", 20)

        tokens = split_words(text)
        tokens_int = tokenizer.encode(tokens)

        mini_sentences = split_to_mini_sentence(tokens, split_size)
        mini_sentences_id = split_to_mini_sentence(tokens_int, split_size)
        assert len(mini_sentences) == len(mini_sentences_id)
        cache_sent = []
        cache_sent_id = torch.from_numpy(np.array([], dtype="int32"))
        skip_num = 0
        sentence_punc_list = []
        sentence_words_list = []
        cache_pop_trigger_limit = 200
        results = []
        meta_data = {}
        punc_array = None
        for mini_sentence_i in range(len(mini_sentences)):
            mini_sentence = mini_sentences[mini_sentence_i]
            mini_sentence_id = mini_sentences_id[mini_sentence_i]
            mini_sentence = cache_sent + mini_sentence
            mini_sentence_id = np.concatenate((cache_sent_id, mini_sentence_id), axis=0)
            data = {
                "text": torch.unsqueeze(torch.from_numpy(mini_sentence_id), 0),
                "text_lengths": torch.from_numpy(np.array([len(mini_sentence_id)], dtype="int32")),
                "vad_indexes": torch.from_numpy(np.array([len(cache["pre_text"])], dtype="int32")),
            }
            data = to_device(data, kwargs["device"])
            # y, _ = self.wrapped_model(**data)
            y, _ = self.punc_forward(**data)
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
                    if (
                        self.punc_list[punctuations[i]] == "。"
                        or self.punc_list[punctuations[i]] == "？"
                    ):
                        sentenceEnd = i
                        break
                    if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                        last_comma_index = i

                if (
                    sentenceEnd < 0
                    and len(mini_sentence) > cache_pop_trigger_limit
                    and last_comma_index >= 0
                ):
                    # The sentence it too long, cut off at a comma.
                    sentenceEnd = last_comma_index
                    punctuations[sentenceEnd] = self.sentence_end_id
                cache_sent = mini_sentence[sentenceEnd + 1 :]
                cache_sent_id = mini_sentence_id[sentenceEnd + 1 :]
                mini_sentence = mini_sentence[0 : sentenceEnd + 1]
                punctuations = punctuations[0 : sentenceEnd + 1]

            # if len(punctuations) == 0:
            #    continue

            punctuations_np = punctuations.cpu().numpy()
            sentence_punc_list += [self.punc_list[int(x)] for x in punctuations_np]
            sentence_words_list += mini_sentence

        assert len(sentence_punc_list) == len(sentence_words_list)
        words_with_punc = []
        sentence_punc_list_out = []
        for i in range(0, len(sentence_words_list)):
            if i > 0:
                if (
                    len(sentence_words_list[i][0].encode()) == 1
                    and len(sentence_words_list[i - 1][-1].encode()) == 1
                ):
                    sentence_words_list[i] = " " + sentence_words_list[i]
            if skip_num < len(cache["pre_text"]):
                skip_num += 1
            else:
                words_with_punc.append(sentence_words_list[i])
            if skip_num >= len(cache["pre_text"]):
                sentence_punc_list_out.append(sentence_punc_list[i])
                if sentence_punc_list[i] != "_":
                    words_with_punc.append(sentence_punc_list[i])
        sentence_out = "".join(words_with_punc)

        sentenceEnd = -1
        for i in range(len(sentence_punc_list) - 2, 1, -1):
            if sentence_punc_list[i] == "。" or sentence_punc_list[i] == "？":
                sentenceEnd = i
                break
        cache["pre_text"] = sentence_words_list[sentenceEnd + 1 :]
        if sentence_out[-1] in self.punc_list:
            sentence_out = sentence_out[:-1]
            sentence_punc_list_out[-1] = "_"
        # keep a punctuations array for punc segment
        if punc_array is None:
            punc_array = punctuations
        else:
            punc_array = torch.cat([punc_array, punctuations], dim=0)

        result_i = {"key": key[0], "text": sentence_out, "punc_array": punc_array}
        results.append(result_i)

        return results, meta_data

    def export(self, **kwargs):

        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models
