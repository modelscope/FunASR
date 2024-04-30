#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import copy
import torch
import numpy as np
import torch.nn.functional as F
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Any, List, Tuple, Optional

from funasr.register import tables
from funasr.train_utils.device_funcs import to_device
from funasr.train_utils.device_funcs import force_gatherable
from funasr.utils.load_utils import load_audio_text_image_video
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.ct_transformer.utils import split_to_mini_sentence, split_words

try:
    import jieba
except:
    pass
if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "CTTransformer")
class CTTransformer(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection
    https://arxiv.org/pdf/2003.01309.pdf
    """

    def __init__(
        self,
        encoder: str = None,
        encoder_conf: dict = None,
        vocab_size: int = -1,
        punc_list: list = None,
        punc_weight: list = None,
        embed_unit: int = 128,
        att_unit: int = 256,
        dropout_rate: float = 0.5,
        ignore_id: int = -1,
        sos: int = 1,
        eos: int = 2,
        sentence_end_id: int = 3,
        **kwargs,
    ):
        super().__init__()

        punc_size = len(punc_list)
        if punc_weight is None:
            punc_weight = [1] * punc_size

        self.embed = torch.nn.Embedding(vocab_size, embed_unit)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(**encoder_conf)

        self.decoder = torch.nn.Linear(att_unit, punc_size)
        self.encoder = encoder
        self.punc_list = punc_list
        self.punc_weight = punc_weight
        self.ignore_id = ignore_id
        self.sos = sos
        self.eos = eos
        self.sentence_end_id = sentence_end_id
        self.jieba_usr_dict = None
        if kwargs.get("jieba_usr_dict", None) is not None:
            jieba.load_userdict(kwargs["jieba_usr_dict"])
            self.jieba_usr_dict = jieba

    def punc_forward(self, text: torch.Tensor, text_lengths: torch.Tensor, **kwargs):
        """Compute loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(text)
        # mask = self._target_mask(input)
        h, _, _ = self.encoder(x, text_lengths)
        y = self.decoder(h)
        return y, None

    def with_vad(self):
        return False

    def score(self, y: torch.Tensor, state: Any, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (vocab_size)
                and next state for ys

        """
        y = y.unsqueeze(0)
        h, _, cache = self.encoder.forward_one_step(
            self.embed(y), self._target_mask(y), cache=state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, vocab_size)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)]) for i in range(n_layers)
            ]

        # batch decoding
        h, _, states = self.encoder.forward_one_step(
            self.embed(ys), self._target_mask(ys), cache=batch_state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def nll(
        self,
        text: torch.Tensor,
        punc: torch.Tensor,
        text_lengths: torch.Tensor,
        punc_lengths: torch.Tensor,
        max_length: Optional[int] = None,
        vad_indexes: Optional[torch.Tensor] = None,
        vad_indexes_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length)
            punc: (Batch, Length)
            text_lengths: (Batch,)
            max_lengths: int
        """
        batch_size = text.size(0)
        # For data parallel
        if max_length is None:
            text = text[:, : text_lengths.max()]
            punc = punc[:, : text_lengths.max()]
        else:
            text = text[:, :max_length]
            punc = punc[:, :max_length]

        if self.with_vad():
            # Should be VadRealtimeTransformer
            assert vad_indexes is not None
            y, _ = self.punc_forward(text, text_lengths, vad_indexes)
        else:
            # Should be TargetDelayTransformer,
            y, _ = self.punc_forward(text, text_lengths)

        # Calc negative log likelihood
        # nll: (BxL,)
        if self.training == False:
            _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
            from sklearn.metrics import f1_score

            f1_score = f1_score(
                punc.view(-1).detach().cpu().numpy(),
                indices.squeeze(-1).detach().cpu().numpy(),
                average="micro",
            )
            nll = torch.Tensor([f1_score]).repeat(text_lengths.sum())
            return nll, text_lengths
        else:
            self.punc_weight = self.punc_weight.to(punc.device)
            nll = F.cross_entropy(
                y.view(-1, y.shape[-1]),
                punc.view(-1),
                self.punc_weight,
                reduction="none",
                ignore_index=self.ignore_id,
            )
        # nll: (BxL,) -> (BxL,)
        if max_length is None:
            nll.masked_fill_(make_pad_mask(text_lengths).to(nll.device).view(-1), 0.0)
        else:
            nll.masked_fill_(
                make_pad_mask(text_lengths, maxlen=max_length + 1).to(nll.device).view(-1),
                0.0,
            )
        # nll: (BxL,) -> (B, L)
        nll = nll.view(batch_size, -1)
        return nll, text_lengths

    def forward(
        self,
        text: torch.Tensor,
        punc: torch.Tensor,
        text_lengths: torch.Tensor,
        punc_lengths: torch.Tensor,
        vad_indexes: Optional[torch.Tensor] = None,
        vad_indexes_lengths: Optional[torch.Tensor] = None,
    ):
        nll, y_lengths = self.nll(text, punc, text_lengths, punc_lengths, vad_indexes=vad_indexes)
        ntokens = y_lengths.sum()
        loss = nll.sum() / ntokens
        stats = dict(loss=loss.detach())

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, ntokens), loss.device)
        return loss, stats, weight

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        assert len(data_in) == 1
        text = load_audio_text_image_video(data_in, data_type=kwargs.get("kwargs", "text"))[0]
        vad_indexes = kwargs.get("vad_indexes", None)
        # text = data_in[0]
        # text_lengths = data_lengths[0] if data_lengths is not None else None
        split_size = kwargs.get("split_size", 20)

        tokens = split_words(text, jieba_usr_dict=self.jieba_usr_dict)
        tokens_int = tokenizer.encode(tokens)

        mini_sentences = split_to_mini_sentence(tokens, split_size)
        mini_sentences_id = split_to_mini_sentence(tokens_int, split_size)
        assert len(mini_sentences) == len(mini_sentences_id)
        cache_sent = []
        cache_sent_id = torch.from_numpy(np.array([], dtype="int32"))
        new_mini_sentence = ""
        new_mini_sentence_punc = []
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
            }
            data = to_device(data, kwargs["device"])
            # y, _ = self.wrapped_model(**data)
            y, _ = self.punc_forward(**data)
            _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
            punctuations = torch.squeeze(indices, dim=1)
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
            new_mini_sentence_punc += [int(x) for x in punctuations_np]
            words_with_punc = []
            for i in range(len(mini_sentence)):
                if (
                    i == 0
                    or self.punc_list[punctuations[i - 1]] == "。"
                    or self.punc_list[punctuations[i - 1]] == "？"
                ) and len(mini_sentence[i][0].encode()) == 1:
                    mini_sentence[i] = mini_sentence[i].capitalize()
                if i == 0:
                    if len(mini_sentence[i][0].encode()) == 1:
                        mini_sentence[i] = " " + mini_sentence[i]
                if i > 0:
                    if (
                        len(mini_sentence[i][0].encode()) == 1
                        and len(mini_sentence[i - 1][0].encode()) == 1
                    ):
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
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [
                        self.sentence_end_id
                    ]
                elif new_mini_sentence[-1] == ",":
                    new_mini_sentence_out = new_mini_sentence[:-1] + "."
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [
                        self.sentence_end_id
                    ]
                elif (
                    new_mini_sentence[-1] != "。"
                    and new_mini_sentence[-1] != "？"
                    and len(new_mini_sentence[-1].encode()) != 1
                ):
                    new_mini_sentence_out = new_mini_sentence + "。"
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [
                        self.sentence_end_id
                    ]
                    if len(punctuations):
                        punctuations[-1] = 2
                elif (
                    new_mini_sentence[-1] != "."
                    and new_mini_sentence[-1] != "?"
                    and len(new_mini_sentence[-1].encode()) == 1
                ):
                    new_mini_sentence_out = new_mini_sentence + "."
                    new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [
                        self.sentence_end_id
                    ]
                    if len(punctuations):
                        punctuations[-1] = 2
            # keep a punctuations array for punc segment
            if punc_array is None:
                punc_array = punctuations
            else:
                punc_array = torch.cat([punc_array, punctuations], dim=0)

        # post processing when using word level punc model
        if self.jieba_usr_dict is not None:
            punc_array = punc_array.reshape(-1)
            len_tokens = len(tokens)
            new_punc_array = copy.copy(punc_array).tolist()
            # for i, (token, punc_id) in enumerate(zip(tokens[::-1], punc_array.tolist()[::-1])):
            for i, token in enumerate(tokens[::-1]):
                if "\u0e00" <= token[0] <= "\u9fa5":  # ignore en words
                    if len(token) > 1:
                        num_append = len(token) - 1
                        ind_append = len_tokens - i - 1
                        for _ in range(num_append):
                            new_punc_array.insert(ind_append, 1)
            punc_array = torch.tensor(new_punc_array)

        result_i = {"key": key[0], "text": new_mini_sentence_out, "punc_array": punc_array}
        results.append(result_i)
        return results, meta_data

    def export(self, **kwargs):

        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models
