# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os.path
from pathlib import Path
from typing import List, Union, Tuple

import copy
import librosa
import numpy as np

from .utils.utils import (CharTokenizer, Hypothesis, ONNXRuntimeError,
                          OrtInferSession, TokenIDConverter, get_logger,
                          read_yaml)
from .utils.postprocess_utils import sentence_postprocess
from .utils.frontend import WavFrontend
from .utils.timestamp_utils import time_stamp_lfr6_onnx

logging = get_logger()


class Paraformer():
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """
    def __init__(self, model_dir: Union[str, Path] = None,
                 batch_size: int = 1,
                 device_id: Union[str, int] = "-1",
                 plot_timestamp_to: str = "",
                 quantize: bool = False,
                 intra_op_num_threads: int = 4,
                 ):

        if not Path(model_dir).exists():
            raise FileNotFoundError(f'{model_dir} does not exist.')

        model_file = os.path.join(model_dir, 'model.onnx')
        if quantize:
            model_file = os.path.join(model_dir, 'model_quant.onnx')
        config_file = os.path.join(model_dir, 'config.yaml')
        cmvn_file = os.path.join(model_dir, 'am.mvn')
        config = read_yaml(config_file)

        self.converter = TokenIDConverter(config['token_list'])
        self.tokenizer = CharTokenizer()
        self.frontend = WavFrontend(
            cmvn_file=cmvn_file,
            **config['frontend_conf']
        )
        self.ort_infer = OrtInferSession(model_file, device_id, intra_op_num_threads=intra_op_num_threads)
        self.batch_size = batch_size
        self.plot_timestamp_to = plot_timestamp_to
        if "predictor_bias" in config['model_conf'].keys():
            self.pred_bias = config['model_conf']['predictor_bias']
        else:
            self.pred_bias = 0

    def __call__(self, wav_content: Union[str, np.ndarray, List[str]], **kwargs) -> List:
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        asr_res = []
        for beg_idx in range(0, waveform_nums, self.batch_size):
            
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            try:
                outputs = self.infer(feats, feats_len)
                am_scores, valid_token_lens = outputs[0], outputs[1]
                if len(outputs) == 4:
                    # for BiCifParaformer Inference
                    us_alphas, us_peaks = outputs[2], outputs[3]
                else:
                    us_alphas, us_peaks = None, None
            except ONNXRuntimeError:
                #logging.warning(traceback.format_exc())
                logging.warning("input wav is silence or noise")
                preds = ['']
            else:
                preds = self.decode(am_scores, valid_token_lens)
                if us_peaks is None:
                    for pred in preds:
                        pred = sentence_postprocess(pred)
                        asr_res.append({'preds': pred})
                else:
                    for pred, us_peaks_ in zip(preds, us_peaks):
                        raw_tokens = pred
                        timestamp, timestamp_raw = time_stamp_lfr6_onnx(us_peaks_, copy.copy(raw_tokens))
                        text_proc, timestamp_proc, _ = sentence_postprocess(raw_tokens, timestamp_raw)
                        # logging.warning(timestamp)
                        if len(self.plot_timestamp_to):
                            self.plot_wave_timestamp(waveform_list[0], timestamp, self.plot_timestamp_to)
                        asr_res.append({'preds': text_proc, 'timestamp': timestamp_proc, "raw_tokens": raw_tokens})
        return asr_res

    def plot_wave_timestamp(self, wav, text_timestamp, dest):
        # TODO: Plot the wav and timestamp results with matplotlib
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rc("font", family='Alibaba PuHuiTi')  # set it to a font that your system supports
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(11, 3.5), dpi=320)
        ax2 = ax1.twinx()
        ax2.set_ylim([0, 2.0])
        # plot waveform
        ax1.set_ylim([-0.3, 0.3])
        time = np.arange(wav.shape[0]) / 16000
        ax1.plot(time, wav/wav.max()*0.3, color='gray', alpha=0.4)
        # plot lines and text
        for (char, start, end) in text_timestamp:
            ax1.vlines(start, -0.3, 0.3, ls='--')
            ax1.vlines(end, -0.3, 0.3, ls='--')
            x_adj = 0.045 if char != '<sil>' else 0.12
            ax1.text((start + end) * 0.5 - x_adj, 0, char)
        # plt.legend()
        plotname = "{}/timestamp.png".format(dest)
        plt.savefig(plotname, bbox_inches='tight')

    def load_data(self,
                  wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(
            f'The type of {wav_content} is not in [str, np.ndarray, list]')

    def extract_feat(self,
                     waveform_list: List[np.ndarray]
                     ) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, 'constant', constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, feats: np.ndarray,
              feats_len: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer([feats, feats_len])
        return outputs

    def decode(self, am_scores: np.ndarray, token_nums: int) -> List[str]:
        return [self.decode_one(am_score, token_num)
                for am_score, token_num in zip(am_scores, token_nums)]

    def decode_one(self,
                   am_score: np.ndarray,
                   valid_token_num: int) -> List[str]:
        yseq = am_score.argmax(axis=-1)
        score = am_score.max(axis=-1)
        score = np.sum(score, axis=-1)

        # pad with mask tokens to ensure compatibility with sos/eos tokens
        # asr_model.sos:1  asr_model.eos:2
        yseq = np.array([1] + yseq.tolist() + [2])
        hyp = Hypothesis(yseq=yseq, score=score)

        # remove sos/eos and get results
        last_pos = -1
        token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x not in (0, 2), token_int))

        # Change integer-ids to tokens
        token = self.converter.ids2tokens(token_int)
        token = token[:valid_token_num-self.pred_bias]
        # texts = sentence_postprocess(token)
        return token

