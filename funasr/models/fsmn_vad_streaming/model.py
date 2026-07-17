#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import json
import time
import math
import torch
import numpy as np
from torch import nn
from enum import Enum
from dataclasses import dataclass
from funasr.register import tables
from typing import List, Tuple, Dict, Any, Optional

from funasr.utils.datadir_writer import DatadirWriter
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

# Dynamic silence threshold schedule: (accumulated_speech_ms, silence_threshold_ms)
# Streaming mode: longer initial patience (waiting for user to continue)
STREAMING_SILENCE_SCHEDULE = [
    (5000, 2000),
    (10000, 1500),
    (15000, 1000),
    (30000, 800),
    (45000, 400),
    (float("inf"), 100),
]

# Non-streaming mode: shorter patience (audio already complete, cut by natural pauses)
DEFAULT_SILENCE_SCHEDULE = [
    (10000, 2000),
    (20000, 1000),
    (30000, 800),
    (40000, 600),
    (50000, 400),
    (60000, 200),
    (float("inf"), 100),
]


class VadStateMachine(Enum):
    kVadInStateStartPointNotDetected = 1
    kVadInStateInSpeechSegment = 2
    kVadInStateEndPointDetected = 3


class FrameState(Enum):
    kFrameStateInvalid = -1
    kFrameStateSpeech = 1
    kFrameStateSil = 0


# final voice/unvoice state per frame
class AudioChangeState(Enum):
    kChangeStateSpeech2Speech = 0
    kChangeStateSpeech2Sil = 1
    kChangeStateSil2Sil = 2
    kChangeStateSil2Speech = 3
    kChangeStateNoBegin = 4
    kChangeStateInvalid = 5


class VadDetectMode(Enum):
    kVadSingleUtteranceDetectMode = 0
    kVadMutipleUtteranceDetectMode = 1


class VADXOptions:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        detect_mode: int = VadDetectMode.kVadMutipleUtteranceDetectMode.value,
        snr_mode: int = 0,
        max_end_silence_time: int = 800,
        max_start_silence_time: int = 3000,
        do_start_point_detection: bool = True,
        do_end_point_detection: bool = True,
        window_size_ms: int = 200,
        sil_to_speech_time_thres: int = 150,
        speech_to_sil_time_thres: int = 150,
        speech_2_noise_ratio: float = 1.0,
        do_extend: int = 1,
        lookback_time_start_point: int = 200,
        lookahead_time_end_point: int = 100,
        max_single_segment_time: int = 60000,
        nn_eval_block_size: int = 8,
        dcd_block_size: int = 4,
        snr_thres: int = -100.0,
        noise_frame_num_used_for_snr: int = 100,
        decibel_thres: int = -100.0,
        speech_noise_thres: float = 0.6,
        fe_prior_thres: float = 1e-4,
        silence_pdf_num: int = 1,
        sil_pdf_ids: List[int] = [0],
        speech_noise_thresh_low: float = -0.1,
        speech_noise_thresh_high: float = 0.3,
        output_frame_probs: bool = False,
        frame_in_ms: int = 10,
        frame_length_ms: int = 25,
        **kwargs,
    ):
        """Initialize VADXOptions.
        
            Args:
                sample_rate: TODO.
                detect_mode: TODO.
                snr_mode: TODO.
                max_end_silence_time: TODO.
                max_start_silence_time: TODO.
                do_start_point_detection: TODO.
                do_end_point_detection: TODO.
                window_size_ms: TODO.
                sil_to_speech_time_thres: TODO.
                speech_to_sil_time_thres: TODO.
                speech_2_noise_ratio: TODO.
                do_extend: TODO.
                lookback_time_start_point: TODO.
                lookahead_time_end_point: TODO.
                max_single_segment_time: TODO.
                nn_eval_block_size: Size/dimension parameter.
                dcd_block_size: Size/dimension parameter.
                snr_thres: TODO.
                noise_frame_num_used_for_snr: TODO.
                decibel_thres: TODO.
                speech_noise_thres: TODO.
                fe_prior_thres: TODO.
                silence_pdf_num: TODO.
                sil_pdf_ids: TODO.
                speech_noise_thresh_low: TODO.
                speech_noise_thresh_high: TODO.
                output_frame_probs: TODO.
                frame_in_ms: TODO.
                frame_length_ms: TODO.
                **kwargs: Additional keyword arguments.
            """
        self.sample_rate = sample_rate
        self.detect_mode = detect_mode
        self.snr_mode = snr_mode
        self.max_end_silence_time = max_end_silence_time
        self.max_start_silence_time = max_start_silence_time
        self.do_start_point_detection = do_start_point_detection
        self.do_end_point_detection = do_end_point_detection
        self.window_size_ms = window_size_ms
        self.sil_to_speech_time_thres = sil_to_speech_time_thres
        self.speech_to_sil_time_thres = speech_to_sil_time_thres
        self.speech_2_noise_ratio = speech_2_noise_ratio
        self.do_extend = do_extend
        self.lookback_time_start_point = lookback_time_start_point
        self.lookahead_time_end_point = lookahead_time_end_point
        self.max_single_segment_time = max_single_segment_time
        self.nn_eval_block_size = nn_eval_block_size
        self.dcd_block_size = dcd_block_size
        self.snr_thres = snr_thres
        self.noise_frame_num_used_for_snr = noise_frame_num_used_for_snr
        self.decibel_thres = decibel_thres
        self.speech_noise_thres = speech_noise_thres
        self.fe_prior_thres = fe_prior_thres
        self.silence_pdf_num = silence_pdf_num
        self.sil_pdf_ids = sil_pdf_ids
        self.speech_noise_thresh_low = speech_noise_thresh_low
        self.speech_noise_thresh_high = speech_noise_thresh_high
        self.output_frame_probs = output_frame_probs
        self.frame_in_ms = frame_in_ms
        self.frame_length_ms = frame_length_ms


class E2EVadSpeechBufWithDoa(object):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(self):
        """Initialize E2EVadSpeechBufWithDoa."""
        self.start_ms = 0
        self.end_ms = 0
        self.buffer = []
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0

    def Reset(self):
        """Reset."""
        self.start_ms = 0
        self.end_ms = 0
        self.buffer = []
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0


class E2EVadFrameProb(object):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(self):
        """Initialize E2EVadFrameProb."""
        self.noise_prob = 0.0
        self.speech_prob = 0.0
        self.score = 0.0
        self.frame_id = 0
        self.frm_state = 0


class WindowDetector(object):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        window_size_ms: int,
        sil_to_speech_time: int,
        speech_to_sil_time: int,
        frame_size_ms: int,
    ):
        """Initialize WindowDetector.
        
            Args:
                window_size_ms: TODO.
                sil_to_speech_time: TODO.
                speech_to_sil_time: TODO.
                frame_size_ms: TODO.
            """
        self.window_size_ms = window_size_ms
        self.sil_to_speech_time = sil_to_speech_time
        self.speech_to_sil_time = speech_to_sil_time
        self.frame_size_ms = frame_size_ms

        self.win_size_frame = int(window_size_ms / frame_size_ms)
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame  # 初始化窗

        self.cur_win_pos = 0
        self.pre_frame_state = FrameState.kFrameStateSil
        self.cur_frame_state = FrameState.kFrameStateSil
        self.sil_to_speech_frmcnt_thres = int(sil_to_speech_time / frame_size_ms)
        self.speech_to_sil_frmcnt_thres = int(speech_to_sil_time / frame_size_ms)

        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def Reset(self) -> None:
        """Reset."""
        self.cur_win_pos = 0
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.pre_frame_state = FrameState.kFrameStateSil
        self.cur_frame_state = FrameState.kFrameStateSil
        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def GetWinSize(self) -> int:
        """Getwinsize."""
        return int(self.win_size_frame)

    def DetectOneFrame(
        self, frameState: FrameState, frame_count: int, cache: dict = None
    ) -> AudioChangeState:
        """Detectoneframe.
        
            Args:
                frameState: TODO.
                frame_count: TODO.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        cur_frame_state = FrameState.kFrameStateSil
        if frameState == FrameState.kFrameStateSpeech:
            cur_frame_state = 1
        elif frameState == FrameState.kFrameStateSil:
            cur_frame_state = 0
        else:
            return AudioChangeState.kChangeStateInvalid
        self.win_sum -= self.win_state[self.cur_win_pos]
        self.win_sum += cur_frame_state
        self.win_state[self.cur_win_pos] = cur_frame_state
        self.cur_win_pos = (self.cur_win_pos + 1) % self.win_size_frame

        if (
            self.pre_frame_state == FrameState.kFrameStateSil
            and self.win_sum >= self.sil_to_speech_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.kFrameStateSpeech
            return AudioChangeState.kChangeStateSil2Speech

        if (
            self.pre_frame_state == FrameState.kFrameStateSpeech
            and self.win_sum <= self.speech_to_sil_frmcnt_thres
        ):
            self.pre_frame_state = FrameState.kFrameStateSil
            return AudioChangeState.kChangeStateSpeech2Sil

        if self.pre_frame_state == FrameState.kFrameStateSil:
            return AudioChangeState.kChangeStateSil2Sil
        if self.pre_frame_state == FrameState.kFrameStateSpeech:
            return AudioChangeState.kChangeStateSpeech2Speech
        return AudioChangeState.kChangeStateInvalid

    def FrameSizeMs(self) -> int:
        """Framesizems."""
        return int(self.frame_size_ms)


class Stats(object):
    def __init__(
        self,
        sil_pdf_ids,
        max_end_sil_frame_cnt_thresh,
        speech_noise_thres,
    ):
        """Initialize Stats.
        
            Args:
                sil_pdf_ids: TODO.
                max_end_sil_frame_cnt_thresh: TODO.
                speech_noise_thres: TODO.
            """
        self.data_buf_start_frame = 0
        self.frm_cnt = 0
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.continous_silence_frame_count = 0
        self.vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.number_end_time_detected = 0
        self.sil_frame = 0
        self.sil_pdf_ids = sil_pdf_ids
        self.noise_average_decibel = -100.0
        self.pre_end_silence_detected = False
        self.next_seg = True

        self.output_data_buf = []
        self.output_data_buf_offset = 0
        self.frame_probs = []
        self.max_end_sil_frame_cnt_thresh = max_end_sil_frame_cnt_thresh
        self.speech_noise_thres = speech_noise_thres
        self.scores = None
        self.max_time_out = False
        self.decibel = []
        self.data_buf = None
        self.data_buf_all = None
        self.waveform = None
        self.last_drop_frames = 0


@tables.register("model_classes", "FsmnVADStreaming")
class FsmnVADStreaming(nn.Module):
    """FSMN-based Voice Activity Detection (streaming/offline).

    Detects speech segments in audio, returning start/end timestamps (milliseconds).
    Supports both offline (full audio) and streaming (chunk-by-chunk) modes.

    Offline output: [{"key": "...", "value": [[start_ms, end_ms], ...]}]
    Streaming output: [[beg, -1]] (start), [[-1, end]] (end), [[beg, end]] (complete), [] (no event)

    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(
        self,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        vad_post_args: Dict[str, Any] = None,
        **kwargs,
    ):
        """Initialize FsmnVADStreaming.
        
            Args:
                encoder: TODO.
                encoder_conf: Configuration dict for encoder.
                vad_post_args: TODO.
                **kwargs: Additional keyword arguments.
            """
        super().__init__()
        self.vad_opts = VADXOptions(**kwargs)

        encoder_class = tables.encoder_classes.get(encoder)
        if encoder_conf is None:
            encoder_conf = {}
        encoder = encoder_class(**encoder_conf)
        self.encoder = encoder
        self.encoder_conf = encoder_conf

    def DropCachedFrames(self, drop_frames: int, cache: dict = None) -> None:
        """Release score, decibel, and waveform history before an absolute frame."""
        if cache is None:
            cache = {}
        stats = cache["stats"]
        if drop_frames > stats.frm_cnt:
            raise RuntimeError(
                f"Cannot drop through frame {drop_frames}; only {stats.frm_cnt} frames exist"
            )

        real_drop_frames = drop_frames - stats.last_drop_frames
        if real_drop_frames <= 0:
            return

        frame_shift_length = int(
            self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
        )
        drop_samples = real_drop_frames * frame_shift_length
        stats.data_buf_all = stats.data_buf_all[drop_samples:].clone()
        stats.decibel = stats.decibel[real_drop_frames:]
        stats.scores = stats.scores[:, real_drop_frames:, :].clone()
        stats.last_drop_frames = drop_frames

        data_buf_offset = max(
            stats.data_buf_start_frame - stats.last_drop_frames,
            0,
        ) * frame_shift_length
        stats.data_buf = stats.data_buf_all[data_buf_offset:]

    def ResetDetection(self, cache: dict = None):
        """Resetdetection.
        
            Args:
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        cache["stats"].continous_silence_frame_count = 0
        cache["stats"].latest_confirmed_speech_frame = 0
        cache["stats"].lastest_confirmed_silence_frame = -1
        cache["stats"].confirmed_start_frame = -1
        cache["stats"].confirmed_end_frame = -1
        cache["stats"].vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        cache["windows_detector"].Reset()
        cache["stats"].sil_frame = 0
        cache["stats"].frame_probs = []

        if cache["stats"].output_data_buf:
            assert cache["stats"].output_data_buf[-1].contain_seg_end_point == True
            drop_frames = int(cache["stats"].output_data_buf[-1].end_ms / self.vad_opts.frame_in_ms)
            self.DropCachedFrames(drop_frames, cache=cache)

    def ComputeDecibel(
        self, cache: dict = None, frame_count: Optional[int] = None
    ) -> None:
        """Computedecibel.
        
            Args:
                frame_count: Number of VAD score frames emitted for this waveform.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        frame_sample_length = int(self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000)
        frame_shift_length = int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000)
        if frame_count is not None and int(frame_count) <= 0:
            return
        stats = cache["stats"]
        waveform = stats.waveform[0]

        frame_count_was_provided = frame_count is not None
        available_frames = max(
            (waveform.numel() - frame_sample_length) // frame_shift_length + 1,
            0,
        )
        frame_count = available_frames if frame_count is None else int(frame_count)
        if frame_count <= 0:
            return

        # The frontend provides the exact waveform span for these VAD score frames.
        aligned_sample_count = (
            (frame_count - 1) * frame_shift_length + frame_sample_length
        )
        if waveform.numel() != aligned_sample_count:
            if frame_count_was_provided or waveform.numel() < aligned_sample_count:
                raise RuntimeError(
                    "VAD score frames and waveform samples are not aligned: "
                    f"expected {aligned_sample_count}, got {waveform.numel()}"
                )
            waveform = waveform[:aligned_sample_count]

        aligned_waveform = waveform

        if stats.data_buf_all is None:
            stats.data_buf_all = aligned_waveform.clone()
        else:
            new_sample_count = frame_count * frame_shift_length
            stats.data_buf_all = torch.cat(
                (stats.data_buf_all, aligned_waveform[-new_sample_count:])
            )

        data_buf_offset = max(
            stats.data_buf_start_frame - stats.last_drop_frames,
            0,
        ) * frame_shift_length
        stats.data_buf = stats.data_buf_all[data_buf_offset:]

        waveform_numpy = aligned_waveform.detach().cpu().numpy()

        offsets = np.arange(
            0,
            waveform_numpy.shape[0] - frame_sample_length + 1,
            frame_shift_length,
        )
        frames = waveform_numpy[
            offsets[:, np.newaxis] + np.arange(frame_sample_length)
        ]

        decibel_numpy = 10 * np.log10(
            np.sum(np.square(frames), axis=1) + 0.000001
        )
        decibel_numpy = decibel_numpy.tolist()

        stats.decibel.extend(decibel_numpy)


    def ComputeScores(self, feats: torch.Tensor, cache: dict = None) -> None:
        """Computescores.
        
            Args:
                feats: Feature tensor (e.g., fbank), shape (batch, frames, dim).
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        scores = self.encoder(feats, cache=cache["encoder"])  # return B * T * D
        assert (
            scores.shape[1] == feats.shape[1]
        ), "The shape between feats and scores does not match"
        self.vad_opts.nn_eval_block_size = scores.shape[1]
        cache["stats"].frm_cnt += scores.shape[1]  # count total frames
        if cache["stats"].scores is None:
            cache["stats"].scores = scores  # the first calculation
        else:
            cache["stats"].scores = torch.cat((cache["stats"].scores, scores), dim=1)

    def PopDataBufTillFrame(self, frame_idx: int, cache: dict = None) -> None:  # need check again
        """Popdatabuftillframe.
        
            Args:
                frame_idx: TODO.
                cache: State cache dict for streaming inference.
            """
        while cache["stats"].data_buf_start_frame < frame_idx:
            if len(cache["stats"].data_buf) >= int(
                self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            ):
                if cache is None:
                    cache = {}
                cache["stats"].data_buf_start_frame += 1
                cache["stats"].data_buf = cache["stats"].data_buf_all[
                    (cache["stats"].data_buf_start_frame - cache["stats"].last_drop_frames)
                    * int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000) :
                ]

    def PopDataToOutputBuf(
        self,
        start_frm: int,
        frm_cnt: int,
        first_frm_is_start_point: bool,
        last_frm_is_end_point: bool,
        end_point_is_sent_end: bool,
        cache: dict = None,
    ) -> None:
        """Popdatatooutputbuf.
        
            Args:
                start_frm: TODO.
                frm_cnt: TODO.
                first_frm_is_start_point: TODO.
                last_frm_is_end_point: TODO.
                end_point_is_sent_end: TODO.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        self.PopDataBufTillFrame(start_frm, cache=cache)
        expected_sample_number = int(
            frm_cnt * self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
        )
        if last_frm_is_end_point:
            extra_sample = max(
                0,
                int(
                    self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000
                    - self.vad_opts.sample_rate * self.vad_opts.frame_in_ms / 1000
                ),
            )
            expected_sample_number += int(extra_sample)
        if end_point_is_sent_end:
            expected_sample_number = max(expected_sample_number, len(cache["stats"].data_buf))
        if len(cache["stats"].data_buf) < expected_sample_number:
            print("error in calling pop data_buf\n")

        if len(cache["stats"].output_data_buf) == 0 or first_frm_is_start_point:
            cache["stats"].output_data_buf.append(E2EVadSpeechBufWithDoa())
            cache["stats"].output_data_buf[-1].Reset()
            cache["stats"].output_data_buf[-1].start_ms = start_frm * self.vad_opts.frame_in_ms
            cache["stats"].output_data_buf[-1].end_ms = cache["stats"].output_data_buf[-1].start_ms
            cache["stats"].output_data_buf[-1].doa = 0
        cur_seg = cache["stats"].output_data_buf[-1]
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            print("warning\n")
        data_to_pop = 0
        if end_point_is_sent_end:
            data_to_pop = expected_sample_number
        else:
            data_to_pop = int(
                frm_cnt * self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            )
        if data_to_pop > len(cache["stats"].data_buf):
            print('VAD data_to_pop is bigger than cache["stats"].data_buf.size()!!!\n')
            data_to_pop = len(cache["stats"].data_buf)
            expected_sample_number = len(cache["stats"].data_buf)

        cur_seg.doa = 0
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            print("Something wrong with the VAD algorithm\n")
        cache["stats"].data_buf_start_frame += frm_cnt
        cur_seg.end_ms = (start_frm + frm_cnt) * self.vad_opts.frame_in_ms
        if first_frm_is_start_point:
            cur_seg.contain_seg_start_point = True
        if last_frm_is_end_point:
            cur_seg.contain_seg_end_point = True

    def OnSilenceDetected(self, valid_frame: int, cache: dict = None):
        """Onsilencedetected.
        
            Args:
                valid_frame: TODO.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        cache["stats"].lastest_confirmed_silence_frame = valid_frame
        if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
            self.PopDataBufTillFrame(valid_frame, cache=cache)

    # silence_detected_callback_
    # pass

    def OnVoiceDetected(self, valid_frame: int, cache: dict = None) -> None:
        """Onvoicedetected.
        
            Args:
                valid_frame: TODO.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        cache["stats"].latest_confirmed_speech_frame = valid_frame
        self.PopDataToOutputBuf(valid_frame, 1, False, False, False, cache=cache)

    def OnVoiceStart(self, start_frame: int, fake_result: bool = False, cache: dict = None) -> None:
        """Onvoicestart.
        
            Args:
                start_frame: TODO.
                fake_result: TODO.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        if self.vad_opts.do_start_point_detection:
            pass
        if cache["stats"].confirmed_start_frame != -1:
            print("not reset vad properly\n")
        else:
            cache["stats"].confirmed_start_frame = start_frame

        if (
            not fake_result
            and cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected
        ):
            self.PopDataToOutputBuf(
                cache["stats"].confirmed_start_frame, 1, True, False, False, cache=cache
            )

    def OnVoiceEnd(
        self, end_frame: int, fake_result: bool, is_last_frame: bool, cache: dict = None
    ) -> None:
        """Onvoiceend.
        
            Args:
                end_frame: TODO.
                fake_result: TODO.
                is_last_frame: Boolean flag for last frame.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        for t in range(cache["stats"].latest_confirmed_speech_frame + 1, end_frame):
            self.OnVoiceDetected(t, cache=cache)
        if self.vad_opts.do_end_point_detection:
            pass
        if cache["stats"].confirmed_end_frame != -1:
            print("not reset vad properly\n")
        else:
            cache["stats"].confirmed_end_frame = end_frame
        if not fake_result:
            cache["stats"].sil_frame = 0
            self.PopDataToOutputBuf(
                cache["stats"].confirmed_end_frame, 1, False, True, is_last_frame, cache=cache
            )
        cache["stats"].number_end_time_detected += 1

    def MaybeOnVoiceEndIfLastFrame(
        self, is_final_frame: bool, cur_frm_idx: int, cache: dict = None
    ) -> None:
        """Maybeonvoiceendiflastframe.
        
            Args:
                is_final_frame: Boolean flag for final frame.
                cur_frm_idx: TODO.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        if is_final_frame:
            self.OnVoiceEnd(cur_frm_idx, False, True, cache=cache)
            cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected

    def GetLatency(self, cache: dict = None) -> int:
        """Getlatency.
        
            Args:
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        return int(self.LatencyFrmNumAtStartPoint(cache=cache) * self.vad_opts.frame_in_ms)

    def LatencyFrmNumAtStartPoint(self, cache: dict = None) -> int:
        """Latencyfrmnumatstartpoint.
        
            Args:
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        vad_latency = cache["windows_detector"].GetWinSize()
        if self.vad_opts.do_extend:
            vad_latency += int(self.vad_opts.lookback_time_start_point / self.vad_opts.frame_in_ms)
        return vad_latency

    def GetFrameState(self, t: int, cache: dict = None):
        """Getframestate.
        
            Args:
                t: TODO.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        frame_state = FrameState.kFrameStateInvalid
        if t >= len(cache["stats"].decibel):
            return FrameState.kFrameStateSil
        cur_decibel = cache["stats"].decibel[t]
        cur_snr = cur_decibel - cache["stats"].noise_average_decibel
        # for each frame, calc log posterior probability of each state
        if cur_decibel < self.vad_opts.decibel_thres:
            frame_state = FrameState.kFrameStateSil
            return frame_state

        sum_score = 0.0
        noise_prob = 0.0
        assert len(cache["stats"].sil_pdf_ids) == self.vad_opts.silence_pdf_num
        if len(cache["stats"].sil_pdf_ids) > 0:
            assert len(cache["stats"].scores) == 1  # 只支持batch_size = 1的测试
            """
            - Change type of `sum_score` to float. The reason is that `sum_score` is a tensor with single element.
              and `torch.Tensor` is slower `float` when tensor has only one element.
            - Put the iteration of `sil_pdf_ids` inside `sum()` to reduce the overhead of creating a new list.
            - The default `sil_pdf_ids` is [0], the `if` statement is used to reduce the overhead of expression
              generation, which result in a mere (~2%) performance gain.
            """
            if len(cache["stats"].sil_pdf_ids) > 1:
                sum_score = sum(cache["stats"].scores[0][t][sil_pdf_id].item() for sil_pdf_id in cache["stats"].sil_pdf_ids)
            else:
                sum_score = cache["stats"].scores[0][t][cache["stats"].sil_pdf_ids[0]].item()
            noise_prob = math.log(sum_score) * self.vad_opts.speech_2_noise_ratio
            total_score = 1.0
            sum_score = total_score - sum_score
        speech_prob = math.log(sum_score)
        if self.vad_opts.output_frame_probs:
            frame_prob = E2EVadFrameProb()
            frame_prob.noise_prob = noise_prob
            frame_prob.speech_prob = speech_prob
            frame_prob.score = sum_score
            frame_prob.frame_id = t
            cache["stats"].frame_probs.append(frame_prob)
        if math.exp(speech_prob) >= math.exp(noise_prob) + cache["stats"].speech_noise_thres:
            if cur_snr >= self.vad_opts.snr_thres and cur_decibel >= self.vad_opts.decibel_thres:
                frame_state = FrameState.kFrameStateSpeech
            else:
                frame_state = FrameState.kFrameStateSil
        else:
            frame_state = FrameState.kFrameStateSil
            if cache["stats"].noise_average_decibel < -99.9:
                cache["stats"].noise_average_decibel = cur_decibel
            else:
                cache["stats"].noise_average_decibel = (
                    cur_decibel
                    + cache["stats"].noise_average_decibel
                    * (self.vad_opts.noise_frame_num_used_for_snr - 1)
                ) / self.vad_opts.noise_frame_num_used_for_snr

        return frame_state

    def forward(
        self,
        feats: torch.Tensor,
        waveform: torch.tensor,
        cache: dict = None,
        is_final: bool = False,
        **kwargs,
    ):
        """Forward pass for training.
        
            Args:
                feats: Feature tensor (e.g., fbank), shape (batch, frames, dim).
                waveform: TODO.
                cache: State cache dict for streaming inference.
                is_final: Whether this is the final chunk in streaming.
                **kwargs: Additional keyword arguments.
            """
        if cache is None:
            cache = {}
        if feats.numel() == 0:
            return []
        # if len(cache) == 0:
        #     self.AllResetDetection()
        # self.waveform = waveform  # compute decibel for each frame
        cache["stats"].waveform = waveform
        is_streaming_input = kwargs.get("is_streaming_input", True)
        self.ComputeDecibel(frame_count=feats.shape[1], cache=cache)
        self.ComputeScores(feats, cache=cache)
        if not is_final:
            self.DetectCommonFrames(cache=cache)
        else:
            self.DetectLastFrames(cache=cache)
        self.DropCachedFrames(cache["stats"].data_buf_start_frame, cache=cache)
        segments = []
        for batch_num in range(0, feats.shape[0]):  # only support batch_size = 1 now
            segment_batch = []
            if len(cache["stats"].output_data_buf) > 0:
                for i in range(
                    cache["stats"].output_data_buf_offset, len(cache["stats"].output_data_buf)
                ):
                    if (
                        is_streaming_input
                    ):  # in this case, return [beg, -1], [], [-1, end], [beg, end]
                        if not cache["stats"].output_data_buf[i].contain_seg_start_point:
                            continue
                        if (
                            not cache["stats"].next_seg
                            and not cache["stats"].output_data_buf[i].contain_seg_end_point
                        ):
                            continue
                        start_ms = (
                            cache["stats"].output_data_buf[i].start_ms
                            if cache["stats"].next_seg
                            else -1
                        )
                        if cache["stats"].output_data_buf[i].contain_seg_end_point:
                            end_ms = cache["stats"].output_data_buf[i].end_ms
                            cache["stats"].next_seg = True
                            cache["stats"].output_data_buf_offset += 1
                        else:
                            end_ms = -1
                            cache["stats"].next_seg = False
                        segment = [start_ms, end_ms]

                    else:  # in this case, return [beg, end]

                        if not is_final and (
                            not cache["stats"].output_data_buf[i].contain_seg_start_point
                            or not cache["stats"].output_data_buf[i].contain_seg_end_point
                        ):
                            continue
                        segment = [
                            cache["stats"].output_data_buf[i].start_ms,
                            cache["stats"].output_data_buf[i].end_ms,
                        ]
                        cache["stats"].output_data_buf_offset += 1  # need update this parameter

                    segment_batch.append(segment)

            if segment_batch:
                segments.append(segment_batch)
        # if is_final:
        #     # reset class variables and clear the dict for the next query
        #     self.AllResetDetection()
        return segments

    def init_cache(self, cache: dict = None, **kwargs):
        """Init cache.
        
            Args:
                cache: State cache dict for streaming inference.
                **kwargs: Additional keyword arguments.
            """
        if cache is None:
            cache = {}

        cache["frontend"] = {}
        cache["prev_samples"] = torch.empty(0)
        cache["encoder"] = {}

        if kwargs.get("max_end_silence_time") is not None:
            # update the max_end_silence_time
            self.vad_opts.max_end_silence_time = kwargs.get("max_end_silence_time")

        windows_detector = WindowDetector(
            self.vad_opts.window_size_ms,
            self.vad_opts.sil_to_speech_time_thres,
            self.vad_opts.speech_to_sil_time_thres,
            self.vad_opts.frame_in_ms,
        )
        windows_detector.Reset()

        stats = Stats(
            sil_pdf_ids=self.vad_opts.sil_pdf_ids,
            max_end_sil_frame_cnt_thresh=self.vad_opts.max_end_silence_time
            - self.vad_opts.speech_to_sil_time_thres,
            speech_noise_thres=kwargs.get(
                "speech_noise_thres", self.vad_opts.speech_noise_thres
            ),
        )
        cache["windows_detector"] = windows_detector
        cache["stats"] = stats
        return cache

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        cache: dict = None,
        **kwargs,
    ):
        """Run inference on input data.
        
            Args:
                data_in: Input data (audio samples, file paths, or text).
                data_lengths: Lengths of each input sample in the batch.
                key: Sample identifiers.
                tokenizer: Tokenizer instance for text encoding/decoding.
                frontend: Audio frontend for feature extraction.
                cache: State cache dict for streaming inference.
                **kwargs: Additional keyword arguments.
            """
        if cache is None:
            cache = {}
        if len(cache) == 0:
            self.init_cache(cache, **kwargs)

        meta_data = {}
        chunk_size = kwargs.get("chunk_size", 60000)  # 50ms
        chunk_stride_samples = int(chunk_size * frontend.fs / 1000)

        time1 = time.perf_counter()
        is_streaming_input = (
            kwargs.get("is_streaming_input", False)
            if chunk_size >= 15000
            else kwargs.get("is_streaming_input", True)
        )
        is_final = (
            kwargs.get("is_final", False) if is_streaming_input else kwargs.get("is_final", True)
        )
        cfg = {"is_final": is_final, "is_streaming_input": is_streaming_input}
        audio_sample_list = load_audio_text_image_video(
            data_in,
            fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
            cache=cfg,
        )
        _is_final = cfg["is_final"]  # if data_in is a file or url, set is_final=True
        is_streaming_input = cfg["is_streaming_input"]
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        if len(audio_sample_list) == 0 or (hasattr(audio_sample_list[0], '__len__') and len(audio_sample_list[0]) == 0):
            return [{"key": key[0] if key else "", "value": []}], meta_data
        assert len(audio_sample_list) == 1, "batch_size must be set 1"

        audio_sample = torch.cat((cache["prev_samples"], audio_sample_list[0]))

        n = int(len(audio_sample) // chunk_stride_samples + int(_is_final))
        m = int(len(audio_sample) % chunk_stride_samples * (1 - int(_is_final)))
        segments = []
        # Keep explicit fixed-threshold requests from being overwritten by the dynamic schedule.
        dynamic_silence = kwargs.get(
            "dynamic_silence", kwargs.get("max_end_silence_time") is None
        )
        silence_schedule = kwargs.get("silence_schedule", DEFAULT_SILENCE_SCHEDULE)
        speech_to_sil_ms = self.vad_opts.speech_to_sil_time_thres
        accumulated_ms = cache.get("_dynamic_accumulated_ms", 0)
        in_speech = cache.get("_dynamic_in_speech", False)
        frontend_inference_kwargs = {}
        if getattr(frontend, "supports_aligned_waveforms", False):
            frontend_inference_kwargs["return_waveform"] = True

        for i in range(n):
            kwargs["is_final"] = _is_final and i == n - 1
            audio_sample_i = audio_sample[i * chunk_stride_samples : (i + 1) * chunk_stride_samples]

            # Apply dynamic silence threshold (only accumulate while in speech)
            if dynamic_silence and "stats" in cache:
                vad_state = cache["stats"].vad_state_machine
                # VadStateMachine: 2=kVadInStateStartPointNotDetected, 3=kVadInStateInSpeechSegment
                if vad_state.value == 2 or in_speech:  # kVadInStateInSpeechSegment
                    accumulated_ms += chunk_size
                    in_speech = True
                for limit_ms, silence_ms in silence_schedule:
                    if accumulated_ms <= limit_ms:
                        cache["stats"].max_end_sil_frame_cnt_thresh = max(silence_ms - speech_to_sil_ms, 0)
                        cache["stats"].speech_noise_thres = 0.5
                        break
                cache["_dynamic_accumulated_ms"] = accumulated_ms
                cache["_dynamic_in_speech"] = in_speech

            # extract fbank feats
            speech, speech_lengths = extract_fbank(
                [audio_sample_i],
                data_type=kwargs.get("data_type", "sound"),
                frontend=frontend,
                cache=cache["frontend"],
                is_final=kwargs["is_final"],
                **frontend_inference_kwargs,
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            meta_data["batch_data_time"] = (
                speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
            )
            speech = speech.to(device=kwargs["device"])
            speech_lengths = speech_lengths.to(device=kwargs["device"])

            waveform = cache["frontend"].get("aligned_waveforms")
            if waveform is None or waveform.numel() == 0:
                # Custom frontends may expose an exact span under the legacy key.
                waveform = cache["frontend"].get("waveforms")
            if speech.numel() > 0 and (waveform is None or waveform.numel() == 0):
                raise RuntimeError(
                    "Streaming VAD frontend must provide aligned_waveforms or waveforms "
                    "when emitting feature frames"
                )
            batch = {
                "feats": speech,
                "waveform": waveform,
                "is_final": kwargs["is_final"],
                "cache": cache,
                "is_streaming_input": is_streaming_input,
            }
            segments_i = self.forward(**batch)
            if len(segments_i) > 0:
                segments.extend(*segments_i)
                if dynamic_silence:
                    accumulated_ms = 0
                    in_speech = False
                    cache["_dynamic_accumulated_ms"] = 0
                    cache["_dynamic_in_speech"] = False

        cache["prev_samples"] = audio_sample[-m:] if m > 0 else torch.empty(0)
        if _is_final:
            self.init_cache(cache)

        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer[f"{1}best_recog"]

        results = []
        result_i = {"key": key[0], "value": segments}
        # if "MODELSCOPE_ENVIRONMENT" in os.environ and os.environ["MODELSCOPE_ENVIRONMENT"] == "eas":
        # 	result_i = json.dumps(result_i)

        results.append(result_i)

        if ibest_writer is not None:
            ibest_writer["text"][key[0]] = segments

        return results, meta_data

    def export(self, **kwargs):

        """Export.
        
            Args:
                **kwargs: Additional keyword arguments.
            """
        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models

    def DetectCommonFrames(self, cache: dict = None) -> int:
        """Detectcommonframes.
        
            Args:
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = FrameState.kFrameStateInvalid
            frame_state = self.GetFrameState(
                cache["stats"].frm_cnt - 1 - i - cache["stats"].last_drop_frames, cache=cache
            )
            self.DetectOneFrame(frame_state, cache["stats"].frm_cnt - 1 - i, False, cache=cache)

        return 0

    def DetectLastFrames(self, cache: dict = None) -> int:
        """Detectlastframes.
        
            Args:
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = FrameState.kFrameStateInvalid
            frame_state = self.GetFrameState(
                cache["stats"].frm_cnt - 1 - i - cache["stats"].last_drop_frames, cache=cache
            )
            if i != 0:
                self.DetectOneFrame(frame_state, cache["stats"].frm_cnt - 1 - i, False, cache=cache)
            else:
                self.DetectOneFrame(frame_state, cache["stats"].frm_cnt - 1, True, cache=cache)

        return 0

    def DetectOneFrame(
        self, cur_frm_state: FrameState, cur_frm_idx: int, is_final_frame: bool, cache: dict = None
    ) -> None:
        """Detectoneframe.
        
            Args:
                cur_frm_state: TODO.
                cur_frm_idx: TODO.
                is_final_frame: Boolean flag for final frame.
                cache: State cache dict for streaming inference.
            """
        if cache is None:
            cache = {}
        tmp_cur_frm_state = FrameState.kFrameStateInvalid
        if cur_frm_state == FrameState.kFrameStateSpeech:
            if math.fabs(1.0) > self.vad_opts.fe_prior_thres:
                tmp_cur_frm_state = FrameState.kFrameStateSpeech
            else:
                tmp_cur_frm_state = FrameState.kFrameStateSil
        elif cur_frm_state == FrameState.kFrameStateSil:
            tmp_cur_frm_state = FrameState.kFrameStateSil
        state_change = cache["windows_detector"].DetectOneFrame(
            tmp_cur_frm_state, cur_frm_idx, cache=cache
        )
        frm_shift_in_ms = self.vad_opts.frame_in_ms
        if AudioChangeState.kChangeStateSil2Speech == state_change:
            silence_frame_count = cache["stats"].continous_silence_frame_count
            cache["stats"].continous_silence_frame_count = 0
            cache["stats"].pre_end_silence_detected = False
            start_frame = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                start_frame = max(
                    cache["stats"].data_buf_start_frame,
                    cur_frm_idx - self.LatencyFrmNumAtStartPoint(cache=cache),
                )
                self.OnVoiceStart(start_frame, cache=cache)
                cache["stats"].vad_state_machine = VadStateMachine.kVadInStateInSpeechSegment
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.OnVoiceDetected(t, cache=cache)
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                for t in range(cache["stats"].latest_confirmed_speech_frame + 1, cur_frm_idx):
                    self.OnVoiceDetected(t, cache=cache)
                if (
                    cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Sil == state_change:
            cache["stats"].continous_silence_frame_count = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                pass
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Speech == state_change:
            cache["stats"].continous_silence_frame_count = 0
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    cache["stats"].max_time_out = True
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass
        elif AudioChangeState.kChangeStateSil2Sil == state_change:
            cache["stats"].continous_silence_frame_count += 1
            if cache["stats"].vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                # silence timeout, return zero length decision
                if (
                    (self.vad_opts.detect_mode == VadDetectMode.kVadSingleUtteranceDetectMode.value)
                    and (
                        cache["stats"].continous_silence_frame_count * frm_shift_in_ms
                        > self.vad_opts.max_start_silence_time
                    )
                ) or (is_final_frame and cache["stats"].number_end_time_detected == 0):
                    for t in range(cache["stats"].lastest_confirmed_silence_frame + 1, cur_frm_idx):
                        self.OnSilenceDetected(t, cache=cache)
                    self.OnVoiceStart(0, True, cache=cache)
                    self.OnVoiceEnd(0, True, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    if cur_frm_idx >= self.LatencyFrmNumAtStartPoint(cache=cache):
                        self.OnSilenceDetected(
                            cur_frm_idx - self.LatencyFrmNumAtStartPoint(cache=cache), cache=cache
                        )
            elif cache["stats"].vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    cache["stats"].continous_silence_frame_count * frm_shift_in_ms
                    >= cache["stats"].max_end_sil_frame_cnt_thresh
                ):
                    lookback_frame = int(
                        cache["stats"].max_end_sil_frame_cnt_thresh / frm_shift_in_ms
                    )
                    if self.vad_opts.do_extend:
                        lookback_frame -= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                        )
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.OnVoiceEnd(cur_frm_idx - lookback_frame, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif (
                    cur_frm_idx - cache["stats"].confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False, cache=cache)
                    cache["stats"].vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif self.vad_opts.do_extend and not is_final_frame:
                    if cache["stats"].continous_silence_frame_count <= int(
                        self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                    ):
                        self.OnVoiceDetected(cur_frm_idx, cache=cache)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx, cache=cache)
            else:
                pass

        if (
            cache["stats"].vad_state_machine == VadStateMachine.kVadInStateEndPointDetected
            and self.vad_opts.detect_mode == VadDetectMode.kVadMutipleUtteranceDetectMode.value
        ):
            self.ResetDetection(cache=cache)
