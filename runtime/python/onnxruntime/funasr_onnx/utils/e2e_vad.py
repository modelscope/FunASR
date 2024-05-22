# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from enum import Enum
from typing import List, Tuple, Dict, Any

import math
import numpy as np


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
    ):
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
    def __init__(self):
        self.start_ms = 0
        self.end_ms = 0
        self.buffer = []
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0

    def Reset(self):
        self.start_ms = 0
        self.end_ms = 0
        self.buffer = []
        self.contain_seg_start_point = False
        self.contain_seg_end_point = False
        self.doa = 0


class E2EVadFrameProb(object):
    def __init__(self):
        self.noise_prob = 0.0
        self.speech_prob = 0.0
        self.score = 0.0
        self.frame_id = 0
        self.frm_state = 0


class WindowDetector(object):
    def __init__(
        self,
        window_size_ms: int,
        sil_to_speech_time: int,
        speech_to_sil_time: int,
        frame_size_ms: int,
    ):
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
        self.cur_win_pos = 0
        self.win_sum = 0
        self.win_state = [0] * self.win_size_frame
        self.pre_frame_state = FrameState.kFrameStateSil
        self.cur_frame_state = FrameState.kFrameStateSil
        self.voice_last_frame_count = 0
        self.noise_last_frame_count = 0
        self.hydre_frame_count = 0

    def GetWinSize(self) -> int:
        return int(self.win_size_frame)

    def DetectOneFrame(self, frameState: FrameState, frame_count: int) -> AudioChangeState:
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
        return int(self.frame_size_ms)


class E2EVadModel:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Deep-FSMN for Large Vocabulary Continuous Speech Recognition
    https://arxiv.org/abs/1803.05030
    """

    def __init__(self, vad_post_args: Dict[str, Any]):
        super(E2EVadModel, self).__init__()
        self.vad_opts = VADXOptions(**vad_post_args)
        self.windows_detector = WindowDetector(
            self.vad_opts.window_size_ms,
            self.vad_opts.sil_to_speech_time_thres,
            self.vad_opts.speech_to_sil_time_thres,
            self.vad_opts.frame_in_ms,
        )
        # self.encoder = encoder
        # init variables
        self.is_final = False
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
        self.sil_pdf_ids = self.vad_opts.sil_pdf_ids
        self.noise_average_decibel = -100.0
        self.pre_end_silence_detected = False
        self.next_seg = True

        self.output_data_buf = []
        self.output_data_buf_offset = 0
        self.frame_probs = []
        self.max_end_sil_frame_cnt_thresh = (
            self.vad_opts.max_end_silence_time - self.vad_opts.speech_to_sil_time_thres
        )
        self.speech_noise_thres = self.vad_opts.speech_noise_thres
        self.scores = None
        self.idx_pre_chunk = 0
        self.max_time_out = False
        self.decibel = []
        self.data_buf_size = 0
        self.data_buf_all_size = 0
        self.waveform = None
        self.ResetDetection()

    def AllResetDetection(self):
        self.is_final = False
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
        self.sil_pdf_ids = self.vad_opts.sil_pdf_ids
        self.noise_average_decibel = -100.0
        self.pre_end_silence_detected = False
        self.next_seg = True

        self.output_data_buf = []
        self.output_data_buf_offset = 0
        self.frame_probs = []
        self.max_end_sil_frame_cnt_thresh = (
            self.vad_opts.max_end_silence_time - self.vad_opts.speech_to_sil_time_thres
        )
        self.speech_noise_thres = self.vad_opts.speech_noise_thres
        self.scores = None
        self.idx_pre_chunk = 0
        self.max_time_out = False
        self.decibel = []
        self.data_buf_size = 0
        self.data_buf_all_size = 0
        self.waveform = None
        self.ResetDetection()

    def ResetDetection(self):
        self.continous_silence_frame_count = 0
        self.latest_confirmed_speech_frame = 0
        self.lastest_confirmed_silence_frame = -1
        self.confirmed_start_frame = -1
        self.confirmed_end_frame = -1
        self.vad_state_machine = VadStateMachine.kVadInStateStartPointNotDetected
        self.windows_detector.Reset()
        self.sil_frame = 0
        self.frame_probs = []

    def ComputeDecibel(self) -> None:
        frame_sample_length = int(self.vad_opts.frame_length_ms * self.vad_opts.sample_rate / 1000)
        frame_shift_length = int(self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000)
        if self.data_buf_all_size == 0:
            self.data_buf_all_size = len(self.waveform[0])
            self.data_buf_size = self.data_buf_all_size
        else:
            self.data_buf_all_size += len(self.waveform[0])
        for offset in range(
            0, self.waveform.shape[1] - frame_sample_length + 1, frame_shift_length
        ):
            self.decibel.append(
                10
                * math.log10(
                    np.square((self.waveform[0][offset : offset + frame_sample_length])).sum()
                    + 0.000001
                )
            )

    def ComputeScores(self, scores: np.ndarray) -> None:
        # scores = self.encoder(feats, in_cache)  # return B * T * D
        self.vad_opts.nn_eval_block_size = scores.shape[1]
        self.frm_cnt += scores.shape[1]  # count total frames
        self.scores = scores

    def PopDataBufTillFrame(self, frame_idx: int) -> None:  # need check again
        while self.data_buf_start_frame < frame_idx:
            if self.data_buf_size >= int(
                self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            ):
                self.data_buf_start_frame += 1
                self.data_buf_size = self.data_buf_all_size - self.data_buf_start_frame * int(
                    self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
                )

    def PopDataToOutputBuf(
        self,
        start_frm: int,
        frm_cnt: int,
        first_frm_is_start_point: bool,
        last_frm_is_end_point: bool,
        end_point_is_sent_end: bool,
    ) -> None:
        self.PopDataBufTillFrame(start_frm)
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
            expected_sample_number = max(expected_sample_number, self.data_buf_size)
        if self.data_buf_size < expected_sample_number:
            print("error in calling pop data_buf\n")

        if len(self.output_data_buf) == 0 or first_frm_is_start_point:
            self.output_data_buf.append(E2EVadSpeechBufWithDoa())
            self.output_data_buf[-1].Reset()
            self.output_data_buf[-1].start_ms = start_frm * self.vad_opts.frame_in_ms
            self.output_data_buf[-1].end_ms = self.output_data_buf[-1].start_ms
            self.output_data_buf[-1].doa = 0
        cur_seg = self.output_data_buf[-1]
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            print("warning\n")
        out_pos = len(cur_seg.buffer)  # cur_seg.buff现在没做任何操作
        data_to_pop = 0
        if end_point_is_sent_end:
            data_to_pop = expected_sample_number
        else:
            data_to_pop = int(
                frm_cnt * self.vad_opts.frame_in_ms * self.vad_opts.sample_rate / 1000
            )
        if data_to_pop > self.data_buf_size:
            print("VAD data_to_pop is bigger than self.data_buf_size!!!\n")
            data_to_pop = self.data_buf_size
            expected_sample_number = self.data_buf_size

        cur_seg.doa = 0
        for sample_cpy_out in range(0, data_to_pop):
            # cur_seg.buffer[out_pos ++] = data_buf_.back();
            out_pos += 1
        for sample_cpy_out in range(data_to_pop, expected_sample_number):
            # cur_seg.buffer[out_pos++] = data_buf_.back()
            out_pos += 1
        if cur_seg.end_ms != start_frm * self.vad_opts.frame_in_ms:
            print("Something wrong with the VAD algorithm\n")
        self.data_buf_start_frame += frm_cnt
        cur_seg.end_ms = (start_frm + frm_cnt) * self.vad_opts.frame_in_ms
        if first_frm_is_start_point:
            cur_seg.contain_seg_start_point = True
        if last_frm_is_end_point:
            cur_seg.contain_seg_end_point = True

    def OnSilenceDetected(self, valid_frame: int):
        self.lastest_confirmed_silence_frame = valid_frame
        if self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
            self.PopDataBufTillFrame(valid_frame)
        # silence_detected_callback_
        # pass

    def OnVoiceDetected(self, valid_frame: int) -> None:
        self.latest_confirmed_speech_frame = valid_frame
        self.PopDataToOutputBuf(valid_frame, 1, False, False, False)

    def OnVoiceStart(self, start_frame: int, fake_result: bool = False) -> None:
        if self.vad_opts.do_start_point_detection:
            pass
        if self.confirmed_start_frame != -1:
            print("not reset vad properly\n")
        else:
            self.confirmed_start_frame = start_frame

        if (
            not fake_result
            and self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected
        ):
            self.PopDataToOutputBuf(self.confirmed_start_frame, 1, True, False, False)

    def OnVoiceEnd(self, end_frame: int, fake_result: bool, is_last_frame: bool) -> None:
        for t in range(self.latest_confirmed_speech_frame + 1, end_frame):
            self.OnVoiceDetected(t)
        if self.vad_opts.do_end_point_detection:
            pass
        if self.confirmed_end_frame != -1:
            print("not reset vad properly\n")
        else:
            self.confirmed_end_frame = end_frame
        if not fake_result:
            self.sil_frame = 0
            self.PopDataToOutputBuf(self.confirmed_end_frame, 1, False, True, is_last_frame)
        self.number_end_time_detected += 1

    def MaybeOnVoiceEndIfLastFrame(self, is_final_frame: bool, cur_frm_idx: int) -> None:
        if is_final_frame:
            self.OnVoiceEnd(cur_frm_idx, False, True)
            self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected

    def GetLatency(self) -> int:
        return int(self.LatencyFrmNumAtStartPoint() * self.vad_opts.frame_in_ms)

    def LatencyFrmNumAtStartPoint(self) -> int:
        vad_latency = self.windows_detector.GetWinSize()
        if self.vad_opts.do_extend:
            vad_latency += int(self.vad_opts.lookback_time_start_point / self.vad_opts.frame_in_ms)
        return vad_latency

    def GetFrameState(self, t: int) -> FrameState:
        frame_state = FrameState.kFrameStateInvalid
        cur_decibel = self.decibel[t]
        cur_snr = cur_decibel - self.noise_average_decibel
        # for each frame, calc log posterior probability of each state
        if cur_decibel < self.vad_opts.decibel_thres:
            frame_state = FrameState.kFrameStateSil
            self.DetectOneFrame(frame_state, t, False)
            return frame_state

        sum_score = 0.0
        noise_prob = 0.0
        assert len(self.sil_pdf_ids) == self.vad_opts.silence_pdf_num
        if len(self.sil_pdf_ids) > 0:
            assert len(self.scores) == 1  # 只支持batch_size = 1的测试
            sil_pdf_scores = [
                self.scores[0][t - self.idx_pre_chunk][sil_pdf_id]
                for sil_pdf_id in self.sil_pdf_ids
            ]
            sum_score = sum(sil_pdf_scores)
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
            self.frame_probs.append(frame_prob)
        if math.exp(speech_prob) >= math.exp(noise_prob) + self.speech_noise_thres:
            if cur_snr >= self.vad_opts.snr_thres and cur_decibel >= self.vad_opts.decibel_thres:
                frame_state = FrameState.kFrameStateSpeech
            else:
                frame_state = FrameState.kFrameStateSil
        else:
            frame_state = FrameState.kFrameStateSil
            if self.noise_average_decibel < -99.9:
                self.noise_average_decibel = cur_decibel
            else:
                self.noise_average_decibel = (
                    cur_decibel
                    + self.noise_average_decibel * (self.vad_opts.noise_frame_num_used_for_snr - 1)
                ) / self.vad_opts.noise_frame_num_used_for_snr

        return frame_state

    def __call__(
        self,
        score: np.ndarray,
        waveform: np.ndarray,
        is_final: bool = False,
        max_end_sil: int = 800,
        online: bool = False,
    ):
        self.max_end_sil_frame_cnt_thresh = max_end_sil - self.vad_opts.speech_to_sil_time_thres
        self.waveform = waveform  # compute decibel for each frame
        self.ComputeDecibel()
        self.ComputeScores(score)
        if not is_final:
            self.DetectCommonFrames()
        else:
            self.DetectLastFrames()
        segments = []
        for batch_num in range(0, score.shape[0]):  # only support batch_size = 1 now
            segment_batch = []
            if len(self.output_data_buf) > 0:
                for i in range(self.output_data_buf_offset, len(self.output_data_buf)):
                    if online:
                        if not self.output_data_buf[i].contain_seg_start_point:
                            continue
                        if not self.next_seg and not self.output_data_buf[i].contain_seg_end_point:
                            continue
                        start_ms = self.output_data_buf[i].start_ms if self.next_seg else -1
                        if self.output_data_buf[i].contain_seg_end_point:
                            end_ms = self.output_data_buf[i].end_ms
                            self.next_seg = True
                            self.output_data_buf_offset += 1
                        else:
                            end_ms = -1
                            self.next_seg = False
                    else:
                        if not is_final and (
                            not self.output_data_buf[i].contain_seg_start_point
                            or not self.output_data_buf[i].contain_seg_end_point
                        ):
                            continue
                        start_ms = self.output_data_buf[i].start_ms
                        end_ms = self.output_data_buf[i].end_ms
                        self.output_data_buf_offset += 1
                    segment = [start_ms, end_ms]
                    segment_batch.append(segment)

            if segment_batch:
                segments.append(segment_batch)
        if is_final:
            # reset class variables and clear the dict for the next query
            self.AllResetDetection()
        return segments

    def DetectCommonFrames(self) -> int:
        if self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = FrameState.kFrameStateInvalid
            frame_state = self.GetFrameState(self.frm_cnt - 1 - i)
            self.DetectOneFrame(frame_state, self.frm_cnt - 1 - i, False)
        self.idx_pre_chunk += self.scores.shape[1]
        return 0

    def DetectLastFrames(self) -> int:
        if self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected:
            return 0
        for i in range(self.vad_opts.nn_eval_block_size - 1, -1, -1):
            frame_state = FrameState.kFrameStateInvalid
            frame_state = self.GetFrameState(self.frm_cnt - 1 - i)
            if i != 0:
                self.DetectOneFrame(frame_state, self.frm_cnt - 1 - i, False)
            else:
                self.DetectOneFrame(frame_state, self.frm_cnt - 1, True)

        return 0

    def DetectOneFrame(
        self, cur_frm_state: FrameState, cur_frm_idx: int, is_final_frame: bool
    ) -> None:
        tmp_cur_frm_state = FrameState.kFrameStateInvalid
        if cur_frm_state == FrameState.kFrameStateSpeech:
            if math.fabs(1.0) > self.vad_opts.fe_prior_thres:
                tmp_cur_frm_state = FrameState.kFrameStateSpeech
            else:
                tmp_cur_frm_state = FrameState.kFrameStateSil
        elif cur_frm_state == FrameState.kFrameStateSil:
            tmp_cur_frm_state = FrameState.kFrameStateSil
        state_change = self.windows_detector.DetectOneFrame(tmp_cur_frm_state, cur_frm_idx)
        frm_shift_in_ms = self.vad_opts.frame_in_ms
        if AudioChangeState.kChangeStateSil2Speech == state_change:
            silence_frame_count = self.continous_silence_frame_count
            self.continous_silence_frame_count = 0
            self.pre_end_silence_detected = False
            start_frame = 0
            if self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                start_frame = max(
                    self.data_buf_start_frame, cur_frm_idx - self.LatencyFrmNumAtStartPoint()
                )
                self.OnVoiceStart(start_frame)
                self.vad_state_machine = VadStateMachine.kVadInStateInSpeechSegment
                for t in range(start_frame + 1, cur_frm_idx + 1):
                    self.OnVoiceDetected(t)
            elif self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                for t in range(self.latest_confirmed_speech_frame + 1, cur_frm_idx):
                    self.OnVoiceDetected(t)
                if (
                    cur_frm_idx - self.confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Sil == state_change:
            self.continous_silence_frame_count = 0
            if self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                pass
            elif self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    cur_frm_idx - self.confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx)
            else:
                pass
        elif AudioChangeState.kChangeStateSpeech2Speech == state_change:
            self.continous_silence_frame_count = 0
            if self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    cur_frm_idx - self.confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.max_time_out = True
                    self.OnVoiceEnd(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif not is_final_frame:
                    self.OnVoiceDetected(cur_frm_idx)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx)
            else:
                pass
        elif AudioChangeState.kChangeStateSil2Sil == state_change:
            self.continous_silence_frame_count += 1
            if self.vad_state_machine == VadStateMachine.kVadInStateStartPointNotDetected:
                # silence timeout, return zero length decision
                if (
                    (self.vad_opts.detect_mode == VadDetectMode.kVadSingleUtteranceDetectMode.value)
                    and (
                        self.continous_silence_frame_count * frm_shift_in_ms
                        > self.vad_opts.max_start_silence_time
                    )
                ) or (is_final_frame and self.number_end_time_detected == 0):
                    for t in range(self.lastest_confirmed_silence_frame + 1, cur_frm_idx):
                        self.OnSilenceDetected(t)
                    self.OnVoiceStart(0, True)
                    self.OnVoiceEnd(0, True, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                else:
                    if cur_frm_idx >= self.LatencyFrmNumAtStartPoint():
                        self.OnSilenceDetected(cur_frm_idx - self.LatencyFrmNumAtStartPoint())
            elif self.vad_state_machine == VadStateMachine.kVadInStateInSpeechSegment:
                if (
                    self.continous_silence_frame_count * frm_shift_in_ms
                    >= self.max_end_sil_frame_cnt_thresh
                ):
                    lookback_frame = int(self.max_end_sil_frame_cnt_thresh / frm_shift_in_ms)
                    if self.vad_opts.do_extend:
                        lookback_frame -= int(
                            self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                        )
                        lookback_frame -= 1
                        lookback_frame = max(0, lookback_frame)
                    self.OnVoiceEnd(cur_frm_idx - lookback_frame, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif (
                    cur_frm_idx - self.confirmed_start_frame + 1
                    > self.vad_opts.max_single_segment_time / frm_shift_in_ms
                ):
                    self.OnVoiceEnd(cur_frm_idx, False, False)
                    self.vad_state_machine = VadStateMachine.kVadInStateEndPointDetected
                elif self.vad_opts.do_extend and not is_final_frame:
                    if self.continous_silence_frame_count <= int(
                        self.vad_opts.lookahead_time_end_point / frm_shift_in_ms
                    ):
                        self.OnVoiceDetected(cur_frm_idx)
                else:
                    self.MaybeOnVoiceEndIfLastFrame(is_final_frame, cur_frm_idx)
            else:
                pass

        if (
            self.vad_state_machine == VadStateMachine.kVadInStateEndPointDetected
            and self.vad_opts.detect_mode == VadDetectMode.kVadMutipleUtteranceDetectMode.value
        ):
            self.ResetDetection()
