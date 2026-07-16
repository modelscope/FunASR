#!/usr/bin/env python3
"""DynamicStreamingVAD — 动态阈值流式 VAD 封装。

在 fsmn-vad 基础上，根据当前语音段的累积时长动态调整静音切分阈值：
短句等待更长静音（避免切碎），长句快速切分（避免堆积）。

支持流式（逐帧喂入）和非流式（一次性处理完整音频）两种调用方式。

Usage (流式):
    from funasr import AutoModel
    from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD

    vad_model = AutoModel(model="fsmn-vad", device="cuda:0")
    vad = DynamicStreamingVAD(vad_model)

    for audio_chunk in audio_stream:
        segments = vad.feed(audio_chunk)
        for seg in segments:
            print(f"Speech: {seg[0]}-{seg[1]}ms")

    # 结束时
    final_segments = vad.finalize()

Usage (非流式):
    segments = vad.process(full_audio_tensor)
    for seg in segments:
        print(f"Speech: {seg[0]}-{seg[1]}ms")
"""

from typing import List, Optional, Tuple

import torch
import numpy as np


# 默认动态阈值配置：(累积时长上限ms, 静音阈值ms)
DEFAULT_SILENCE_SCHEDULE = [
    (5000, 2000),
    (10000, 1500),
    (15000, 1000),
    (30000, 800),
    (45000, 400),
    (float('inf'), 100),
]


class DynamicStreamingVAD:
    """动态阈值流式 VAD。

    在 fsmn-vad 的流式推理基础上，根据当前语音段已累积的时长
    动态调整静音切分阈值，实现「短句不切碎、长句快切分」。

    Args:
        vad_model: FunASR AutoModel 加载的 fsmn-vad 模型实例。
        chunk_size_ms: 每次喂入 VAD 的 chunk 大小（毫秒），默认 60。
        speech_noise_thres: 语音/噪声判别阈值，默认 0.5。
        speech_to_sil_thres_ms: 语音转静音的基础时间（毫秒），默认 150。
        silence_schedule: 动态阈值配置表，格式为
            [(累积时长上限ms, 对应的静音阈值ms), ...]。
            当累积时长 <= 上限时，使用对应的静音阈值。
            默认值适合实时对话场景。设为 None 禁用动态调整（使用固定阈值）。
        sample_rate: 采样率，默认 16000。

    Example:
        # 自定义阈值：更激进的切分
        vad = DynamicStreamingVAD(
            vad_model,
            silence_schedule=[
                (3000, 1500),
                (8000, 800),
                (15000, 400),
                (float('inf'), 200),
            ],
        )
    """

    def __init__(
        self,
        vad_model,
        chunk_size_ms: int = 60,
        speech_noise_thres: float = 0.5,
        speech_to_sil_thres_ms: int = 150,
        silence_schedule: Optional[List[Tuple[float, int]]] = None,
        sample_rate: int = 16000,
    ):
        self.model = vad_model
        self.chunk_size_ms = chunk_size_ms
        self.speech_noise_thres = speech_noise_thres
        self.speech_to_sil_thres_ms = speech_to_sil_thres_ms
        self.silence_schedule = silence_schedule if silence_schedule is not None else DEFAULT_SILENCE_SCHEDULE
        self.sample_rate = sample_rate

        self.cache = {}
        self.confirmed_segments: List[List[int]] = []
        self.current_speech_start: Optional[int] = None
        self.accumulated_since_cut_ms: int = 0

    def _get_silence_threshold(self) -> int:
        """根据当前累积时长，从 schedule 中查询静音阈值。"""
        for limit_ms, silence_ms in self.silence_schedule:
            if self.accumulated_since_cut_ms <= limit_ms:
                return silence_ms
        return self.silence_schedule[-1][1]

    def _apply_dynamic_threshold(self):
        """将动态阈值应用到 VAD 内部 cache。"""
        if "stats" not in self.cache:
            return
        stats = self.cache["stats"]
        stats.speech_noise_thres = self.speech_noise_thres
        desired_silence_ms = self._get_silence_threshold()
        stats.max_end_sil_frame_cnt_thresh = max(desired_silence_ms - self.speech_to_sil_thres_ms, 0)

    def feed(self, audio_chunk: torch.Tensor, is_final: bool = False) -> List[List[int]]:
        """喂入一段音频，返回新确认的语音段。

        Args:
            audio_chunk: 音频数据（float32 tensor，16kHz）。
                可以是任意长度，内部按 chunk_size_ms 处理。
            is_final: 是否为最后一段音频。设为 True 时会强制结束当前语音段。

        Returns:
            新确认的语音段列表，每段为 [start_ms, end_ms]。
            仅在检测到语音结束时返回非空列表。
        """
        if audio_chunk.dim() > 1:
            audio_chunk = audio_chunk.squeeze()

        chunk_samples = len(audio_chunk)
        self.accumulated_since_cut_ms += int(chunk_samples * 1000 / self.sample_rate)

        self._apply_dynamic_threshold()

        res = self.model.generate(
            input=[audio_chunk], cache=self.cache,
            is_final=is_final, chunk_size=self.chunk_size_ms,
            silence_schedule=self.silence_schedule, #fix here
        )

        signals = res[0].get("value", [])
        new_confirmed = []

        for sig in signals:
            if sig[0] >= 0 and sig[1] == -1:
                self.current_speech_start = sig[0]
            elif sig[0] == -1 and sig[1] >= 0:
                start = self.current_speech_start if self.current_speech_start is not None else 0
                seg = [start, sig[1]]
                self.confirmed_segments.append(seg)
                new_confirmed.append(seg)
                self.current_speech_start = None
                self.accumulated_since_cut_ms = 0
            elif sig[0] >= 0 and sig[1] >= 0:
                self.confirmed_segments.append(sig)
                new_confirmed.append(sig)
                self.current_speech_start = None
                self.accumulated_since_cut_ms = 0

        return new_confirmed

    def finalize(self) -> List[List[int]]:
        """结束流式处理，返回最后可能未结束的语音段。

        调用此方法后，VAD 状态会被重置。
        如果当前有正在进行的语音段，会被强制结束。

        Returns:
            最后确认的语音段列表。
        """
        # Feed empty with is_final=True to flush
        empty = torch.zeros(int(self.sample_rate * 0.01), dtype=torch.float32)
        return self.feed(empty, is_final=True)

    def process(self, audio: torch.Tensor) -> List[List[int]]:
        """非流式接口：一次性处理完整音频，返回所有语音段。

        Args:
            audio: 完整音频（float32 tensor，16kHz）。

        Returns:
            所有检测到的语音段 [[start_ms, end_ms], ...]。
        """
        self.reset()

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.dim() > 1:
            audio = audio.squeeze()

        # 分 chunk 喂入
        chunk_samples = int(self.sample_rate * self.chunk_size_ms / 1000)
        total = len(audio)
        all_segments = []

        for i in range(0, total, chunk_samples):
            chunk = audio[i:i + chunk_samples]
            is_last = (i + chunk_samples >= total)
            segs = self.feed(chunk, is_final=is_last)
            all_segments.extend(segs)

        return all_segments

    @property
    def is_speaking(self) -> bool:
        """当前是否在语音状态中。"""
        return self.current_speech_start is not None

    @property
    def current_duration_ms(self) -> int:
        """当前段已累积的时长（毫秒）。"""
        return self.accumulated_since_cut_ms

    @property
    def current_threshold_ms(self) -> int:
        """当前使用的静音阈值（毫秒）。"""
        return self._get_silence_threshold()

    def reset(self):
        """重置所有状态，开始新一轮检测。"""
        self.cache = {}
        self.confirmed_segments = []
        self.current_speech_start = None
        self.accumulated_since_cut_ms = 0
