"""Helpers for the GLM-ASR vLLM serving path.

Kept dependency-free (standard library only) so the dtype guard can be unit
tested without a CUDA device, a torch build, or a vLLM installation.
"""

import logging

logger = logging.getLogger("funasr.glm_asr.vllm")

# Compute dtype that is known to degrade GLM-ASR transcription quality.
DEGRADED_DTYPE = "fp16"

# Warn only once per process so batch loops do not spam the log.
_warned_fp16 = False


def warn_if_degraded_dtype(dtype):
    """Warn once when a compute dtype is known to degrade GLM-ASR output.

    ``fp16`` can produce degraded or garbage transcription for GLM-ASR
    (numerical overflow in the audio embedding path), matching the documented
    Fun-ASR-Nano behaviour. The value is still honoured -- some GPUs only
    support fp16 -- but the caller is warned once about why output may be poor.

    Args:
        dtype: Requested compute dtype string ("bf16", "fp16", "fp32").

    Returns:
        ``dtype`` unchanged, so callers can wrap the value inline.
    """
    global _warned_fp16

    if dtype == DEGRADED_DTYPE and not _warned_fp16:
        logger.warning(
            "dtype='fp16' can produce degraded or garbage transcription for "
            "GLM-ASR (numerical overflow in the audio embedding path). "
            "Use dtype='bf16' (recommended) or dtype='fp32'. On GPUs without "
            "bfloat16 support (e.g. NVIDIA V100), use 'fp32'."
        )
        _warned_fp16 = True

    return dtype
