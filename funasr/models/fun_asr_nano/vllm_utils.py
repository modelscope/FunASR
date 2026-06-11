"""Helpers shared by the Fun-ASR-Nano vLLM serving paths.

Kept dependency-free (standard library only) so it can be imported and unit
tested without a CUDA device or a vLLM installation.
"""

import logging

logger = logging.getLogger("funasr.fun_asr_nano.vllm")

# A repetition penalty of 1.0 is the identity value, i.e. "no penalty".
NEUTRAL_REPETITION_PENALTY = 1.0

# Warn only once per process so streaming/batch loops do not spam the log.
_warned_prompt_embeds = False


def resolve_repetition_penalty(repetition_penalty, *, prompt_embeds=True):
    """Return a repetition penalty that is safe for the requested vLLM mode.

    Fun-ASR-Nano feeds vLLM precomputed audio/text *embeddings* with
    ``enable_prompt_embeds=True``. In that mode a request carries no prompt
    token IDs. vLLM applies ``repetition_penalty`` by scattering over the
    prompt's token IDs, so any value other than 1.0 indexes an empty token-id
    tensor and aborts the engine with a CUDA
    ``scatter gather kernel index out of bounds`` assertion (issue #2948).

    When ``prompt_embeds`` is True we therefore force the penalty back to the
    neutral value and warn once. With ``prompt_embeds=False`` (regular
    token-prompt decoding) the requested value is passed through unchanged.

    Args:
        repetition_penalty: Penalty requested by the caller. ``None`` is
            treated as "unset" and maps to the neutral value.
        prompt_embeds: Whether the request runs in vLLM prompt-embeds mode.

    Returns:
        A repetition penalty that will not crash the engine.
    """
    global _warned_prompt_embeds

    if repetition_penalty is None:
        return NEUTRAL_REPETITION_PENALTY

    if prompt_embeds and repetition_penalty != NEUTRAL_REPETITION_PENALTY:
        if not _warned_prompt_embeds:
            logger.warning(
                "repetition_penalty=%s is not supported in vLLM prompt-embeds "
                "mode (no prompt token IDs to penalize) and would trigger a CUDA "
                "scatter index-out-of-bounds crash; using repetition_penalty=%s "
                "instead. See https://github.com/modelscope/FunASR/issues/2948.",
                repetition_penalty,
                NEUTRAL_REPETITION_PENALTY,
            )
            _warned_prompt_embeds = True
        return NEUTRAL_REPETITION_PENALTY

    return repetition_penalty
