# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Optional

from transformers import TrainingArguments


class FDivergenceType(Enum):
    REVERSE_KL = "reverse_kl"
    JS_DIVERGENCE = "js_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"


class FDivergenceConstants:
    ALPHA_DIVERGENCE_COEF_KEY = "alpha_divergence_coef"
    ALPHA_DIVERGENCE_COEF_DEFAULT = 1.0


@dataclass
class DPOConfig(TrainingArguments):
    r"""
    Initialize DPOConfig.

    Args:
        beta (`float`, *optional*, defaults to `0.1`):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, *optional*, defaults to `0.0`):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report and [Robust DPO](https://huggingface.co/papers/2403.00409) paper that should be between 0 and 0.5.
        loss_type (`str`, *optional*, defaults to `"sigmoid"`):
            The type of DPO loss to use. Possible values are:

                - `"sigmoid"`: sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"hinge"`: hinge loss on the normalized likelihood from the [SLiC](https://huggingface.co/papers/2305.10425) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
                - `"exo_pair"`: pairwise EXO loss from the [EXO](https://huggingface.co/papers/2402.00856) paper.
                - `"nca_pair"`: pairwise NCA loss from the [NCA](https://huggingface.co/papers/2402.05369) paper.
                - `"robust"`: unbiased estimate of the DPO loss that is robust to preference noise from the [Robust DPO](https://huggingface.co/papers/2403.00409) paper.
                - `"bco_pair"`: pairwise BCO loss from the [BCO](https://huggingface.co/papers/2404.04656) paper.
                - `"sppo_hard"`: SPPO loss with hard label from the [SPPO](https://huggingface.co/papers/2405.00675) paper.
                - `"aot"`: AOT loss for paired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
                - `"aot_pair"`: AOT loss for unpaired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
                - `"apo_zero"`: APO-zero loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
                - `"apo_down"`: APO-down loss from the [APO](https://huggingface.co/papers/2408.06266) paper.

        label_pad_token_id (`int`, *optional*, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`Optional[int]`, *optional*, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, *optional*, defaults to `"keep_end"`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        max_length (`Optional[int]`, *optional*, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`Optional[int]`, *optional*, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`Optional[int]`, *optional*, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        is_encoder_decoder(`Optional[int]`, *optional*, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, *optional*, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        precompute_ref_log_probs (`bool`, *optional*, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            The number of workers to use to tokenize the data. Defaults to None.
        model_init_kwargs (`Optional[Dict]`, *optional*, defaults to `None`):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs (`Optional[Dict]`, *optional*, defaults to `None`):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
        model_adapter_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool`, *optional*, defaults to `False`):
            If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
        force_use_ref_model (`bool`, *optional*, defaults to `False`):
            In case one passes a PEFT model for the active model and you want to use a different model for the ref_model, set this flag to `True`.
        f_divergence_type (`FDivergenceType`, *optional*, defaults to `FDivergenceType.REVERSE_KL`):
            The type of f-divergence regularization function to compute divergence between policy and reference model. This argument is optional, defaults to `FDivergenceType.REVERSE_KL`.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `1.0`):
            The alpha coef in alpha-divergence(u^-alpha) regularization function for DPO loss.
        sync_ref_model ('bool', *optional*, defaults to `False`):
            The flag for syncing reference model during training from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha ('float', *optional*, defaults to `1.0`):
            The alpha parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_sync_steps ('int', *optional*, defaults to `2`):
            The tau parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        rpo_alpha ('float', *optional*, defaults to `None`):
            The alpha parameter from the [RPO](https://huggingface.co/papers/2404.19733) paper V3. If None, no weighting is applied and the loss is the same as the DPO loss. The paper recommends `rpo_alpha=1.0`.
    """

    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: Literal[
        "sigmoid",
        "hinge",
        "ipo",
        "exo_pair",
        "nca_pair",
        "robust",
        "bco_pair",
        "sppo_hard",
        "aot",
        "aot_pair",
        "apo_zero",
        "apo_down",
    ] = "sigmoid"
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_target_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    f_divergence_type: FDivergenceType = FDivergenceType.REVERSE_KL
    f_alpha_divergence_coef: float = 1.0
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None

    def __post_init__(self):
        if self.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")
        return super().__post_init__()
