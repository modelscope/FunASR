"""Set of methods to build Transducer encoder architecture."""

from typing import Any, Dict, List, Optional, Union

from funasr.modules.activation import get_activation
from funasr.models.encoder.chunk_encoder_blocks.branchformer import Branchformer
from funasr.models.encoder.chunk_encoder_blocks.conformer import Conformer
from funasr.models.encoder.chunk_encoder_blocks.conv1d import Conv1d
from funasr.models.encoder.chunk_encoder_blocks.conv_input import ConvInput
from funasr.models.encoder.chunk_encoder_blocks.linear_input import LinearInput
from funasr.models.encoder.chunk_encoder_modules.attention import (  # noqa: H301
    RelPositionMultiHeadedAttention,
)
from funasr.models.encoder.chunk_encoder_modules.convolution import (  # noqa: H301
    ConformerConvolution,
    ConvolutionalSpatialGatingUnit,
)
from funasr.models.encoder.chunk_encoder_modules.multi_blocks import MultiBlocks
from funasr.models.encoder.chunk_encoder_modules.normalization import get_normalization
from funasr.models.encoder.chunk_encoder_modules.positional_encoding import (  # noqa: H301
    RelPositionalEncoding,
)
from funasr.modules.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


def build_main_parameters(
    pos_wise_act_type: str = "swish",
    conv_mod_act_type: str = "swish",
    pos_enc_dropout_rate: float = 0.0,
    pos_enc_max_len: int = 5000,
    simplified_att_score: bool = False,
    norm_type: str = "layer_norm",
    conv_mod_norm_type: str = "layer_norm",
    after_norm_eps: Optional[float] = None,
    after_norm_partial: Optional[float] = None,
    dynamic_chunk_training: bool = False,
    short_chunk_threshold: float = 0.75,
    short_chunk_size: int = 25,
    left_chunk_size: int = 0,
    time_reduction_factor: int = 1,
    unified_model_training: bool = False,
    default_chunk_size: int = 16,
    jitter_range: int =4,
    **activation_parameters,
) -> Dict[str, Any]:
    """Build encoder main parameters.

    Args:
        pos_wise_act_type: Conformer position-wise feed-forward activation type.
        conv_mod_act_type: Conformer convolution module activation type.
        pos_enc_dropout_rate: Positional encoding dropout rate.
        pos_enc_max_len: Positional encoding maximum length.
        simplified_att_score: Whether to use simplified attention score computation.
        norm_type: X-former normalization module type.
        conv_mod_norm_type: Conformer convolution module normalization type.
        after_norm_eps: Epsilon value for the final normalization.
        after_norm_partial: Value for the final normalization with RMSNorm.
        dynamic_chunk_training: Whether to use dynamic chunk training.
        short_chunk_threshold: Threshold for dynamic chunk selection.
        short_chunk_size: Minimum number of frames during dynamic chunk training.
        left_chunk_size: Number of frames in left context.
        **activations_parameters: Parameters of the activation functions.
                                    (See espnet2/asr_transducer/activation.py)

    Returns:
        : Main encoder parameters

    """
    main_params = {}

    main_params["pos_wise_act"] = get_activation(
        pos_wise_act_type, **activation_parameters
    )

    main_params["conv_mod_act"] = get_activation(
        conv_mod_act_type, **activation_parameters
    )

    main_params["pos_enc_dropout_rate"] = pos_enc_dropout_rate
    main_params["pos_enc_max_len"] = pos_enc_max_len

    main_params["simplified_att_score"] = simplified_att_score

    main_params["norm_type"] = norm_type
    main_params["conv_mod_norm_type"] = conv_mod_norm_type

    (
        main_params["after_norm_class"],
        main_params["after_norm_args"],
    ) = get_normalization(norm_type, eps=after_norm_eps, partial=after_norm_partial)

    main_params["dynamic_chunk_training"] = dynamic_chunk_training
    main_params["short_chunk_threshold"] = max(0, short_chunk_threshold)
    main_params["short_chunk_size"] = max(0, short_chunk_size)
    main_params["left_chunk_size"] = max(0, left_chunk_size)
    
    main_params["unified_model_training"] = unified_model_training
    main_params["default_chunk_size"] = max(0, default_chunk_size)
    main_params["jitter_range"] = max(0, jitter_range)
   
    main_params["time_reduction_factor"] = time_reduction_factor

    return main_params


def build_positional_encoding(
    block_size: int, configuration: Dict[str, Any]
) -> RelPositionalEncoding:
    """Build positional encoding block.

    Args:
        block_size: Input/output size.
        configuration: Positional encoding configuration.

    Returns:
        : Positional encoding module.

    """
    return RelPositionalEncoding(
        block_size,
        configuration.get("pos_enc_dropout_rate", 0.0),
        max_len=configuration.get("pos_enc_max_len", 5000),
    )


def build_input_block(
    input_size: int,
    configuration: Dict[str, Union[str, int]],
) -> ConvInput:
    """Build encoder input block.

    Args:
        input_size: Input size.
        configuration: Input block configuration.

    Returns:
        : ConvInput block function.

    """
    if configuration["linear"]:
        return LinearInput(
            input_size,
            configuration["output_size"],
            configuration["subsampling_factor"],
        )
    else:
        return ConvInput(
            input_size,
            configuration["conv_size"],
            configuration["subsampling_factor"],
            vgg_like=configuration["vgg_like"],
            output_size=configuration["output_size"],
        )


def build_branchformer_block(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
) -> Conformer:
    """Build Branchformer block.

    Args:
        configuration: Branchformer block configuration.
        main_params: Encoder main parameters.

    Returns:
        : Branchformer block function.

    """
    hidden_size = configuration["hidden_size"]
    linear_size = configuration["linear_size"]

    dropout_rate = configuration.get("dropout_rate", 0.0)

    conv_mod_norm_class, conv_mod_norm_args = get_normalization(
        main_params["conv_mod_norm_type"],
        eps=configuration.get("conv_mod_norm_eps"),
        partial=configuration.get("conv_mod_norm_partial"),
    )

    conv_mod_args = (
        linear_size,
        configuration["conv_mod_kernel_size"],
        conv_mod_norm_class,
        conv_mod_norm_args,
        dropout_rate,
        main_params["dynamic_chunk_training"],
    )

    mult_att_args = (
        configuration.get("heads", 4),
        hidden_size,
        configuration.get("att_dropout_rate", 0.0),
        main_params["simplified_att_score"],
    )

    norm_class, norm_args = get_normalization(
        main_params["norm_type"],
        eps=configuration.get("norm_eps"),
        partial=configuration.get("norm_partial"),
    )

    return lambda: Branchformer(
        hidden_size,
        linear_size,
        RelPositionMultiHeadedAttention(*mult_att_args),
        ConvolutionalSpatialGatingUnit(*conv_mod_args),
        norm_class=norm_class,
        norm_args=norm_args,
        dropout_rate=dropout_rate,
    )


def build_conformer_block(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
) -> Conformer:
    """Build Conformer block.

    Args:
        configuration: Conformer block configuration.
        main_params: Encoder main parameters.

    Returns:
        : Conformer block function.

    """
    hidden_size = configuration["hidden_size"]
    linear_size = configuration["linear_size"]

    pos_wise_args = (
        hidden_size,
        linear_size,
        configuration.get("pos_wise_dropout_rate", 0.0),
        main_params["pos_wise_act"],
    )

    conv_mod_norm_args = {
        "eps": configuration.get("conv_mod_norm_eps", 1e-05),
        "momentum": configuration.get("conv_mod_norm_momentum", 0.1),
    }

    conv_mod_args = (
        hidden_size,
        configuration["conv_mod_kernel_size"],
        main_params["conv_mod_act"],
        conv_mod_norm_args,
        main_params["dynamic_chunk_training"] or main_params["unified_model_training"],
    )

    mult_att_args = (
        configuration.get("heads", 4),
        hidden_size,
        configuration.get("att_dropout_rate", 0.0),
        main_params["simplified_att_score"],
    )

    norm_class, norm_args = get_normalization(
        main_params["norm_type"],
        eps=configuration.get("norm_eps"),
        partial=configuration.get("norm_partial"),
    )

    return lambda: Conformer(
        hidden_size,
        RelPositionMultiHeadedAttention(*mult_att_args),
        PositionwiseFeedForward(*pos_wise_args),
        PositionwiseFeedForward(*pos_wise_args),
        ConformerConvolution(*conv_mod_args),
        norm_class=norm_class,
        norm_args=norm_args,
        dropout_rate=configuration.get("dropout_rate", 0.0),
    )


def build_conv1d_block(
    configuration: List[Dict[str, Any]],
    causal: bool,
) -> Conv1d:
    """Build Conv1d block.

    Args:
        configuration: Conv1d block configuration.

    Returns:
        : Conv1d block function.

    """
    return lambda: Conv1d(
        configuration["input_size"],
        configuration["output_size"],
        configuration["kernel_size"],
        stride=configuration.get("stride", 1),
        dilation=configuration.get("dilation", 1),
        groups=configuration.get("groups", 1),
        bias=configuration.get("bias", True),
        relu=configuration.get("relu", True),
        batch_norm=configuration.get("batch_norm", False),
        causal=causal,
        dropout_rate=configuration.get("dropout_rate", 0.0),
    )


def build_body_blocks(
    configuration: List[Dict[str, Any]],
    main_params: Dict[str, Any],
    output_size: int,
) -> MultiBlocks:
    """Build encoder body blocks.

    Args:
        configuration: Body blocks configuration.
        main_params: Encoder main parameters.
        output_size: Architecture output size.

    Returns:
        MultiBlocks function encapsulation all encoder blocks.

    """
    fn_modules = []
    extended_conf = []

    for c in configuration:
        if c.get("num_blocks") is not None:
            extended_conf += c["num_blocks"] * [
                {c_i: c[c_i] for c_i in c if c_i != "num_blocks"}
            ]
        else:
            extended_conf += [c]

    for i, c in enumerate(extended_conf):
        block_type = c["block_type"]

        if block_type == "branchformer":
            module = build_branchformer_block(c, main_params)
        elif block_type == "conformer":
            module = build_conformer_block(c, main_params)
        elif block_type == "conv1d":
            module = build_conv1d_block(c, main_params["dynamic_chunk_training"])
        else:
            raise NotImplementedError

        fn_modules.append(module)

    return MultiBlocks(
        [fn() for fn in fn_modules],
        output_size,
        norm_class=main_params["after_norm_class"],
        norm_args=main_params["after_norm_args"],
    )
