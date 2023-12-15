# Copyright 2020 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer common arguments."""


def add_arguments_rnn_encoder_common(group):
    """Define common arguments for RNN encoder."""
    group.add_argument(
        "--etype",
        default="blstmp",
        type=str,
        choices=[
            "lstm",
            "blstm",
            "lstmp",
            "blstmp",
            "vgglstmp",
            "vggblstmp",
            "vgglstm",
            "vggblstm",
            "gru",
            "bgru",
            "grup",
            "bgrup",
            "vgggrup",
            "vggbgrup",
            "vgggru",
            "vggbgru",
        ],
        help="Type of encoder network architecture",
    )
    group.add_argument(
        "--elayers",
        default=4,
        type=int,
        help="Number of encoder layers",
    )
    group.add_argument(
        "--eunits",
        "-u",
        default=300,
        type=int,
        help="Number of encoder hidden units",
    )
    group.add_argument(
        "--eprojs", default=320, type=int, help="Number of encoder projection units"
    )
    group.add_argument(
        "--subsample",
        default="1",
        type=str,
        help="Subsample input frames x_y_z means "
        "subsample every x frame at 1st layer, "
        "every y frame at 2nd layer etc.",
    )
    return group


def add_arguments_rnn_decoder_common(group):
    """Define common arguments for RNN decoder."""
    group.add_argument(
        "--dtype",
        default="lstm",
        type=str,
        choices=["lstm", "gru"],
        help="Type of decoder network architecture",
    )
    group.add_argument(
        "--dlayers", default=1, type=int, help="Number of decoder layers"
    )
    group.add_argument(
        "--dunits", default=320, type=int, help="Number of decoder hidden units"
    )
    group.add_argument(
        "--dropout-rate-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder",
    )
    group.add_argument(
        "--sampling-probability",
        default=0.0,
        type=float,
        help="Ratio of predicted labels fed back to decoder",
    )
    group.add_argument(
        "--lsm-type",
        const="",
        default="",
        type=str,
        nargs="?",
        choices=["", "unigram"],
        help="Apply label smoothing with a specified distribution type",
    )
    return group


def add_arguments_rnn_attention_common(group):
    """Define common arguments for RNN attention."""
    group.add_argument(
        "--atype",
        default="dot",
        type=str,
        choices=[
            "noatt",
            "dot",
            "add",
            "location",
            "coverage",
            "coverage_location",
            "location2d",
            "location_recurrent",
            "multi_head_dot",
            "multi_head_add",
            "multi_head_loc",
            "multi_head_multi_res_loc",
        ],
        help="Type of attention architecture",
    )
    group.add_argument(
        "--adim",
        default=320,
        type=int,
        help="Number of attention transformation dimensions",
    )
    group.add_argument(
        "--awin", default=5, type=int, help="Window size for location2d attention"
    )
    group.add_argument(
        "--aheads",
        default=4,
        type=int,
        help="Number of heads for multi head attention",
    )
    group.add_argument(
        "--aconv-chans",
        default=-1,
        type=int,
        help="Number of attention convolution channels \
                       (negative value indicates no location-aware attention)",
    )
    group.add_argument(
        "--aconv-filts",
        default=100,
        type=int,
        help="Number of attention convolution filters \
                       (negative value indicates no location-aware attention)",
    )
    group.add_argument(
        "--dropout-rate",
        default=0.0,
        type=float,
        help="Dropout rate for the encoder",
    )
    return group
