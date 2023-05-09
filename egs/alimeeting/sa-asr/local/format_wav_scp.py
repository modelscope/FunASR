#!/usr/bin/env python3
import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Tuple, Optional

import kaldiio
import humanfriendly
import numpy as np
import resampy
import soundfile
from tqdm import tqdm
from typeguard import check_argument_types

from funasr.utils.cli_utils import get_commandline_args
from funasr.fileio.read_text import read_2column_text
from funasr.fileio.sound_scp import SoundScpWriter


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def str2int_tuple(integers: str) -> Optional[Tuple[int, ...]]:
    """

    >>> str2int_tuple('3,4,5')
    (3, 4, 5)

    """
    assert check_argument_types()
    if integers.strip() in ("none", "None", "NONE", "null", "Null", "NULL"):
        return None
    return tuple(map(int, integers.strip().split(",")))


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='Create waves list from "wav.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp")
    parser.add_argument("outdir")
    parser.add_argument(
        "--name",
        default="wav",
        help="Specify the prefix word of output file name " 'such as "wav.scp"',
    )
    parser.add_argument("--segments", default=None)
    parser.add_argument(
        "--fs",
        type=humanfriendly_or_none,
        default=None,
        help="If the sampling rate specified, " "Change the sampling rate.",
    )
    parser.add_argument("--audio-format", default="wav")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ref-channels", default=None, type=str2int_tuple)
    group.add_argument("--utt2ref-channels", default=None, type=str)
    args = parser.parse_args()

    out_num_samples = Path(args.outdir) / f"utt2num_samples"

    if args.ref_channels is not None:

        def utt2ref_channels(x) -> Tuple[int, ...]:
            return args.ref_channels

    elif args.utt2ref_channels is not None:
        utt2ref_channels_dict = read_2column_text(args.utt2ref_channels)

        def utt2ref_channels(x, d=utt2ref_channels_dict) -> Tuple[int, ...]:
            chs_str = d[x]
            return tuple(map(int, chs_str.split()))

    else:
        utt2ref_channels = None

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_wavscp = Path(args.outdir) / f"{args.name}.scp"
    if args.segments is not None:
        # Note: kaldiio supports only wav-pcm-int16le file.
        loader = kaldiio.load_scp_sequential(args.scp, segments=args.segments)
        if args.audio_format.endswith("ark"):
            fark = open(Path(args.outdir) / f"data_{args.name}.ark", "wb")
            fscp = out_wavscp.open("w")
        else:
            writer = SoundScpWriter(
                args.outdir,
                out_wavscp,
                format=args.audio_format,
            )

        with out_num_samples.open("w") as fnum_samples:
            for uttid, (rate, wave) in tqdm(loader):
                # wave: (Time,) or (Time, Nmic)
                if wave.ndim == 2 and utt2ref_channels is not None:
                    wave = wave[:, utt2ref_channels(uttid)]

                if args.fs is not None and args.fs != rate:
                    # FIXME(kamo): To use sox?
                    wave = resampy.resample(
                        wave.astype(np.float64), rate, args.fs, axis=0
                    )
                    wave = wave.astype(np.int16)
                    rate = args.fs
                if args.audio_format.endswith("ark"):
                    if "flac" in args.audio_format:
                        suf = "flac"
                    elif "wav" in args.audio_format:
                        suf = "wav"
                    else:
                        raise RuntimeError("wav.ark or flac")

                    # NOTE(kamo): Using extended ark format style here.
                    # This format is incompatible with Kaldi
                    kaldiio.save_ark(
                        fark,
                        {uttid: (wave, rate)},
                        scp=fscp,
                        append=True,
                        write_function=f"soundfile_{suf}",
                    )

                else:
                    writer[uttid] = rate, wave
                fnum_samples.write(f"{uttid} {len(wave)}\n")
    else:
        if args.audio_format.endswith("ark"):
            fark = open(Path(args.outdir) / f"data_{args.name}.ark", "wb")
        else:
            wavdir = Path(args.outdir) / f"data_{args.name}"
            wavdir.mkdir(parents=True, exist_ok=True)

        with Path(args.scp).open("r") as fscp, out_wavscp.open(
            "w"
        ) as fout, out_num_samples.open("w") as fnum_samples:
            for line in tqdm(fscp):
                uttid, wavpath = line.strip().split(None, 1)

                if wavpath.endswith("|"):
                    # Streaming input e.g. cat a.wav |
                    with kaldiio.open_like_kaldi(wavpath, "rb") as f:
                        with BytesIO(f.read()) as g:
                            wave, rate = soundfile.read(g, dtype=np.int16)
                            if wave.ndim == 2 and utt2ref_channels is not None:
                                wave = wave[:, utt2ref_channels(uttid)]

                        if args.fs is not None and args.fs != rate:
                            # FIXME(kamo): To use sox?
                            wave = resampy.resample(
                                wave.astype(np.float64), rate, args.fs, axis=0
                            )
                            wave = wave.astype(np.int16)
                            rate = args.fs

                        if args.audio_format.endswith("ark"):
                            if "flac" in args.audio_format:
                                suf = "flac"
                            elif "wav" in args.audio_format:
                                suf = "wav"
                            else:
                                raise RuntimeError("wav.ark or flac")

                            # NOTE(kamo): Using extended ark format style here.
                            # This format is incompatible with Kaldi
                            kaldiio.save_ark(
                                fark,
                                {uttid: (wave, rate)},
                                scp=fout,
                                append=True,
                                write_function=f"soundfile_{suf}",
                            )
                        else:
                            owavpath = str(wavdir / f"{uttid}.{args.audio_format}")
                            soundfile.write(owavpath, wave, rate)
                            fout.write(f"{uttid} {owavpath}\n")
                else:
                    wave, rate = soundfile.read(wavpath, dtype=np.int16)
                    if wave.ndim == 2 and utt2ref_channels is not None:
                        wave = wave[:, utt2ref_channels(uttid)]
                        save_asis = False

                    elif args.audio_format.endswith("ark"):
                        save_asis = False

                    elif Path(wavpath).suffix == "." + args.audio_format and (
                        args.fs is None or args.fs == rate
                    ):
                        save_asis = True

                    else:
                        save_asis = False

                    if save_asis:
                        # Neither --segments nor --fs are specified and
                        # the line doesn't end with "|",
                        # i.e. not using unix-pipe,
                        # only in this case,
                        # just using the original file as is.
                        fout.write(f"{uttid} {wavpath}\n")
                    else:
                        if args.fs is not None and args.fs != rate:
                            # FIXME(kamo): To use sox?
                            wave = resampy.resample(
                                wave.astype(np.float64), rate, args.fs, axis=0
                            )
                            wave = wave.astype(np.int16)
                            rate = args.fs

                        if args.audio_format.endswith("ark"):
                            if "flac" in args.audio_format:
                                suf = "flac"
                            elif "wav" in args.audio_format:
                                suf = "wav"
                            else:
                                raise RuntimeError("wav.ark or flac")

                            # NOTE(kamo): Using extended ark format style here.
                            # This format is not supported in Kaldi.
                            kaldiio.save_ark(
                                fark,
                                {uttid: (wave, rate)},
                                scp=fout,
                                append=True,
                                write_function=f"soundfile_{suf}",
                            )
                        else:
                            owavpath = str(wavdir / f"{uttid}.{args.audio_format}")
                            soundfile.write(owavpath, wave, rate)
                            fout.write(f"{uttid} {owavpath}\n")
                fnum_samples.write(f"{uttid} {len(wave)}\n")


if __name__ == "__main__":
    main()
