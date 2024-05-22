#!/usr/bin/env python3
import argparse
from collections import Counter
import logging
from pathlib import Path
import sys
from typing import List
from typing import Optional


from funasr.utils.cli_utils import get_commandline_args
from funasr.tokenizer.build_tokenizer import build_tokenizer
from funasr.tokenizer.cleaner import TextCleaner
from funasr.tokenizer.phoneme_tokenizer import g2p_classes
from funasr.utils.types import str2bool
from funasr.utils.types import str_or_none


def field2slice(field: Optional[str]) -> slice:
    """Convert field string to slice

    Note that field string accepts 1-based integer.

    Examples:
        >>> field2slice("1-")
        slice(0, None, None)
        >>> field2slice("1-3")
        slice(0, 3, None)
        >>> field2slice("-3")
        slice(None, 3, None)
    """
    field = field.strip()
    try:
        if "-" in field:
            # e.g. "2-" or "2-5" or "-7"
            s1, s2 = field.split("-", maxsplit=1)
            if s1.strip() == "":
                s1 = None
            else:
                s1 = int(s1)
                if s1 == 0:
                    raise ValueError("1-based string")
            if s2.strip() == "":
                s2 = None
            else:
                s2 = int(s2)
        else:
            # e.g. "2"
            s1 = int(field)
            s2 = s1 + 1
            if s1 == 0:
                raise ValueError("must be 1 or more value")
    except ValueError:
        raise RuntimeError(f"Format error: e.g. '2-', '2-5', or '-5': {field}")

    if s1 is None:
        slic = slice(None, s2)
    else:
        # -1 because of 1-based integer following "cut" command
        # e.g "1-3" -> slice(0, 3)
        slic = slice(s1 - 1, s2)
    return slic


def tokenize(
    input: str,
    output: str,
    field: Optional[str],
    delimiter: Optional[str],
    token_type: str,
    space_symbol: str,
    non_linguistic_symbols: Optional[str],
    bpemodel: Optional[str],
    log_level: str,
    write_vocabulary: bool,
    vocabulary_size: int,
    remove_non_linguistic_symbols: bool,
    cutoff: int,
    add_symbol: List[str],
    cleaner: Optional[str],
    g2p: Optional[str],
):

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    if input == "-":
        fin = sys.stdin
    else:
        fin = Path(input).open("r", encoding="utf-8")
    if output == "-":
        fout = sys.stdout
    else:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        fout = p.open("w", encoding="utf-8")

    cleaner = TextCleaner(cleaner)
    tokenizer = build_tokenizer(
        token_type=token_type,
        bpemodel=bpemodel,
        delimiter=delimiter,
        space_symbol=space_symbol,
        non_linguistic_symbols=non_linguistic_symbols,
        remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        g2p_type=g2p,
    )

    counter = Counter()
    if field is not None:
        field = field2slice(field)

    for line in fin:
        line = line.rstrip()
        if field is not None:
            # e.g. field="2-"
            # uttidA hello world!! -> hello world!!
            tokens = line.split(delimiter)
            tokens = tokens[field]
            if delimiter is None:
                line = " ".join(tokens)
            else:
                line = delimiter.join(tokens)

        line = cleaner(line)
        tokens = tokenizer.text2tokens(line)
        if not write_vocabulary:
            fout.write(" ".join(tokens) + "\n")
        else:
            for t in tokens:
                counter[t] += 1

    if not write_vocabulary:
        return

    ## FIXME
    ## del duplicate add_symbols in counter
    for symbol_and_id in add_symbol:
        # e.g symbol="<blank>:0"
        try:
            symbol, idx = symbol_and_id.split(":")
        except ValueError:
            raise RuntimeError(f"Format error: e.g. '<blank>:0': {symbol_and_id}")
        symbol = symbol.strip()
        if symbol in counter:
            del counter[symbol]

    # ======= write_vocabulary mode from here =======
    # Sort by the number of occurrences in descending order
    # and filter lower frequency words than cutoff value
    words_and_counts = list(
        filter(lambda x: x[1] > cutoff, sorted(counter.items(), key=lambda x: -x[1]))
    )
    # Restrict the vocabulary size
    if vocabulary_size > 0:
        if vocabulary_size < len(add_symbol):
            raise RuntimeError(f"vocabulary_size is too small: {vocabulary_size}")
        words_and_counts = words_and_counts[: vocabulary_size - len(add_symbol)]

    # Parse the values of --add_symbol
    for symbol_and_id in add_symbol:
        # e.g symbol="<blank>:0"
        try:
            symbol, idx = symbol_and_id.split(":")
            idx = int(idx)
        except ValueError:
            raise RuntimeError(f"Format error: e.g. '<blank>:0': {symbol_and_id}")
        symbol = symbol.strip()

        # e.g. idx=0  -> append as the first symbol
        # e.g. idx=-1 -> append as the last symbol
        if idx < 0:
            idx = len(words_and_counts) + 1 + idx
        words_and_counts.insert(idx, (symbol, None))

    # Write words
    for w, c in words_and_counts:
        fout.write(w + "\n")

    # Logging
    total_count = sum(counter.values())
    invocab_count = sum(c for w, c in words_and_counts if c is not None)
    logging.info(f"OOV rate = {(total_count - invocab_count) / total_count * 100} %")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize texts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--input", "-i", required=True, help="Input text. - indicates sys.stdin")
    parser.add_argument("--output", "-o", required=True, help="Output text. - indicates sys.stdout")
    parser.add_argument(
        "--field",
        "-f",
        help="The target columns of the input text as 1-based integer. e.g 2-",
    )
    parser.add_argument(
        "--token_type",
        "-t",
        default="char",
        choices=["char", "bpe", "word", "phn"],
        help="Token type",
    )
    parser.add_argument("--delimiter", "-d", default=None, help="The delimiter")
    parser.add_argument("--space_symbol", default="<space>", help="The space symbol")
    parser.add_argument("--bpemodel", default=None, help="The bpemodel file path")
    parser.add_argument(
        "--non_linguistic_symbols",
        type=str_or_none,
        help="non_linguistic_symbols file path",
    )
    parser.add_argument(
        "--remove_non_linguistic_symbols",
        type=str2bool,
        default=False,
        help="Remove non-language-symbols from tokens",
    )
    parser.add_argument(
        "--cleaner",
        type=str_or_none,
        choices=[None, "tacotron", "jaconv", "vietnamese", "korean_cleaner"],
        default=None,
        help="Apply text cleaning",
    )
    parser.add_argument(
        "--g2p",
        type=str_or_none,
        choices=g2p_classes,
        default=None,
        help="Specify g2p method if --token_type=phn",
    )

    group = parser.add_argument_group("write_vocabulary mode related")
    group.add_argument(
        "--write_vocabulary",
        type=str2bool,
        default=False,
        help="Write tokens list instead of tokenized text per line",
    )
    group.add_argument("--vocabulary_size", type=int, default=0, help="Vocabulary size")
    group.add_argument(
        "--cutoff",
        default=0,
        type=int,
        help="cut-off frequency used for write-vocabulary mode",
    )
    group.add_argument(
        "--add_symbol",
        type=str,
        default=[],
        action="append",
        help="Append symbol e.g. --add_symbol '<blank>:0' --add_symbol '<unk>:1'",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    tokenize(**kwargs)


if __name__ == "__main__":
    main()
