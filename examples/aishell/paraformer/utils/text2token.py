#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import argparse
import codecs
import re
import sys
import json

is_python2 = sys.version_info[0] == 2


def exist_or_not(i, match_pos):
    start_pos = None
    end_pos = None
    for pos in match_pos:
        if pos[0] <= i < pos[1]:
            start_pos = pos[0]
            end_pos = pos[1]
            break

    return start_pos, end_pos


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert raw text to tokenized text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nchar",
        "-n",
        default=1,
        type=int,
        help="number of characters to split, i.e., \
                        aabb -> a a b b with -n 1 and aa bb with -n 2",
    )
    parser.add_argument("--skip-ncols", "-s", default=0, type=int, help="skip first n columns")
    parser.add_argument("--space", default="<space>", type=str, help="space symbol")
    parser.add_argument(
        "--non-lang-syms",
        "-l",
        default=None,
        type=str,
        help="list of non-linguistic symobles, e.g., <NOISE> etc.",
    )
    parser.add_argument("text", type=str, default=False, nargs="?", help="input text")
    parser.add_argument(
        "--trans_type",
        "-t",
        type=str,
        default="char",
        choices=["char", "phn"],
        help="""Transcript type. char/phn. e.g., for TIMIT FADG0_SI1279 -
                        If trans_type is char,
                        read from SI1279.WRD file -> "bricks are an alternative"
                        Else if trans_type is phn,
                        read from SI1279.PHN file -> "sil b r ih sil k s aa r er n aa l
                        sil t er n ih sil t ih v sil" """,
    )
    parser.add_argument(
        "--text_format",
        default="text",
        type=str,
        help="text, jsonl",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    rs = []
    if args.non_lang_syms is not None:
        with codecs.open(args.non_lang_syms, "r", encoding="utf-8") as f:
            nls = [x.rstrip() for x in f.readlines()]
            rs = [re.compile(re.escape(x)) for x in nls]

    if args.text:
        f = codecs.open(args.text, encoding="utf-8")
    else:
        f = codecs.getreader("utf-8")(sys.stdin if is_python2 else sys.stdin.buffer)

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout if is_python2 else sys.stdout.buffer)
    line = f.readline()
    n = args.nchar
    while line:
        if args.text_format == "jsonl":
            data = json.loads(line.strip())
            line = data["target"]
        x = line.split()
        print(" ".join(x[: args.skip_ncols]), end=" ")
        a = " ".join(x[args.skip_ncols :])

        # get all matched positions
        match_pos = []
        for r in rs:
            i = 0
            while i >= 0:
                m = r.search(a, i)
                if m:
                    match_pos.append([m.start(), m.end()])
                    i = m.end()
                else:
                    break

        if args.trans_type == "phn":
            a = a.split(" ")
        else:
            if len(match_pos) > 0:
                chars = []
                i = 0
                while i < len(a):
                    start_pos, end_pos = exist_or_not(i, match_pos)
                    if start_pos is not None:
                        chars.append(a[start_pos:end_pos])
                        i = end_pos
                    else:
                        chars.append(a[i])
                        i += 1
                a = chars

            a = [a[j : j + n] for j in range(0, len(a), n)]

        a_flat = []
        for z in a:
            a_flat.append("".join(z))

        a_chars = [z.replace(" ", args.space) for z in a_flat]
        if args.trans_type == "phn":
            a_chars = [z.replace("sil", args.space) for z in a_chars]
        print(" ".join(a_chars))
        line = f.readline()


if __name__ == "__main__":
    main()
