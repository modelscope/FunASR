#!/usr/bin/python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from time import perf_counter
from typing import List

from fun_text_processing.text_normalization.data_loader_utils import load_file, write_file
from fun_text_processing.text_normalization.normalize import Normalizer
from fun_text_processing.text_normalization.token_parser import TokenParser


class InverseNormalizer(Normalizer):
    """
    Inverse normalizer that converts text from spoken to written form. Useful for ASR postprocessing.
    Input is expected to have no punctuation outside of approstrophe (') and dash (-) and be lower cased.

    Args:
        lang: language specifying the ITN
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(
        self,
        lang: str = "en",
        cache_dir: str = None,
        overwrite_cache: bool = False,
        enable_standalone_number: bool = True,
        enable_0_to_9: bool = True,
    ):

        if lang == "en":
            from fun_text_processing.inverse_text_normalization.en.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.en.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == "id":
            from fun_text_processing.inverse_text_normalization.id.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.id.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == "ja":
            from fun_text_processing.inverse_text_normalization.ja.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.ja.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == "es":
            from fun_text_processing.inverse_text_normalization.es.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.es.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == "pt":
            from fun_text_processing.inverse_text_normalization.pt.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.pt.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == "ru":
            from fun_text_processing.inverse_text_normalization.ru.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.ru.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        elif lang == "de":
            from fun_text_processing.inverse_text_normalization.de.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.de.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == "fr":
            from fun_text_processing.inverse_text_normalization.fr.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.fr.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == "vi":
            from fun_text_processing.inverse_text_normalization.vi.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.vi.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == "ko":
            from fun_text_processing.inverse_text_normalization.ko.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.ko.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == "zh":
            from fun_text_processing.inverse_text_normalization.zh.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.zh.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == "tl":
            from fun_text_processing.inverse_text_normalization.tl.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.inverse_text_normalization.tl.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )

        self.tagger = ClassifyFst(cache_dir=cache_dir, overwrite_cache=overwrite_cache)
        self.verbalizer = VerbalizeFinalFst()
        self.parser = TokenParser()
        self.lang = lang
        self.convert_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9

    def inverse_normalize_list(self, texts: List[str], verbose=False) -> List[str]:
        """
        NeMo inverse text normalizer

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information

        Returns converted list of input strings
        """
        # print(texts)
        return self.normalize_list(texts=texts, verbose=verbose)

    def inverse_normalize(self, text: str, verbose: bool) -> str:
        """
        Main function. Inverse normalizes tokens from spoken to written form
            e.g. twelve kilograms -> 12 kg

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information

        Returns: written form
        """
        print(text)
        return self.normalize(text=text, verbose=verbose)


def str2bool(s, default=False):
    s = s.lower()
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        return default


def parse_args():
    parser = ArgumentParser()
    input = parser.add_mutually_exclusive_group()
    input.add_argument("--text", dest="input_string", help="input string", type=str)
    input.add_argument("--input_file", dest="input_file", help="input file path", type=str)
    parser.add_argument("--output_file", dest="output_file", help="output file path", type=str)
    parser.add_argument(
        "--language",
        help="language",
        choices=["en", "id", "ja", "de", "es", "pt", "ru", "fr", "vi", "ko", "zh", "tl"],
        default="en",
        type=str,
    )
    parser.add_argument("--verbose", help="print info for debugging", action="store_true")
    parser.add_argument(
        "--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true"
    )
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--enable_standalone_number", type=str, default="True", help="enable standalone number"
    )
    parser.add_argument(
        "--enable_0_to_9", type=str, default="True", help="enable convert number 0 to 9"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_time = perf_counter()
    if args.language == "ja":
        inverse_normalizer = InverseNormalizer(
            lang=args.language,
            cache_dir=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            enable_standalone_number=str2bool(args.enable_standalone_number),
            enable_0_to_9=str2bool(args.enable_0_to_9),
        )
    else:
        inverse_normalizer = InverseNormalizer(
            lang=args.language, cache_dir=args.cache_dir, overwrite_cache=args.overwrite_cache
        )
    print(f"Time to generate graph: {round(perf_counter() - start_time, 2)} sec")

    if args.input_string:
        print(inverse_normalizer.inverse_normalize(args.input_string, verbose=args.verbose))
    elif args.input_file:
        print("Loading data: " + args.input_file)
        data = load_file(args.input_file)

        print("- Data: " + str(len(data)) + " sentences")
        prediction = inverse_normalizer.inverse_normalize_list(data, verbose=args.verbose)
        if args.output_file:
            write_file(args.output_file, prediction)
            print(f"- Denormalized. Writing out to {args.output_file}")
        else:
            print(prediction)
