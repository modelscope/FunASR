import itertools
import os
import re
from argparse import ArgumentParser
from collections import OrderedDict
from math import factorial
from time import perf_counter
from typing import Dict, List, Union

import pynini
import regex
from joblib import Parallel, delayed
from fun_text_processing.text_normalization.data_loader_utils import (
    load_file,
    post_process_punct,
    pre_process,
    write_file,
)
from fun_text_processing.text_normalization.token_parser import PRESERVE_ORDER_KEY, TokenParser
from pynini.lib.rewrite import top_rewrite
from tqdm import tqdm

try:
    from nemo.collections.common.tokenizers.moses_tokenizers import MosesProcessor

    NLP_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e:
    NLP_AVAILABLE = False

SPACE_DUP = re.compile(" {2,}")


class Normalizer:
    """
    Normalizer class that converts text from written to spoken form.
    Useful for TTS preprocessing.

    Args:
        input_case: expected input capitalization
        lang: language specifying the TN rules, by default: English
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
        post_process: WFST-based post processing, e.g. to remove extra spaces added during TN.
            Note: punct_post_process flag in normalize() supports all languages.
    """

    def __init__(
        self,
        input_case: str,
        lang: str = "en",
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
        lm: bool = False,
        post_process: bool = True,
    ):
        assert input_case in ["lower_cased", "cased"]

        self.post_processor = None

        if lang == "en":
            from fun_text_processing.text_normalization.en.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
            from fun_text_processing.text_normalization.en.verbalizers.post_processing import (
                PostProcessingFst,
            )

            if post_process:
                self.post_processor = PostProcessingFst(
                    cache_dir=cache_dir, overwrite_cache=overwrite_cache
                )

            if deterministic:
                from fun_text_processing.text_normalization.en.taggers.tokenize_and_classify import (
                    ClassifyFst,
                )
            else:
                if lm:
                    from fun_text_processing.text_normalization.en.taggers.tokenize_and_classify_lm import (
                        ClassifyFst,
                    )
                else:
                    from fun_text_processing.text_normalization.en.taggers.tokenize_and_classify_with_audio import (
                        ClassifyFst,
                    )

        elif lang == "ru":
            # Ru TN only support non-deterministic cases and produces multiple normalization options
            # use normalize_with_audio.py
            from fun_text_processing.text_normalization.ru.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.text_normalization.ru.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == "de":
            from fun_text_processing.text_normalization.de.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.text_normalization.de.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == "es":
            from fun_text_processing.text_normalization.es.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.text_normalization.es.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        elif lang == "zh":
            from fun_text_processing.text_normalization.zh.taggers.tokenize_and_classify import (
                ClassifyFst,
            )
            from fun_text_processing.text_normalization.zh.verbalizers.verbalize_final import (
                VerbalizeFinalFst,
            )
        self.tagger = ClassifyFst(
            input_case=input_case,
            deterministic=deterministic,
            cache_dir=cache_dir,
            overwrite_cache=overwrite_cache,
            whitelist=whitelist,
        )

        self.verbalizer = VerbalizeFinalFst(
            deterministic=deterministic, cache_dir=cache_dir, overwrite_cache=overwrite_cache
        )

        self.parser = TokenParser()
        self.lang = lang

        if NLP_AVAILABLE:
            self.processor = MosesProcessor(lang_id=lang)
        else:
            self.processor = None
            print("NeMo NLP is not available. Moses de-tokenization will be skipped.")

    def normalize_list(
        self,
        texts: List[str],
        verbose: bool = False,
        punct_pre_process: bool = False,
        punct_post_process: bool = False,
        batch_size: int = 1,
        n_jobs: int = 1,
    ):
        """
        NeMo text normalizer

        Args:
            texts: list of input strings
            verbose: whether to print intermediate meta information
            punct_pre_process: whether to do punctuation pre processing
            punct_post_process: whether to do punctuation post processing
            n_jobs: the maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given,
                no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1,
                (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
            batch_size: Number of examples for each process

        Returns converted list input strings
        """

        # to save intermediate results to a file
        batch = min(len(texts), batch_size)

        try:
            normalized_texts = Parallel(n_jobs=n_jobs)(
                delayed(self.process_batch)(
                    texts[i : i + batch], verbose, punct_pre_process, punct_post_process
                )
                for i in range(0, len(texts), batch)
            )
        except BaseException as e:
            raise e

        normalized_texts = list(itertools.chain(*normalized_texts))
        return normalized_texts

    def process_batch(self, batch, verbose, punct_pre_process, punct_post_process):
        """
        Normalizes batch of text sequences
        Args:
            batch: list of texts
            verbose: whether to print intermediate meta information
            punct_pre_process: whether to do punctuation pre processing
            punct_post_process: whether to do punctuation post processing
        """
        normalized_lines = [
            self.normalize(
                text,
                verbose=verbose,
                punct_pre_process=punct_pre_process,
                punct_post_process=punct_post_process,
            )
            for text in tqdm(batch)
        ]
        return normalized_lines

    def _estimate_number_of_permutations_in_nested_dict(
        self, token_group: Dict[str, Union[OrderedDict, str, bool]]
    ) -> int:
        num_perms = 1
        for k, inner in token_group.items():
            if isinstance(inner, dict):
                num_perms *= self._estimate_number_of_permutations_in_nested_dict(inner)
        num_perms *= factorial(len(token_group))
        return num_perms

    def _split_tokens_to_reduce_number_of_permutations(
        self, tokens: List[dict], max_number_of_permutations_per_split: int = 729
    ) -> List[List[dict]]:
        """
        Splits a sequence of tokens in a smaller sequences of tokens in a way that maximum number of composite
        tokens permutations does not exceed ``max_number_of_permutations_per_split``.

        For example,

        .. code-block:: python
            tokens = [
                {"tokens": {"date": {"year": "twenty eighteen", "month": "december", "day": "thirty one"}}},
                {"tokens": {"date": {"year": "twenty eighteen", "month": "january", "day": "eight"}}},
            ]
            split = normalizer._split_tokens_to_reduce_number_of_permutations(
                tokens, max_number_of_permutations_per_split=6
            )
            assert split == [
                [{"tokens": {"date": {"year": "twenty eighteen", "month": "december", "day": "thirty one"}}}],
                [{"tokens": {"date": {"year": "twenty eighteen", "month": "january", "day": "eight"}}}],
            ]

        Date tokens contain 3 items each which gives 6 permutations for every date. Since there are 2 dates, total
        number of permutations would be ``6 * 6 == 36``. Parameter ``max_number_of_permutations_per_split`` equals 6,
        so input sequence of tokens is split into 2 smaller sequences.

        Args:
            tokens (:obj:`List[dict]`): a list of dictionaries, possibly nested.
            max_number_of_permutations_per_split (:obj:`int`, `optional`, defaults to :obj:`243`): a maximum number
                of permutations which can be generated from input sequence of tokens.

        Returns:
            :obj:`List[List[dict]]`: a list of smaller sequences of tokens resulting from ``tokens`` split.
        """
        splits = []
        prev_end_of_split = 0
        current_number_of_permutations = 1
        for i, token_group in enumerate(tokens):
            n = self._estimate_number_of_permutations_in_nested_dict(token_group)
            if n * current_number_of_permutations > max_number_of_permutations_per_split:
                splits.append(tokens[prev_end_of_split:i])
                prev_end_of_split = i
                current_number_of_permutations = 1
            if n > max_number_of_permutations_per_split:
                raise ValueError(
                    f"Could not split token list with respect to condition that every split can generate number of "
                    f"permutations less or equal to "
                    f"`max_number_of_permutations_per_split={max_number_of_permutations_per_split}`. "
                    f"There is an unsplittable token group that generates more than "
                    f"{max_number_of_permutations_per_split} permutations. Try to increase "
                    f"`max_number_of_permutations_per_split` parameter."
                )
            current_number_of_permutations *= n
        splits.append(tokens[prev_end_of_split:])
        assert sum([len(s) for s in splits]) == len(tokens)
        return splits

    def normalize(
        self,
        text: str,
        verbose: bool = False,
        punct_pre_process: bool = False,
        punct_post_process: bool = False,
    ) -> str:
        """
        Main function. Normalizes tokens from written to spoken form
            e.g. 12 kg -> twelve kilograms

        Args:
            text: string that may include semiotic classes
            verbose: whether to print intermediate meta information
            punct_pre_process: whether to perform punctuation pre-processing, for example, [25] -> [ 25 ]
            punct_post_process: whether to normalize punctuation

        Returns: spoken form
        """
        if len(text.split()) > 500:
            print(
                "WARNING! Your input is too long and could take a long time to normalize."
                "Use split_text_into_sentences() to make the input shorter and then call normalize_list()."
            )

        original_text = text
        if punct_pre_process:
            text = pre_process(text)
        text = text.strip()
        if not text:
            if verbose:
                print(text)
            return text
        text = pynini.escape(text)
        tagged_lattice = self.find_tags(text)
        tagged_text = self.select_tag(tagged_lattice)
        if verbose:
            print(tagged_text)
        self.parser(tagged_text)
        tokens = self.parser.parse()
        split_tokens = self._split_tokens_to_reduce_number_of_permutations(tokens)
        output = ""
        for s in split_tokens:
            tags_reordered = self.generate_permutations(s)
            verbalizer_lattice = None
            for tagged_text in tags_reordered:
                tagged_text = pynini.escape(tagged_text)

                verbalizer_lattice = self.find_verbalizer(tagged_text)
                if verbalizer_lattice.num_states() != 0:
                    break
            if verbalizer_lattice is None:
                raise ValueError(f"No permutations were generated from tokens {s}")
            output += " " + self.select_verbalizer(verbalizer_lattice)
        output = SPACE_DUP.sub(" ", output[1:])

        if self.lang == "en" and hasattr(self, "post_processor"):
            output = self.post_process(output)

        if punct_post_process:
            # do post-processing based on Moses detokenizer
            if self.processor:
                output = self.processor.moses_detokenizer.detokenize([output], unescape=False)
                output = post_process_punct(input=original_text, normalized_text=output)
            else:
                print("DAMO_NLP collection is not available: skipping punctuation post_processing")

        return output

    def split_text_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: text

        Returns list of sentences
        """
        lower_case_unicode = ""
        upper_case_unicode = ""
        if self.lang == "ru":
            lower_case_unicode = "\u0430-\u04FF"
            upper_case_unicode = "\u0410-\u042F"

        # Read and split transcript by utterance (roughly, sentences)
        split_pattern = f"(?<!\w\.\w.)(?<![A-Z{upper_case_unicode}][a-z{lower_case_unicode}]+\.)(?<![A-Z{upper_case_unicode}]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s(?![0-9]+[a-z]*\.)"

        sentences = regex.split(split_pattern, text)
        return sentences

    def _permute(self, d: OrderedDict) -> List[str]:
        """
        Creates reorderings of dictionary elements and serializes as strings

        Args:
            d: (nested) dictionary of key value pairs

        Return permutations of different string serializations of key value pairs
        """
        l = []
        if PRESERVE_ORDER_KEY in d.keys():
            d_permutations = [d.items()]
        else:
            d_permutations = itertools.permutations(d.items())
        for perm in d_permutations:
            subl = [""]
            for k, v in perm:
                if isinstance(v, str):
                    subl = ["".join(x) for x in itertools.product(subl, [f'{k}: "{v}" '])]
                elif isinstance(v, OrderedDict):
                    rec = self._permute(v)
                    subl = [
                        "".join(x) for x in itertools.product(subl, [f" {k} {{ "], rec, [f" }} "])
                    ]
                elif isinstance(v, bool):
                    subl = ["".join(x) for x in itertools.product(subl, [f"{k}: true "])]
                else:
                    raise ValueError()
            l.extend(subl)
        return l

    def generate_permutations(self, tokens: List[dict]):
        """
        Generates permutations of string serializations of list of dictionaries

        Args:
            tokens: list of dictionaries

        Returns string serialization of list of dictionaries
        """

        def _helper(prefix: str, tokens: List[dict], idx: int):
            """
            Generates permutations of string serializations of given dictionary

            Args:
                tokens: list of dictionaries
                prefix: prefix string
                idx:    index of next dictionary

            Returns string serialization of dictionary
            """
            if idx == len(tokens):
                yield prefix
                return
            token_options = self._permute(tokens[idx])
            for token_option in token_options:
                yield from _helper(prefix + token_option, tokens, idx + 1)

        return _helper("", tokens, 0)

    def find_tags(self, text: str) -> "pynini.FstLike":
        """
        Given text use tagger Fst to tag text

        Args:
            text: sentence

        Returns: tagged lattice
        """
        lattice = text @ self.tagger.fst
        return lattice

    def select_tag(self, lattice: "pynini.FstLike") -> str:
        """
        Given tagged lattice return shortest path

        Args:
            tagged_text: tagged text

        Returns: shortest path
        """
        tagged_text = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return tagged_text

    def find_verbalizer(self, tagged_text: str) -> "pynini.FstLike":
        """
        Given tagged text creates verbalization lattice
        This is context-independent.

        Args:
            tagged_text: input text

        Returns: verbalized lattice
        """
        lattice = tagged_text @ self.verbalizer.fst
        return lattice

    def select_verbalizer(self, lattice: "pynini.FstLike") -> str:
        """
        Given verbalized lattice return shortest path

        Args:
            lattice: verbalization lattice

        Returns: shortest path
        """
        output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        # lattice = output @ self.verbalizer.punct_graph
        # output = pynini.shortestpath(lattice, nshortest=1, unique=True).string()
        return output

    def post_process(self, normalized_text: "pynini.FstLike") -> str:
        """
        Runs post processing graph on normalized text

        Args:
            normalized_text: normalized text

        Returns: shortest path
        """
        normalized_text = normalized_text.strip()
        if not normalized_text:
            return normalized_text
        normalized_text = pynini.escape(normalized_text)

        if self.post_processor is not None:
            normalized_text = top_rewrite(normalized_text, self.post_processor.fst)
        return normalized_text


def parse_args():
    parser = ArgumentParser()
    input = parser.add_mutually_exclusive_group()
    input.add_argument("--text", dest="input_string", help="input string", type=str)
    input.add_argument("--input_file", dest="input_file", help="input file path", type=str)
    parser.add_argument("--output_file", dest="output_file", help="output file path", type=str)
    parser.add_argument(
        "--language", help="language", choices=["en", "de", "es", "zh"], default="en", type=str
    )
    parser.add_argument(
        "--input_case",
        help="input capitalization",
        choices=["lower_cased", "cased"],
        default="cased",
        type=str,
    )
    parser.add_argument("--verbose", help="print info for debugging", action="store_true")
    parser.add_argument(
        "--punct_post_process",
        help="set to True to enable punctuation post processing to match input.",
        action="store_true",
    )
    parser.add_argument(
        "--punct_pre_process",
        help="set to True to enable punctuation pre processing",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite_cache", help="set to True to re-create .far grammar files", action="store_true"
    )
    parser.add_argument(
        "--whitelist", help="path to a file with with whitelist", default=None, type=str
    )
    parser.add_argument(
        "--cache_dir",
        help="path to a dir with .far grammar file. Set to None to avoid using cache",
        default=None,
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    start_time = perf_counter()

    args = parse_args()
    whitelist = os.path.abspath(args.whitelist) if args.whitelist else None

    if not args.input_string and not args.input_file:
        raise ValueError("Either `--text` or `--input_file` required")

    normalizer = Normalizer(
        input_case=args.input_case,
        cache_dir=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        whitelist=whitelist,
        lang=args.language,
    )
    if args.input_string:
        print(
            normalizer.normalize(
                args.input_string,
                verbose=args.verbose,
                punct_pre_process=args.punct_pre_process,
                punct_post_process=args.punct_post_process,
            )
        )
    elif args.input_file:
        print("Loading data: " + args.input_file)
        data = load_file(args.input_file)

        print("- Data: " + str(len(data)) + " sentences")
        normalizer_prediction = normalizer.normalize_list(
            data,
            verbose=args.verbose,
            punct_pre_process=args.punct_pre_process,
            punct_post_process=args.punct_post_process,
        )
        if args.output_file:
            write_file(args.output_file, normalizer_prediction)
            print(f"- Normalized. Writing out to {args.output_file}")
        else:
            print(normalizer_prediction)

    print(f"Execution time: {perf_counter() - start_time:.02f} sec")
