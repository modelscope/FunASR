import os

import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    MIN_NEG_WEIGHT,
    DAMO_ALPHA,
    DAMO_CHAR,
    DAMO_SIGMA,
    DAMO_SPACE,
    generator_main,
)
from fun_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from pynini.lib import pynutil

import logging


class PostProcessingFst:
    """
    Finite state transducer that post-processing an entire sentence after verbalization is complete, e.g.
    removes extra spaces around punctuation marks " ( one hundred and twenty three ) " -> "(one hundred and twenty three)"

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(self, cache_dir: str = None, overwrite_cache: bool = False):

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "en_tn_post_processing.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["post_process_graph"]
            logging.info(f"Post processing graph was restored from {far_file}.")
        else:
            self.set_punct_dict()
            self.fst = self.get_punct_postprocess_graph()

            if far_file:
                generator_main(far_file, {"post_process_graph": self.fst})

    def set_punct_dict(self):
        self.punct_marks = {
            "'": [
                "'",
                "´",
                "ʹ",
                "ʻ",
                "ʼ",
                "ʽ",
                "ʾ",
                "ˈ",
                "ˊ",
                "ˋ",
                "˴",
                "ʹ",
                "΄",
                "՚",
                "՝",
                "י",
                "׳",
                "ߴ",
                "ߵ",
                "ᑊ",
                "ᛌ",
                "᾽",
                "᾿",
                "`",
                "´",
                "῾",
                "‘",
                "’",
                "‛",
                "′",
                "‵",
                "ꞌ",
                "＇",
                "｀",
                "𖽑",
                "𖽒",
            ],
        }

    def get_punct_postprocess_graph(self):
        """
        Returns graph to post process punctuation marks.

        {``} quotes are converted to {"}. Note, if there are spaces around single quote {'}, they will be kept.
        By default, a space is added after a punctuation mark, and spaces are removed before punctuation marks.
        """
        punct_marks_all = PunctuationFst().punct_marks

        # no_space_before_punct assume no space before them
        quotes = ["'", '"', "``", "«"]
        dashes = ["-", "—"]
        brackets = ["<", "{", "("]
        open_close_single_quotes = [
            ("`", "`"),
        ]

        open_close_double_quotes = [('"', '"'), ("``", "``"), ("“", "”")]
        open_close_symbols = open_close_single_quotes + open_close_double_quotes
        allow_space_before_punct = (
            ["&"] + quotes + dashes + brackets + [k[0] for k in open_close_symbols]
        )

        no_space_before_punct = [m for m in punct_marks_all if m not in allow_space_before_punct]
        no_space_before_punct = pynini.union(*no_space_before_punct)
        no_space_after_punct = pynini.union(*brackets)
        delete_space = pynutil.delete(" ")
        delete_space_optional = pynini.closure(delete_space, 0, 1)

        # non_punct allows space
        # delete space before no_space_before_punct marks, if present
        non_punct = pynini.difference(DAMO_CHAR, no_space_before_punct).optimize()
        graph = (
            pynini.closure(non_punct)
            + pynini.closure(
                no_space_before_punct
                | pynutil.add_weight(delete_space + no_space_before_punct, MIN_NEG_WEIGHT)
            )
            + pynini.closure(non_punct)
        )
        graph = pynini.closure(graph).optimize()
        graph = pynini.compose(
            graph, pynini.cdrewrite(pynini.cross("``", '"'), "", "", DAMO_SIGMA).optimize()
        ).optimize()

        # remove space after no_space_after_punct (even if there are no matching closing brackets)
        no_space_after_punct = pynini.cdrewrite(
            delete_space, no_space_after_punct, DAMO_SIGMA, DAMO_SIGMA
        ).optimize()
        graph = pynini.compose(graph, no_space_after_punct).optimize()

        # remove space around text in quotes
        single_quote = pynutil.add_weight(pynini.accep("`"), MIN_NEG_WEIGHT)
        double_quotes = pynutil.add_weight(pynini.accep('"'), MIN_NEG_WEIGHT)
        quotes_graph = (
            single_quote
            + delete_space_optional
            + DAMO_ALPHA
            + DAMO_SIGMA
            + delete_space_optional
            + single_quote
        ).optimize()

        # this is to make sure multiple quotes are tagged from right to left without skipping any quotes in the left
        not_alpha = pynini.difference(DAMO_CHAR, DAMO_ALPHA).optimize() | pynutil.add_weight(
            DAMO_SPACE, MIN_NEG_WEIGHT
        )
        end = pynini.closure(pynutil.add_weight(not_alpha, MIN_NEG_WEIGHT))
        quotes_graph |= (
            double_quotes
            + delete_space_optional
            + DAMO_ALPHA
            + DAMO_SIGMA
            + delete_space_optional
            + double_quotes
            + end
        )

        quotes_graph = pynutil.add_weight(quotes_graph, MIN_NEG_WEIGHT)
        quotes_graph = DAMO_SIGMA + pynini.closure(DAMO_SIGMA + quotes_graph + DAMO_SIGMA)

        graph = pynini.compose(graph, quotes_graph).optimize()

        # remove space between a word and a single quote followed by s
        remove_space_around_single_quote = pynini.cdrewrite(
            delete_space_optional + pynini.union(*self.punct_marks["'"]) + delete_space,
            DAMO_ALPHA,
            pynini.union("s ", "s[EOS]"),
            DAMO_SIGMA,
        )

        graph = pynini.compose(graph, remove_space_around_single_quote).optimize()
        return graph
