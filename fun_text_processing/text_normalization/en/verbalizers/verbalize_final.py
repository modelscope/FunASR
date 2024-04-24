import os

import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from fun_text_processing.text_normalization.en.verbalizers.verbalize import VerbalizeFst
from fun_text_processing.text_normalization.en.verbalizers.word import WordFst
from pynini.lib import pynutil

import logging


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence, e.g.
    tokens { name: "its" } tokens { time { hours: "twelve" minutes: "thirty" } } tokens { name: "now" } tokens { name: "." } -> its twelve thirty now .

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(
        self, deterministic: bool = True, cache_dir: str = None, overwrite_cache: bool = False
    ):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(
                cache_dir, f"en_tn_{deterministic}_deterministic_verbalizer.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
            logging.info(f"VerbalizeFinalFst graph was restored from {far_file}.")
        else:
            verbalize = VerbalizeFst(deterministic=deterministic).fst
            word = WordFst(deterministic=deterministic).fst
            types = verbalize | word

            if deterministic:
                graph = (
                    pynutil.delete("tokens")
                    + delete_space
                    + pynutil.delete("{")
                    + delete_space
                    + types
                    + delete_space
                    + pynutil.delete("}")
                )
            else:
                graph = delete_space + types + delete_space

            graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space

            self.fst = graph.optimize()
            if far_file:
                generator_main(far_file, {"verbalize": self.fst})
                logging.info(f"VerbalizeFinalFst grammars are saved to {far_file}.")
