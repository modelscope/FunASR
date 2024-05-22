import os

import pynini
from fun_text_processing.text_normalization.zh.graph_utils import (
    GraphFst,
    delete_space,
    generator_main,
)
from fun_text_processing.text_normalization.zh.verbalizers.postprocessor import PostProcessor
from fun_text_processing.text_normalization.zh.verbalizers.verbalize import VerbalizeFst
from pynini.lib import pynutil

# import logging


class VerbalizeFinalFst(GraphFst):
    """ """

    def __init__(
        self, deterministic: bool = True, cache_dir: str = None, overwrite_cache: bool = False
    ):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(
                cache_dir, f"zh_tn_{deterministic}_deterministic_verbalizer.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["verbalize"]
        else:
            token_graph = VerbalizeFst(deterministic=deterministic)
            token_verbalizer = (
                pynutil.delete("tokens {")
                + delete_space
                + token_graph.fst
                + delete_space
                + pynutil.delete(" }")
            )
            verbalizer = pynini.closure(delete_space + token_verbalizer + delete_space)

            postprocessor = PostProcessor(
                remove_puncts=False,
                to_upper=False,
                to_lower=False,
                tag_oov=False,
            )

            self.fst = (verbalizer @ postprocessor.fst).optimize()
            if far_file:
                generator_main(far_file, {"verbalize": self.fst})
