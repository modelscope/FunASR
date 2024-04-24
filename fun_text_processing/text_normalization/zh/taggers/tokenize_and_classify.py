import os

import pynini
from fun_text_processing.text_normalization.zh.graph_utils import FUN_SIGMA, GraphFst
from fun_text_processing.text_normalization.zh.taggers.cardinal import Cardinal
from fun_text_processing.text_normalization.zh.taggers.char import Char
from fun_text_processing.text_normalization.zh.taggers.date import Date
from fun_text_processing.text_normalization.zh.taggers.fraction import Fraction
from fun_text_processing.text_normalization.zh.taggers.math_symbol import MathSymbol
from fun_text_processing.text_normalization.zh.taggers.measure import Measure
from fun_text_processing.text_normalization.zh.taggers.money import Money
from fun_text_processing.text_normalization.zh.taggers.preprocessor import PreProcessor
from fun_text_processing.text_normalization.zh.taggers.time import Time
from fun_text_processing.text_normalization.zh.taggers.whitelist import Whitelist
from pynini.lib import pynutil


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir,
                f"zh_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
        else:
            date = Date(deterministic=deterministic)
            cardinal = Cardinal(deterministic=deterministic)
            char = Char(deterministic=deterministic)
            fraction = Fraction(deterministic=deterministic)
            math_symbol = MathSymbol(deterministic=deterministic)
            money = Money(deterministic=deterministic)
            measure = Measure(deterministic=deterministic)
            time = Time(deterministic=deterministic)
            whitelist = Whitelist(deterministic=deterministic)

            classify = pynini.union(
                pynutil.add_weight(date.fst, 1.02),
                pynutil.add_weight(fraction.fst, 1.05),
                pynutil.add_weight(money.fst, 1.05),
                pynutil.add_weight(measure.fst, 1.05),
                pynutil.add_weight(time.fst, 1.05),
                pynutil.add_weight(whitelist.fst, 1.03),
                pynutil.add_weight(cardinal.fst, 1.06),
                pynutil.add_weight(math_symbol.fst, 1.08),
                pynutil.add_weight(char.fst, 100),
            )
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" } ")

            tagger = pynini.cdrewrite(token.optimize(), "", "", FUN_SIGMA).optimize()

            preprocessor = PreProcessor(
                remove_interjections=True,
                fullwidth_to_halfwidth=True,
            )
            self.fst = preprocessor.fst @ tagger
