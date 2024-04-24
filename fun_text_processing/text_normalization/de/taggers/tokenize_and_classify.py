import os

import pynini
from fun_text_processing.text_normalization.de.taggers.cardinal import CardinalFst
from fun_text_processing.text_normalization.de.taggers.date import DateFst
from fun_text_processing.text_normalization.de.taggers.decimal import DecimalFst
from fun_text_processing.text_normalization.de.taggers.electronic import ElectronicFst
from fun_text_processing.text_normalization.de.taggers.fraction import FractionFst
from fun_text_processing.text_normalization.de.taggers.measure import MeasureFst
from fun_text_processing.text_normalization.de.taggers.money import MoneyFst
from fun_text_processing.text_normalization.de.taggers.ordinal import OrdinalFst
from fun_text_processing.text_normalization.de.taggers.telephone import TelephoneFst
from fun_text_processing.text_normalization.de.taggers.time import TimeFst
from fun_text_processing.text_normalization.de.taggers.whitelist import WhiteListFst
from fun_text_processing.text_normalization.de.taggers.word import WordFst
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from fun_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from pynini.lib import pynutil

import logging


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
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
        deterministic: bool = False,
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
                cache_dir, f"_{input_case}_de_tn_{deterministic}_deterministic{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            no_digits = pynini.closure(pynini.difference(DAMO_CHAR, DAMO_DIGIT))
            self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars. This might take some time...")

            self.cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = self.cardinal.fst

            self.ordinal = OrdinalFst(cardinal=self.cardinal, deterministic=deterministic)
            ordinal_graph = self.ordinal.fst

            self.decimal = DecimalFst(cardinal=self.cardinal, deterministic=deterministic)
            decimal_graph = self.decimal.fst

            self.fraction = FractionFst(cardinal=self.cardinal, deterministic=deterministic)
            fraction_graph = self.fraction.fst
            self.measure = MeasureFst(
                cardinal=self.cardinal,
                decimal=self.decimal,
                fraction=self.fraction,
                deterministic=deterministic,
            )
            measure_graph = self.measure.fst
            self.date = DateFst(cardinal=self.cardinal, deterministic=deterministic)
            date_graph = self.date.fst
            word_graph = WordFst(deterministic=deterministic).fst
            self.time = TimeFst(deterministic=deterministic)
            time_graph = self.time.fst
            self.telephone = TelephoneFst(cardinal=self.cardinal, deterministic=deterministic)
            telephone_graph = self.telephone.fst
            self.electronic = ElectronicFst(deterministic=deterministic)
            electronic_graph = self.electronic.fst
            self.money = MoneyFst(
                cardinal=self.cardinal, decimal=self.decimal, deterministic=deterministic
            )
            money_graph = self.money.fst
            self.whitelist = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            )
            whitelist_graph = self.whitelist.fst
            punct_graph = PunctuationFst(deterministic=deterministic).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 1.1)
                | pynutil.add_weight(electronic_graph, 1.1)
            )

            classify |= pynutil.add_weight(word_graph, 100)

            punct = (
                pynutil.insert("tokens { ")
                + pynutil.add_weight(punct_graph, weight=1.1)
                + pynutil.insert(" }")
            )
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + token
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                pynutil.add_weight(delete_extra_space, 1.1) + token_plus_punct
            )
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()
            no_digits = pynini.closure(pynini.difference(DAMO_CHAR, DAMO_DIGIT))
            self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
