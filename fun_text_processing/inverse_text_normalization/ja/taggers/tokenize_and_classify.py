import os

import pynini
from fun_text_processing.inverse_text_normalization.ja.taggers.cardinal import CardinalFst
from fun_text_processing.inverse_text_normalization.ja.taggers.date import DateFst
from fun_text_processing.inverse_text_normalization.ja.taggers.decimal import DecimalFst
from fun_text_processing.inverse_text_normalization.ja.taggers.electronic import ElectronicFst
from fun_text_processing.inverse_text_normalization.ja.taggers.measure import MeasureFst
from fun_text_processing.inverse_text_normalization.ja.taggers.fraction import FractionFst
from fun_text_processing.inverse_text_normalization.ja.taggers.money import MoneyFst
from fun_text_processing.inverse_text_normalization.ja.taggers.ordinal import OrdinalFst
from fun_text_processing.inverse_text_normalization.ja.taggers.punctuation import PunctuationFst
from fun_text_processing.inverse_text_normalization.ja.taggers.telephone import TelephoneFst
from fun_text_processing.inverse_text_normalization.ja.taggers.time import TimeFst
from fun_text_processing.inverse_text_normalization.ja.taggers.whitelist import WhiteListFst
from fun_text_processing.inverse_text_normalization.ja.taggers.word import WordFst
from fun_text_processing.inverse_text_normalization.ja.taggers.preprocessor import PreProcessor
from fun_text_processing.inverse_text_normalization.ja.graph_utils import (
    GraphFst,
    DAMO_SIGMA,
    delete_extra_space,
    delete_space,
    generator_main,
    insert_space,
)
from pynini.lib import pynutil

import logging


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
    """

    def __init__(
        self,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        enable_standalone_number: bool = True,
        enable_0_to_9: bool = True,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify")
        self.convert_number = enable_standalone_number
        self.enable_0_to_9 = enable_0_to_9

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "_ja_itn.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            cardinal = CardinalFst(self.convert_number, self.enable_0_to_9)
            cardinal_graph = cardinal.fst

            fraction = FractionFst(cardinal)
            fraction_graph = fraction.fst

            ordinal = OrdinalFst(cardinal)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal)
            decimal_graph = decimal.fst

            measure_graph = MeasureFst(cardinal=cardinal, decimal=decimal).fst
            date_graph = DateFst(ordinal=ordinal).fst
            word_graph = WordFst().fst
            time_graph = TimeFst().fst
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal).fst
            whitelist_graph = WhiteListFst().fst
            punct_graph = PunctuationFst().fst
            preprocessor = PreProcessor(halfwidth_to_fullwidth=True).fst
            electronic_graph = ElectronicFst().fst
            telephone_graph = TelephoneFst(cardinal).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.09)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 1.1)
                | pynutil.add_weight(electronic_graph, 1.1)
                | pynutil.add_weight(word_graph, 100)
            )

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
                pynini.union(insert_space, delete_extra_space) + token_plus_punct
            )

            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")

            self.token_plus_punct = token_plus_punct.optimize()
