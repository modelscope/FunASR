import os

import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from fun_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from fun_text_processing.text_normalization.ru.taggers.cardinal import CardinalFst
from fun_text_processing.text_normalization.ru.taggers.date import DateFst
from fun_text_processing.text_normalization.ru.taggers.decimals import DecimalFst
from fun_text_processing.text_normalization.ru.taggers.electronic import ElectronicFst
from fun_text_processing.text_normalization.ru.taggers.measure import MeasureFst
from fun_text_processing.text_normalization.ru.taggers.money import MoneyFst
from fun_text_processing.text_normalization.ru.taggers.number_names import (
    get_alternative_formats,
    get_number_names,
)
from fun_text_processing.text_normalization.ru.taggers.ordinal import OrdinalFst
from fun_text_processing.text_normalization.ru.taggers.telephone import TelephoneFst
from fun_text_processing.text_normalization.ru.taggers.time import TimeFst
from fun_text_processing.text_normalization.ru.taggers.whitelist import WhiteListFst
from fun_text_processing.text_normalization.ru.taggers.word import WordFst
from pynini.lib import pynutil

import logging


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
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
        if deterministic:
            raise ValueError(
                "Ru TN only supports non-deterministic cases and produces multiple normalization options."
            )
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"_{input_case}_ru_tn_{deterministic}_deterministic{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars. This might take some time...")
            number_names = get_number_names()
            alternative_formats = get_alternative_formats()

            self.cardinal = CardinalFst(
                number_names=number_names,
                alternative_formats=alternative_formats,
                deterministic=deterministic,
            )
            cardinal_graph = self.cardinal.fst

            self.ordinal = OrdinalFst(
                number_names=number_names,
                alternative_formats=alternative_formats,
                deterministic=deterministic,
            )
            ordinal_graph = self.ordinal.fst

            self.decimal = DecimalFst(cardinal=self.cardinal, deterministic=deterministic)
            decimal_graph = self.decimal.fst

            self.measure = MeasureFst(
                cardinal=self.cardinal, decimal=self.decimal, deterministic=deterministic
            )
            measure_graph = self.measure.fst
            self.date = DateFst(number_names=number_names, deterministic=deterministic)
            date_graph = self.date.fst
            word_graph = WordFst(deterministic=deterministic).fst
            self.time = TimeFst(number_names=number_names, deterministic=deterministic)
            time_graph = self.time.fst
            self.telephone = TelephoneFst(number_names=number_names, deterministic=deterministic)
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
                | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(measure_graph, 0.9)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
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
                pynutil.add_weight(delete_extra_space, 1.1) + token_plus_punct
            )
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
