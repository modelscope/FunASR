import os
import time

import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from fun_text_processing.text_normalization.en.taggers.abbreviation import AbbreviationFst
from fun_text_processing.text_normalization.en.taggers.cardinal import CardinalFst
from fun_text_processing.text_normalization.en.taggers.date import DateFst
from fun_text_processing.text_normalization.en.taggers.decimal import DecimalFst
from fun_text_processing.text_normalization.en.taggers.electronic import ElectronicFst
from fun_text_processing.text_normalization.en.taggers.fraction import FractionFst
from fun_text_processing.text_normalization.en.taggers.measure import MeasureFst
from fun_text_processing.text_normalization.en.taggers.money import MoneyFst
from fun_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst
from fun_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from fun_text_processing.text_normalization.en.taggers.range import RangeFst as RangeFst
from fun_text_processing.text_normalization.en.taggers.roman import RomanFst
from fun_text_processing.text_normalization.en.taggers.serial import SerialFst
from fun_text_processing.text_normalization.en.taggers.telephone import TelephoneFst
from fun_text_processing.text_normalization.en.taggers.time import TimeFst
from fun_text_processing.text_normalization.en.taggers.whitelist import WhiteListFst
from fun_text_processing.text_normalization.en.taggers.word import WordFst
from fun_text_processing.text_normalization.en.verbalizers.date import DateFst as vDateFst
from fun_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as vOrdinalFst
from fun_text_processing.text_normalization.en.verbalizers.time import TimeFst as vTimeFst
from pynini.lib import pynutil

import logging


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
                f"en_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize.far",
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")

            start_time = time.time()
            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst
            logging.debug(
                f"cardinal: {time.time() - start_time: .2f}s -- {cardinal_graph.num_states()} nodes"
            )

            start_time = time.time()
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst
            logging.debug(
                f"ordinal: {time.time() - start_time: .2f}s -- {ordinal_graph.num_states()} nodes"
            )

            start_time = time.time()
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst
            logging.debug(
                f"decimal: {time.time() - start_time: .2f}s -- {decimal_graph.num_states()} nodes"
            )

            start_time = time.time()
            fraction = FractionFst(deterministic=deterministic, cardinal=cardinal)
            fraction_graph = fraction.fst
            logging.debug(
                f"fraction: {time.time() - start_time: .2f}s -- {fraction_graph.num_states()} nodes"
            )

            start_time = time.time()
            measure = MeasureFst(
                cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic
            )
            measure_graph = measure.fst
            logging.debug(
                f"measure: {time.time() - start_time: .2f}s -- {measure_graph.num_states()} nodes"
            )

            start_time = time.time()
            date_graph = DateFst(cardinal=cardinal, deterministic=deterministic).fst
            logging.debug(
                f"date: {time.time() - start_time: .2f}s -- {date_graph.num_states()} nodes"
            )

            start_time = time.time()
            time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
            logging.debug(
                f"time: {time.time() - start_time: .2f}s -- {time_graph.num_states()} nodes"
            )

            start_time = time.time()
            telephone_graph = TelephoneFst(deterministic=deterministic).fst
            logging.debug(
                f"telephone: {time.time() - start_time: .2f}s -- {telephone_graph.num_states()} nodes"
            )

            start_time = time.time()
            electonic_graph = ElectronicFst(deterministic=deterministic).fst
            logging.debug(
                f"electronic: {time.time() - start_time: .2f}s -- {electonic_graph.num_states()} nodes"
            )

            start_time = time.time()
            money_graph = MoneyFst(
                cardinal=cardinal, decimal=decimal, deterministic=deterministic
            ).fst
            logging.debug(
                f"money: {time.time() - start_time: .2f}s -- {money_graph.num_states()} nodes"
            )

            start_time = time.time()
            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst
            logging.debug(
                f"whitelist: {time.time() - start_time: .2f}s -- {whitelist_graph.num_states()} nodes"
            )

            start_time = time.time()
            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.fst
            logging.debug(
                f"punct: {time.time() - start_time: .2f}s -- {punct_graph.num_states()} nodes"
            )

            start_time = time.time()
            word_graph = WordFst(punctuation=punctuation, deterministic=deterministic).fst
            logging.debug(
                f"word: {time.time() - start_time: .2f}s -- {word_graph.num_states()} nodes"
            )

            start_time = time.time()
            serial_graph = SerialFst(
                cardinal=cardinal, ordinal=ordinal, deterministic=deterministic
            ).fst
            logging.debug(
                f"serial: {time.time() - start_time: .2f}s -- {serial_graph.num_states()} nodes"
            )

            start_time = time.time()
            v_time_graph = vTimeFst(deterministic=deterministic).fst
            v_ordinal_graph = vOrdinalFst(deterministic=deterministic)
            v_date_graph = vDateFst(ordinal=v_ordinal_graph, deterministic=deterministic).fst
            time_final = pynini.compose(time_graph, v_time_graph)
            date_final = pynini.compose(date_graph, v_date_graph)
            range_graph = RangeFst(
                time=time_final, date=date_final, cardinal=cardinal, deterministic=deterministic
            ).fst
            logging.debug(
                f"range: {time.time() - start_time: .2f}s -- {range_graph.num_states()} nodes"
            )

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.09)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.1)
                | pynutil.add_weight(telephone_graph, 1.1)
                | pynutil.add_weight(electonic_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
                | pynutil.add_weight(range_graph, 1.1)
                | pynutil.add_weight(
                    serial_graph, 1.1001
                )  # should be higher than the rest of the classes
            )

            roman_graph = RomanFst(deterministic=deterministic).fst
            classify |= pynutil.add_weight(roman_graph, 1.1)

            if not deterministic:
                abbreviation_graph = AbbreviationFst(deterministic=deterministic).fst
                classify |= pynutil.add_weight(abbreviation_graph, 100)

            punct = (
                pynutil.insert("tokens { ")
                + pynutil.add_weight(punct_graph, weight=2.1)
                + pynutil.insert(" }")
            )
            punct = pynini.closure(
                pynini.compose(pynini.closure(DAMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct),
                1,
            )

            classify |= pynutil.add_weight(word_graph, 100)
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + token
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(DAMO_WHITE_SPACE, 1), delete_extra_space)
                    | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                )
                + token_plus_punct
            )

            graph = delete_space + graph + delete_space
            graph |= punct

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
