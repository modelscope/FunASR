import os

import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_DIGIT,
    DAMO_NOT_SPACE,
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
from fun_text_processing.text_normalization.en.verbalizers.abbreviation import (
    AbbreviationFst as vAbbreviation,
)
from fun_text_processing.text_normalization.en.verbalizers.cardinal import CardinalFst as vCardinal
from fun_text_processing.text_normalization.en.verbalizers.date import DateFst as vDate
from fun_text_processing.text_normalization.en.verbalizers.decimal import DecimalFst as vDecimal
from fun_text_processing.text_normalization.en.verbalizers.electronic import (
    ElectronicFst as vElectronic,
)
from fun_text_processing.text_normalization.en.verbalizers.fraction import FractionFst as vFraction
from fun_text_processing.text_normalization.en.verbalizers.measure import MeasureFst as vMeasure
from fun_text_processing.text_normalization.en.verbalizers.money import MoneyFst as vMoney
from fun_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as vOrdinal
from fun_text_processing.text_normalization.en.verbalizers.roman import RomanFst as vRoman
from fun_text_processing.text_normalization.en.verbalizers.telephone import (
    TelephoneFst as vTelephone,
)
from fun_text_processing.text_normalization.en.verbalizers.time import TimeFst as vTime
from fun_text_processing.text_normalization.en.verbalizers.word import WordFst as vWord
from pynini.lib import pynutil

import logging


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
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
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = True,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"_{input_case}_en_tn_{deterministic}_deterministic{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            no_digits = pynini.closure(pynini.difference(DAMO_CHAR, DAMO_DIGIT))
            self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars. This might take some time...")
            # TAGGERS
            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            deterministic_ordinal = OrdinalFst(cardinal=cardinal, deterministic=True)
            ordinal_graph = ordinal.fst

            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst
            fraction = FractionFst(deterministic=deterministic, cardinal=cardinal)
            fraction_graph = fraction.fst

            measure = MeasureFst(
                cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic
            )
            measure_graph = measure.fst
            date_graph = DateFst(cardinal=cardinal, deterministic=deterministic).fst
            punctuation = PunctuationFst(deterministic=True)
            punct_graph = punctuation.graph
            word_graph = WordFst(punctuation=punctuation, deterministic=deterministic).graph
            time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
            telephone_graph = TelephoneFst(deterministic=deterministic).fst
            electronic_graph = ElectronicFst(deterministic=deterministic).fst
            money_graph = MoneyFst(
                cardinal=cardinal, decimal=decimal, deterministic=deterministic
            ).fst
            whitelist = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            )
            whitelist_graph = whitelist.graph
            serial_graph = SerialFst(
                cardinal=cardinal, ordinal=deterministic_ordinal, deterministic=deterministic
            ).fst

            # VERBALIZERS
            cardinal = vCardinal(deterministic=deterministic)
            v_cardinal_graph = cardinal.fst
            decimal = vDecimal(cardinal=cardinal, deterministic=deterministic)
            v_decimal_graph = decimal.fst
            ordinal = vOrdinal(deterministic=deterministic)
            v_ordinal_graph = ordinal.fst
            fraction = vFraction(deterministic=deterministic)
            v_fraction_graph = fraction.fst
            v_telephone_graph = vTelephone(deterministic=deterministic).fst
            v_electronic_graph = vElectronic(deterministic=deterministic).fst
            measure = vMeasure(
                decimal=decimal, cardinal=cardinal, fraction=fraction, deterministic=deterministic
            )
            v_measure_graph = measure.fst
            v_time_graph = vTime(deterministic=deterministic).fst
            v_date_graph = vDate(ordinal=ordinal, deterministic=deterministic).fst
            v_money_graph = vMoney(decimal=decimal, deterministic=deterministic).fst
            v_roman_graph = vRoman(deterministic=deterministic).fst
            v_abbreviation = vAbbreviation(deterministic=deterministic).fst

            det_v_time_graph = vTime(deterministic=True).fst
            det_v_date_graph = vDate(ordinal=vOrdinal(deterministic=True), deterministic=True).fst
            time_final = pynini.compose(time_graph, det_v_time_graph)
            date_final = pynini.compose(date_graph, det_v_date_graph)
            range_graph = RangeFst(
                time=time_final,
                date=date_final,
                cardinal=CardinalFst(deterministic=True),
                deterministic=deterministic,
            ).fst
            v_word_graph = vWord(deterministic=deterministic).fst

            sem_w = 1
            word_w = 100
            punct_w = 2
            classify_and_verbalize = (
                pynutil.add_weight(whitelist_graph, sem_w)
                | pynutil.add_weight(pynini.compose(time_graph, v_time_graph), sem_w)
                | pynutil.add_weight(pynini.compose(decimal_graph, v_decimal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(measure_graph, v_measure_graph), sem_w)
                | pynutil.add_weight(pynini.compose(cardinal_graph, v_cardinal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(ordinal_graph, v_ordinal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(telephone_graph, v_telephone_graph), sem_w)
                | pynutil.add_weight(pynini.compose(electronic_graph, v_electronic_graph), sem_w)
                | pynutil.add_weight(pynini.compose(fraction_graph, v_fraction_graph), sem_w)
                | pynutil.add_weight(pynini.compose(money_graph, v_money_graph), sem_w)
                | pynutil.add_weight(word_graph, word_w)
                | pynutil.add_weight(pynini.compose(date_graph, v_date_graph), sem_w - 0.01)
                | pynutil.add_weight(pynini.compose(range_graph, v_word_graph), sem_w)
                | pynutil.add_weight(
                    pynini.compose(serial_graph, v_word_graph), 1.1001
                )  # should be higher than the rest of the classes
            ).optimize()

            if not deterministic:
                roman_graph = RomanFst(deterministic=deterministic).fst
                # the weight matches the word_graph weight for "I" cases in long sentences with multiple semiotic tokens
                classify_and_verbalize |= pynutil.add_weight(
                    pynini.compose(roman_graph, v_roman_graph), word_w
                )

                abbreviation_graph = AbbreviationFst(
                    whitelist=whitelist, deterministic=deterministic
                ).fst
                classify_and_verbalize |= pynutil.add_weight(
                    pynini.compose(abbreviation_graph, v_abbreviation), word_w
                )

            punct_only = pynutil.add_weight(punct_graph, weight=punct_w)
            punct = pynini.closure(
                pynini.compose(pynini.closure(DAMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct_only),
                1,
            )

            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" "))
                + classify_and_verbalize
                + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(DAMO_WHITE_SPACE, 1), delete_extra_space)
                    | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                )
                + token_plus_punct
            )

            graph |= punct_only + pynini.closure(punct)
            graph = delete_space + graph + delete_space

            remove_extra_spaces = pynini.closure(DAMO_NOT_SPACE, 1) + pynini.closure(
                delete_extra_space + pynini.closure(DAMO_NOT_SPACE, 1)
            )
            remove_extra_spaces |= (
                pynini.closure(pynutil.delete(" "), 1)
                + pynini.closure(DAMO_NOT_SPACE, 1)
                + pynini.closure(delete_extra_space + pynini.closure(DAMO_NOT_SPACE, 1))
            )

            graph = pynini.compose(graph.optimize(), remove_extra_spaces).optimize()
            self.fst = graph
            no_digits = pynini.closure(pynini.difference(DAMO_CHAR, DAMO_DIGIT))
            self.fst_no_digits = pynini.compose(graph, no_digits).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
