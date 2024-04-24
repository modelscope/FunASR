import os

import pynini
from fun_text_processing.inverse_text_normalization.de.taggers.cardinal import CardinalFst
from fun_text_processing.inverse_text_normalization.de.taggers.date import DateFst
from fun_text_processing.inverse_text_normalization.de.taggers.decimal import DecimalFst
from fun_text_processing.inverse_text_normalization.de.taggers.electronic import ElectronicFst
from fun_text_processing.inverse_text_normalization.de.taggers.fraction import FractionFst
from fun_text_processing.inverse_text_normalization.de.taggers.measure import MeasureFst
from fun_text_processing.inverse_text_normalization.de.taggers.money import MoneyFst
from fun_text_processing.inverse_text_normalization.de.taggers.ordinal import OrdinalFst
from fun_text_processing.inverse_text_normalization.de.taggers.telephone import TelephoneFst
from fun_text_processing.inverse_text_normalization.de.taggers.time import TimeFst
from fun_text_processing.inverse_text_normalization.de.taggers.whitelist import WhiteListFst
from fun_text_processing.inverse_text_normalization.en.taggers.punctuation import PunctuationFst
from fun_text_processing.inverse_text_normalization.en.taggers.word import WordFst
from fun_text_processing.text_normalization.de.taggers.cardinal import (
    CardinalFst as TNCardinalTagger,
)
from fun_text_processing.text_normalization.de.taggers.date import DateFst as TNDateTagger
from fun_text_processing.text_normalization.de.taggers.decimal import DecimalFst as TNDecimalTagger
from fun_text_processing.text_normalization.de.taggers.electronic import (
    ElectronicFst as TNElectronicTagger,
)
from fun_text_processing.text_normalization.de.taggers.whitelist import (
    WhiteListFst as TNWhitelistTagger,
)
from fun_text_processing.text_normalization.de.verbalizers.date import DateFst as TNDateVerbalizer
from fun_text_processing.text_normalization.de.verbalizers.electronic import (
    ElectronicFst as TNElectronicVerbalizer,
)
from fun_text_processing.text_normalization.de.verbalizers.fraction import (
    FractionFst as TNFractionVerbalizer,
)
from fun_text_processing.text_normalization.de.verbalizers.ordinal import (
    OrdinalFst as TNOrdinalVerbalizer,
)
from fun_text_processing.text_normalization.de.verbalizers.time import TimeFst as TNTimeVerbalizer
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
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
        self, cache_dir: str = None, overwrite_cache: bool = False, deterministic: bool = True
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            far_file = os.path.join(cache_dir, "_de_itn.far")
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            tn_cardinal_tagger = TNCardinalTagger(deterministic=False)
            tn_date_tagger = TNDateTagger(cardinal=tn_cardinal_tagger, deterministic=False)
            tn_decimal_tagger = TNDecimalTagger(cardinal=tn_cardinal_tagger, deterministic=False)
            tn_ordinal_verbalizer = TNOrdinalVerbalizer(deterministic=False)
            tn_fraction_verbalizer = TNFractionVerbalizer(
                ordinal=tn_ordinal_verbalizer, deterministic=False
            )
            tn_time_verbalizer = TNTimeVerbalizer(
                cardinal_tagger=tn_cardinal_tagger, deterministic=False
            )
            tn_date_verbalizer = TNDateVerbalizer(
                ordinal=tn_ordinal_verbalizer, deterministic=False
            )
            tn_electronic_tagger = TNElectronicTagger(deterministic=False)
            tn_electronic_verbalizer = TNElectronicVerbalizer(deterministic=False)
            tn_whitelist_tagger = TNWhitelistTagger(input_case="cased", deterministic=False)

            cardinal = CardinalFst(tn_cardinal_tagger=tn_cardinal_tagger)
            cardinal_graph = cardinal.fst

            ordinal = OrdinalFst(
                itn_cardinal_tagger=cardinal, tn_ordinal_verbalizer=tn_ordinal_verbalizer
            )
            ordinal_graph = ordinal.fst
            decimal = DecimalFst(itn_cardinal_tagger=cardinal, tn_decimal_tagger=tn_decimal_tagger)
            decimal_graph = decimal.fst

            fraction = FractionFst(
                itn_cardinal_tagger=cardinal, tn_fraction_verbalizer=tn_fraction_verbalizer
            )
            fraction_graph = fraction.fst

            measure_graph = MeasureFst(
                itn_cardinal_tagger=cardinal,
                itn_decimal_tagger=decimal,
                itn_fraction_tagger=fraction,
            ).fst
            date_graph = DateFst(
                itn_cardinal_tagger=cardinal,
                tn_date_verbalizer=tn_date_verbalizer,
                tn_date_tagger=tn_date_tagger,
            ).fst
            word_graph = WordFst().fst
            time_graph = TimeFst(tn_time_verbalizer=tn_time_verbalizer).fst
            money_graph = MoneyFst(itn_cardinal_tagger=cardinal, itn_decimal_tagger=decimal).fst
            whitelist_graph = WhiteListFst(tn_whitelist_tagger=tn_whitelist_tagger).fst
            punct_graph = PunctuationFst().fst
            electronic_graph = ElectronicFst(
                tn_electronic_tagger=tn_electronic_tagger,
                tn_electronic_verbalizer=tn_electronic_verbalizer,
            ).fst
            telephone_graph = TelephoneFst(tn_cardinal_tagger=tn_cardinal_tagger).fst

            classify = (
                pynutil.add_weight(cardinal_graph, 1.1)
                | pynutil.add_weight(whitelist_graph, 1.0)
                | pynutil.add_weight(time_graph, 1.1)
                | pynutil.add_weight(date_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(measure_graph, 1.1)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(fraction_graph, 1.1)
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

            graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
