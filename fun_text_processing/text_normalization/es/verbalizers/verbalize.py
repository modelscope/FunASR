from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from fun_text_processing.text_normalization.en.verbalizers.whitelist import WhiteListFst
from fun_text_processing.text_normalization.es.verbalizers.cardinal import CardinalFst
from fun_text_processing.text_normalization.es.verbalizers.date import DateFst
from fun_text_processing.text_normalization.es.verbalizers.decimals import DecimalFst
from fun_text_processing.text_normalization.es.verbalizers.electronic import ElectronicFst
from fun_text_processing.text_normalization.es.verbalizers.fraction import FractionFst
from fun_text_processing.text_normalization.es.verbalizers.measure import MeasureFst
from fun_text_processing.text_normalization.es.verbalizers.money import MoneyFst
from fun_text_processing.text_normalization.es.verbalizers.ordinal import OrdinalFst
from fun_text_processing.text_normalization.es.verbalizers.telephone import TelephoneFst
from fun_text_processing.text_normalization.es.verbalizers.time import TimeFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        cardinal = CardinalFst(deterministic=deterministic)
        cardinal_graph = cardinal.fst
        ordinal = OrdinalFst(deterministic=deterministic)
        ordinal_graph = ordinal.fst
        decimal = DecimalFst(deterministic=deterministic)
        decimal_graph = decimal.fst
        fraction = FractionFst(deterministic=deterministic)
        fraction_graph = fraction.fst
        date = DateFst(deterministic=deterministic)
        date_graph = date.fst
        measure = MeasureFst(
            cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic
        )
        measure_graph = measure.fst
        electronic = ElectronicFst(deterministic=deterministic)
        electronic_graph = electronic.fst
        whitelist_graph = WhiteListFst(deterministic=deterministic).fst
        money_graph = MoneyFst(decimal=decimal, deterministic=deterministic).fst
        telephone_graph = TelephoneFst(deterministic=deterministic).fst
        time_graph = TimeFst(deterministic=deterministic).fst

        graph = (
            cardinal_graph
            | measure_graph
            | decimal_graph
            | ordinal_graph
            | date_graph
            | electronic_graph
            | money_graph
            | fraction_graph
            | whitelist_graph
            | telephone_graph
            | time_graph
        )
        self.fst = graph
