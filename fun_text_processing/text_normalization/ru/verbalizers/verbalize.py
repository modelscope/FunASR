from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from fun_text_processing.text_normalization.en.verbalizers.whitelist import WhiteListFst
from fun_text_processing.text_normalization.ru.verbalizers.cardinal import CardinalFst
from fun_text_processing.text_normalization.ru.verbalizers.date import DateFst
from fun_text_processing.text_normalization.ru.verbalizers.decimal import DecimalFst
from fun_text_processing.text_normalization.ru.verbalizers.electronic import ElectronicFst
from fun_text_processing.text_normalization.ru.verbalizers.measure import MeasureFst
from fun_text_processing.text_normalization.ru.verbalizers.money import MoneyFst
from fun_text_processing.text_normalization.ru.verbalizers.ordinal import OrdinalFst
from fun_text_processing.text_normalization.ru.verbalizers.telephone import TelephoneFst
from fun_text_processing.text_normalization.ru.verbalizers.time import TimeFst


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
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal_graph = OrdinalFst().fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        date = DateFst()
        date_graph = date.fst
        measure = MeasureFst()
        measure_graph = measure.fst
        electronic = ElectronicFst()
        electronic_graph = electronic.fst
        whitelist_graph = WhiteListFst().fst
        money_graph = MoneyFst().fst
        telephone_graph = TelephoneFst().fst
        time_graph = TimeFst().fst

        graph = (
            measure_graph
            | cardinal_graph
            | decimal_graph
            | ordinal_graph
            | date_graph
            | electronic_graph
            | money_graph
            | whitelist_graph
            | telephone_graph
            | time_graph
        )
        self.fst = graph
