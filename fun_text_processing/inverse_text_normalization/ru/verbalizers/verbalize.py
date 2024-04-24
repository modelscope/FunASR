from fun_text_processing.inverse_text_normalization.en.verbalizers.whitelist import WhiteListFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.cardinal import CardinalFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.date import DateFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.decimal import DecimalFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.electronic import ElectronicFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.measure import MeasureFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.money import MoneyFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.ordinal import OrdinalFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.telephone import TelephoneFst
from fun_text_processing.inverse_text_normalization.ru.verbalizers.time import TimeFst
from fun_text_processing.text_normalization.en.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal = OrdinalFst()
        ordinal_graph = ordinal.fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        whitelist_graph = WhiteListFst().fst
        electronic_graph = ElectronicFst().fst
        money_graph = MoneyFst().fst
        date_graph = DateFst().fst
        measure_graph = MeasureFst().fst
        telephone_graph = TelephoneFst().fst
        time_graph = TimeFst().fst

        graph = (
            whitelist_graph
            | cardinal_graph
            | ordinal_graph
            | decimal_graph
            | electronic_graph
            | date_graph
            | money_graph
            | measure_graph
            | telephone_graph
            | time_graph
        )

        self.fst = graph
