from fun_text_processing.inverse_text_normalization.vi.graph_utils import GraphFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.cardinal import CardinalFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.date import DateFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.decimal import DecimalFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.electronic import ElectronicFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.fraction import FractionFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.measure import MeasureFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.money import MoneyFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.ordinal import OrdinalFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.telephone import TelephoneFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.time import TimeFst
from fun_text_processing.inverse_text_normalization.vi.verbalizers.whitelist import WhiteListFst


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
        ordinal_graph = OrdinalFst().fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        fraction = FractionFst()
        fraction_graph = fraction.fst
        measure_graph = MeasureFst(decimal=decimal, cardinal=cardinal).fst
        money_graph = MoneyFst(decimal=decimal).fst
        time_graph = TimeFst().fst
        date_graph = DateFst().fst
        whitelist_graph = WhiteListFst().fst
        telephone_graph = TelephoneFst().fst
        electronic_graph = ElectronicFst().fst
        graph = (
            time_graph
            | date_graph
            | money_graph
            | measure_graph
            | ordinal_graph
            | fraction_graph
            | decimal_graph
            | cardinal_graph
            | whitelist_graph
            | telephone_graph
            | electronic_graph
        )
        self.fst = graph
