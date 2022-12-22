# Copyright NeMo (https://github.com/NVIDIA/NeMo). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fun_text_processing.text_normalization.de.taggers.cardinal import CardinalFst as CardinalTagger
from fun_text_processing.text_normalization.de.verbalizers.cardinal import CardinalFst
from fun_text_processing.text_normalization.de.verbalizers.date import DateFst
from fun_text_processing.text_normalization.de.verbalizers.decimal import DecimalFst
from fun_text_processing.text_normalization.de.verbalizers.electronic import ElectronicFst
from fun_text_processing.text_normalization.de.verbalizers.fraction import FractionFst
from fun_text_processing.text_normalization.de.verbalizers.measure import MeasureFst
from fun_text_processing.text_normalization.de.verbalizers.money import MoneyFst
from fun_text_processing.text_normalization.de.verbalizers.ordinal import OrdinalFst
from fun_text_processing.text_normalization.de.verbalizers.telephone import TelephoneFst
from fun_text_processing.text_normalization.de.verbalizers.time import TimeFst
from fun_text_processing.text_normalization.en.graph_utils import GraphFst
from fun_text_processing.text_normalization.en.verbalizers.whitelist import WhiteListFst


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
        cardinal_tagger = CardinalTagger(deterministic=deterministic)
        cardinal = CardinalFst(deterministic=deterministic)
        cardinal_graph = cardinal.fst
        ordinal = OrdinalFst(deterministic=deterministic)
        ordinal_graph = ordinal.fst
        decimal = DecimalFst(deterministic=deterministic)
        decimal_graph = decimal.fst
        fraction = FractionFst(ordinal=ordinal, deterministic=deterministic)
        fraction_graph = fraction.fst
        date = DateFst(ordinal=ordinal)
        date_graph = date.fst
        measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
        measure_graph = measure.fst
        electronic = ElectronicFst(deterministic=deterministic)
        electronic_graph = electronic.fst
        whitelist_graph = WhiteListFst(deterministic=deterministic).fst
        money_graph = MoneyFst(decimal=decimal).fst
        telephone_graph = TelephoneFst(deterministic=deterministic).fst
        time_graph = TimeFst(cardinal_tagger=cardinal_tagger, deterministic=deterministic).fst

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
