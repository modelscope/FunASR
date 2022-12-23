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

from fun_text_processing.inverse_text_normalization.de.verbalizers.cardinal import CardinalFst
from fun_text_processing.inverse_text_normalization.de.verbalizers.decimal import DecimalFst
from fun_text_processing.inverse_text_normalization.de.verbalizers.measure import MeasureFst
from fun_text_processing.inverse_text_normalization.de.verbalizers.money import MoneyFst
from fun_text_processing.inverse_text_normalization.de.verbalizers.time import TimeFst
from fun_text_processing.text_normalization.de.verbalizers.cardinal import CardinalFst as TNCardinalVerbalizer
from fun_text_processing.text_normalization.de.verbalizers.decimal import DecimalFst as TNDecimalVerbalizer
from fun_text_processing.text_normalization.en.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        tn_cardinal_verbalizer = TNCardinalVerbalizer(deterministic=False)
        tn_decimal_verbalizer = TNDecimalVerbalizer(deterministic=False)

        cardinal = CardinalFst(tn_cardinal_verbalizer=tn_cardinal_verbalizer)
        cardinal_graph = cardinal.fst
        decimal = DecimalFst(tn_decimal_verbalizer=tn_decimal_verbalizer)
        decimal_graph = decimal.fst
        measure_graph = MeasureFst(decimal=decimal, cardinal=cardinal).fst
        money_graph = MoneyFst(decimal=decimal).fst
        time_graph = TimeFst().fst
        graph = time_graph | money_graph | measure_graph | decimal_graph | cardinal_graph
        self.fst = graph
