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

import pynini
from fun_text_processing.inverse_text_normalization.ja.utils import get_abs_path
from fun_text_processing.inverse_text_normalization.ja.graph_utils import (
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure
        e.g. minus twelve kilograms -> measure { negative: "true" cardinal { integer: "12" } units: "kg" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="measure", kind="classify")

        cardinal_graph = cardinal.graph_no_exception

        graph_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))
        graph_unit_singular = graph_unit  # singular -> abbr
        graph_unit_plural = graph_unit_singular  # plural -> abbr

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("マイナス", "\"true\"") + delete_extra_space, 0, 1
        )

        unit_singular = graph_unit_singular
        unit_plural = graph_unit_plural
        unit_misc = pynutil.insert("/") + pynutil.delete("每") + delete_space + graph_unit_singular

        unit_singular = (
            pynutil.insert("units: \"")
            + (unit_singular | unit_misc | pynutil.add_weight(unit_singular + delete_space + unit_misc, 0.01))
            + pynutil.insert("\"")
        )

        unit_plural = (
            pynutil.insert("units: \"")
            + (unit_plural | unit_misc | pynutil.add_weight(unit_plural + delete_space + unit_misc, 0.01))
            + pynutil.insert("\"")
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_plural
        )
        subgraph_decimal |= (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + pynutil.insert(" }")
            # + delete_extra_space
            + unit_plural
        )
        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((DAMO_SIGMA - "一") @ cardinal_graph)
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_plural
        )
        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross("一", "1")
            + pynutil.insert("\"")
            + pynutil.insert(" }")
            + delete_extra_space
            + unit_singular
        )
        subgraph_cardinal |= (
                pynutil.insert("cardinal { ")
                + optional_graph_negative
                + pynutil.insert("integer: \"")
                + ((DAMO_SIGMA - "一") @ cardinal_graph)
                + pynutil.insert("\"")
                + pynutil.insert(" }")
                + unit_singular
        )
        final_graph = subgraph_decimal | subgraph_cardinal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
