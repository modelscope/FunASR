import pynini
from fun_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_DIGIT,
    TO_LOWER,
    GraphFst,
    insert_space,
)
from pynini.lib import pynutil

graph_teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv"))).optimize()
graph_digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
ties_graph = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv"))).optimize()
delete_leading_zero = (pynutil.delete("0") | (DAMO_DIGIT - "0")) + DAMO_DIGIT


def get_year_graph(cardinal: GraphFst) -> "pynini.FstLike":
    """
    Returns year verbalizations as fst

     < 2000 neunzehn (hundert) (vier und zwanzig), >= 2000 regular cardinal
    **00 ** hundert

    Args:
        delete_leading_zero: removed leading zero
        cardinal: cardinal GraphFst
    """

    year_gt_2000 = (pynini.union("21", "20") + DAMO_DIGIT**2) @ cardinal.graph

    graph_two_digit = delete_leading_zero @ cardinal.two_digit_non_zero
    hundred = pynutil.insert("hundert")
    graph_double_double = (
        (pynini.accep("1") + DAMO_DIGIT) @ graph_two_digit
        + insert_space
        + pynini.closure(hundred + insert_space, 0, 1)
        + graph_two_digit
    )
    # for 20**
    graph_double_double |= pynini.accep("20") @ graph_two_digit + insert_space + graph_two_digit
    graph = (
        graph_double_double
        | (pynini.accep("1") + DAMO_DIGIT) @ graph_two_digit
        + insert_space
        + pynutil.delete("00")
        + hundred
        | year_gt_2000
    )
    return graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "01.04.2010" -> date { day: "erster" month: "april" year: "zwei tausend zehn" preserve_order: true }
        "1994" -> date { year: "neunzehn vier und neuzig" }
        "1900" -> date { year: "neunzehn hundert" }

    Args:
        cardinal: cardinal GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        month_abbr_graph = load_labels(get_abs_path("data/months/abbr_to_name.tsv"))
        number_to_month = pynini.string_file(get_abs_path("data/months/numbers.tsv")).optimize()
        month_graph = pynini.union(*[x[1] for x in month_abbr_graph]).optimize()
        month_abbr_graph = pynini.string_map(month_abbr_graph)
        month_abbr_graph = (
            pynutil.add_weight(month_abbr_graph, weight=0.0001)
            | ((TO_LOWER + pynini.closure(DAMO_CHAR)) @ month_abbr_graph)
        ) + pynini.closure(pynutil.delete(".", weight=-0.0001), 0, 1)

        self.month_abbr = month_abbr_graph
        month_graph |= (TO_LOWER + pynini.closure(DAMO_CHAR)) @ month_graph
        # jan.-> januar, Jan-> januar, januar-> januar
        month_graph |= month_abbr_graph

        numbers = cardinal.graph_hundred_component_at_least_one_none_zero_digit
        optional_leading_zero = delete_leading_zero | DAMO_DIGIT
        # 01, 31, 1
        digit_day = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 32)]) @ numbers
        day = (pynutil.insert('day: "') + digit_day + pynutil.insert('"')).optimize()

        digit_month = optional_leading_zero @ pynini.union(*[str(x) for x in range(1, 13)])
        number_to_month = digit_month @ number_to_month
        digit_month @= numbers

        month_name = (pynutil.insert('month: "') + month_graph + pynutil.insert('"')).optimize()
        month_number = (
            pynutil.insert('month: "')
            + (pynutil.add_weight(digit_month, weight=0.0001) | number_to_month)
            + pynutil.insert('"')
        ).optimize()

        # prefer cardinal over year
        year = pynutil.add_weight(get_year_graph(cardinal=cardinal), weight=0.001)
        self.year = year

        year_only = pynutil.insert('year: "') + year + pynutil.insert('"')

        graph_dmy = (
            day
            + pynutil.delete(".")
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + insert_space
            + month_name
            + pynini.closure(pynini.accep(" ") + year_only, 0, 1)
        )

        separators = ["."]
        for sep in separators:
            year_optional = pynini.closure(pynini.cross(sep, " ") + year_only, 0, 1)
            new_graph = day + pynini.cross(sep, " ") + month_number + year_optional
            graph_dmy |= new_graph

        dash = "-"
        day_optional = pynini.closure(pynini.cross(dash, " ") + day, 0, 1)
        graph_ymd = year_only + pynini.cross(dash, " ") + month_number + day_optional

        final_graph = graph_dmy + pynutil.insert(" preserve_order: true")
        final_graph |= year_only
        final_graph |= graph_ymd

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
