import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    convert_space,
)
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, in the form of (day) month (year) or year
        e.g. vierundzwanzigster juli zwei tausend dreizehn -> tokens { name: "24. Jul. 2013" }
        e.g. neunzehnachtzig -> tokens { name: "1980" }
        e.g. vierzehnter januar -> tokens { name: "14. Jan." }
        e.g. zweiter dritter -> tokens { name: "02.03." }
        e.g. januar neunzehnachtzig -> tokens { name: "Jan. 1980" }
        e.g. zwanzigzwanzig -> tokens { name: "2020" }

    Args:
        itn_cardinal_tagger: ITN cardinal tagger
        tn_date_tagger: TN date tagger
        tn_date_verbalizer: TN date verbalizer
    """

    def __init__(
        self,
        itn_cardinal_tagger: GraphFst,
        tn_date_tagger: GraphFst,
        tn_date_verbalizer: GraphFst,
        deterministic: bool = True,
    ):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        add_leading_zero_to_double_digit = (DAMO_DIGIT + DAMO_DIGIT) | (
            pynutil.insert("0") + DAMO_DIGIT
        )
        optional_delete_space = pynini.closure(DAMO_SIGMA | pynutil.delete(" ", weight=0.0001))
        tagger = tn_date_verbalizer.graph.invert().optimize()

        delete_day_marker = (
            pynutil.delete('day: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        ) @ itn_cardinal_tagger.graph_no_exception

        month_as_number = (
            pynutil.delete('month: "')
            + itn_cardinal_tagger.graph_no_exception
            + pynutil.delete('"')
        )
        month_as_string = (
            pynutil.delete('month: "') + tn_date_tagger.month_abbr.invert() + pynutil.delete('"')
        )

        convert_year = (tn_date_tagger.year @ optional_delete_space).invert().optimize()
        delete_year_marker = (
            pynutil.delete('year: "') + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete('"')
        ) @ convert_year

        # day. month as string (year)
        verbalizer = (
            pynini.closure(delete_day_marker + pynutil.insert(".") + pynini.accep(" "), 0, 1)
            + month_as_string
            + pynini.closure(pynini.accep(" ") + delete_year_marker, 0, 1)
        )

        # day. month as number (year)
        verbalizer |= (
            delete_day_marker @ add_leading_zero_to_double_digit
            + pynutil.insert(".")
            + pynutil.delete(" ")
            + month_as_number @ add_leading_zero_to_double_digit
            + pynutil.insert(".")
            + pynini.closure(pynutil.delete(" ") + delete_year_marker, 0, 1)
        )

        # year
        verbalizer |= delete_year_marker

        final_graph = tagger @ verbalizer

        graph = pynutil.insert('name: "') + convert_space(final_graph) + pynutil.insert('"')
        self.fst = graph.optimize()
