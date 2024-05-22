import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        "01.05" -> tokens { date { day: "первое мая" } }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: dict, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        # Ru format: DD-MM-YYYY or DD-MM-YY
        month_abbr_to_names = pynini.string_file(
            get_abs_path("data/months/abbr_to_name.tsv")
        ).optimize()

        delete_sep = pynutil.add_weight(pynini.cross(".", " "), 1.09) | pynutil.add_weight(
            pynini.cross(pynini.union("/", "-"), " "), 1.1
        )

        numbers = number_names["ordinal_number_names"]

        zero = (pynutil.add_weight(pynini.cross("0", ""), -0.1)) | (
            pynutil.add_weight(pynini.cross("0", "ноль "), 0.1)
        )
        zero_digit = zero + pynini.compose(DAMO_DIGIT, numbers)
        digit_day = (pynini.union("1", "2", "3") + DAMO_DIGIT) | DAMO_DIGIT
        digit_day = pynini.compose(digit_day, numbers)

        day = zero_digit | digit_day
        day = pynini.compose(
            day,
            pynini.difference(DAMO_SIGMA, DAMO_SIGMA + pynini.union("ой", "ая", "ых", "ые", "ыми")),
        )
        day = (pynutil.insert('day: "') + day + pynutil.insert('"')).optimize()

        digit_month = zero_digit | pynini.compose(pynini.accep("1") + DAMO_DIGIT, numbers)
        month_number_to_abbr = pynini.string_file(
            get_abs_path("data/months/numbers.tsv")
        ).optimize()
        month_number_to_abbr = (
            (
                ((pynutil.add_weight(pynini.cross("0", ""), -0.1) | pynini.accep("1")) + DAMO_DIGIT)
                | DAMO_DIGIT
            ).optimize()
            @ month_number_to_abbr
        ).optimize()

        month_name = (
            (month_number_to_abbr @ month_abbr_to_names)
            | pynutil.add_weight(month_abbr_to_names, 0.1)
        ).optimize()
        month = (
            pynutil.insert('month: "')
            + (month_name | pynutil.add_weight(digit_month, 0.1))
            + pynutil.insert('"')
        ).optimize()
        year = pynini.compose(((DAMO_DIGIT**4) | (DAMO_DIGIT**2)), numbers).optimize()
        year |= zero_digit

        # reduce year options
        year_wrong_endings = ["ую", "ая"]
        year_wrong_beginning = ["две тысяча", "два тысяч", "два тысячи", "две тысяч "]
        year = pynini.compose(
            year, pynini.difference(DAMO_SIGMA, DAMO_SIGMA + pynini.union("ой", "ого"))
        ) | pynutil.add_weight(pynini.compose(year, DAMO_SIGMA + pynini.union("ой", "ого")), -0.1)

        year_restrict1 = pynini.difference(
            DAMO_SIGMA, pynini.union(*year_wrong_beginning) + DAMO_SIGMA
        )
        year_restrict2 = pynini.difference(
            DAMO_SIGMA, DAMO_SIGMA + pynini.union(*year_wrong_endings)
        )
        year = pynini.compose(pynini.compose(year, year_restrict1), year_restrict2)

        year_word_singular = ["год", "года", "году", "годом", "годе"]
        year_word_plural = ["годы", "годов", "годам", "годами", "годам", "годах"]

        year_word = pynini.cross("г.", pynini.union(*year_word_singular))
        year_word |= pynini.cross("гг.", pynini.union(*year_word_plural))
        year_word = (
            pynutil.add_weight(insert_space, -0.1) | pynutil.add_weight(pynini.accep(" "), 0.1)
        ) + year_word

        year_optional = (
            pynutil.insert('year: "') + year + pynini.closure(year_word, 0, 1) + pynutil.insert('"')
        )
        year_optional = pynini.closure(delete_sep + year_optional, 0, 1).optimize()
        year_only = pynutil.insert('year: "') + year + year_word + pynutil.insert('"')

        tagger_graph = (day + delete_sep + month + year_optional) | year_only

        # Verbalizer
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete('"')
        )
        year_optional = pynini.closure(delete_extra_space + year, 0, 1)
        graph_dmy = day + delete_extra_space + month + year_optional
        verbalizer_graph = (graph_dmy | year) + delete_space

        self.final_graph = pynini.compose(tagger_graph, verbalizer_graph).optimize()
        self.fst = pynutil.insert('day: "') + self.final_graph + pynutil.insert('"')
        self.fst = self.add_tokens(self.fst).optimize()
