import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_DIGIT,
    DAMO_LOWER,
    DAMO_SIGMA,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.en.utils import (
    augment_labels_with_punct_at_end,
    get_abs_path,
    load_labels,
)
from pynini.examples import plurals
from pynini.lib import pynutil

graph_teen = pynini.invert(pynini.string_file(get_abs_path("data/number/teen.tsv"))).optimize()
graph_digit = pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv"))).optimize()
ties_graph = pynini.invert(pynini.string_file(get_abs_path("data/number/ty.tsv"))).optimize()
year_suffix = load_labels(get_abs_path("data/date/year_suffix.tsv"))
year_suffix.extend(augment_labels_with_punct_at_end(year_suffix))
year_suffix = pynini.string_map(year_suffix).optimize()


def get_ties_graph(deterministic: bool = True):
    """
    Returns two digit transducer, e.g.
    03 -> o three
    12 -> thirteen
    20 -> twenty
    """
    graph = graph_teen | ties_graph + pynutil.delete("0") | ties_graph + insert_space + graph_digit

    if deterministic:
        graph = graph | pynini.cross("0", "oh") + insert_space + graph_digit
    else:
        graph = (
            graph
            | (pynini.cross("0", "oh") | pynini.cross("0", "zero")) + insert_space + graph_digit
        )

    return graph.optimize()


def get_four_digit_year_graph(deterministic: bool = True):
    """
    Returns a four digit transducer which is combination of ties/teen or digits
    (using hundred instead of thousand format), e.g.
    1219 -> twelve nineteen
    3900 -> thirty nine hundred
    """
    graph_ties = get_ties_graph(deterministic)

    graph_with_s = (
        (graph_ties + insert_space + graph_ties)
        | (graph_teen + insert_space + (ties_graph | pynini.cross("1", "ten")))
    ) + pynutil.delete("0s")

    graph_with_s |= (
        (graph_teen | graph_ties)
        + insert_space
        + pynini.cross("00", "hundred")
        + pynutil.delete("s")
    )
    graph_with_s = graph_with_s @ pynini.cdrewrite(
        pynini.cross("y", "ies") | pynutil.insert("s"), "", "[EOS]", DAMO_SIGMA
    )

    graph = graph_ties + insert_space + graph_ties
    graph |= (graph_teen | graph_ties) + insert_space + pynini.cross("00", "hundred")

    thousand_graph = (
        graph_digit
        + insert_space
        + pynini.cross("00", "thousand")
        + (pynutil.delete("0") | insert_space + graph_digit)
    )
    thousand_graph |= (
        graph_digit
        + insert_space
        + pynini.cross("000", "thousand")
        + pynini.closure(pynutil.delete(" "), 0, 1)
        + pynini.accep("s")
    )

    graph |= graph_with_s
    if deterministic:
        graph = plurals._priority_union(thousand_graph, graph, DAMO_SIGMA)
    else:
        graph |= thousand_graph

    return graph.optimize()


def _get_two_digit_year_with_s_graph():
    # to handle '70s -> seventies
    graph = (
        pynini.closure(pynutil.delete("'"), 0, 1)
        + pynini.compose(
            ties_graph + pynutil.delete("0s"),
            pynini.cdrewrite(pynini.cross("y", "ies"), "", "[EOS]", DAMO_SIGMA),
        )
    ).optimize()
    return graph


def _get_year_graph(cardinal_graph, deterministic: bool = True):
    """
    Transducer for year, only from 1000 - 2999 e.g.
    1290 -> twelve nineteen
    2000 - 2009 will be verbalized as two thousand.

    Transducer for 3 digit year, e.g. 123-> one twenty three

    Transducer for year with suffix
    123 A.D., 4200 B.C
    """
    graph = get_four_digit_year_graph(deterministic)
    graph = (
        pynini.union("1", "2")
        + (DAMO_DIGIT**3)
        + pynini.closure(pynini.cross(" s", "s") | "s", 0, 1)
    ) @ graph

    graph |= _get_two_digit_year_with_s_graph()

    three_digit_year = (
        (DAMO_DIGIT @ cardinal_graph) + insert_space + (DAMO_DIGIT**2) @ cardinal_graph
    )
    year_with_suffix = (
        (get_four_digit_year_graph(deterministic=True) | three_digit_year)
        + delete_space
        + insert_space
        + year_suffix
    )
    graph |= year_with_suffix
    return graph.optimize()


def _get_two_digit_year(cardinal_graph, single_digits_graph):
    wo_digit_year = DAMO_DIGIT ** (2) @ plurals._priority_union(
        cardinal_graph, single_digits_graph, DAMO_SIGMA
    )
    return wo_digit_year


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g.
        jan. 5, 2012 -> date { month: "january" day: "five" year: "twenty twelve" preserve_order: true }
        jan. 5 -> date { month: "january" day: "five" preserve_order: true }
        5 january 2012 -> date { day: "five" month: "january" year: "twenty twelve" preserve_order: true }
        2012-01-05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012.01.05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012/01/05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012 -> date { year: "twenty twelve" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool, lm: bool = False):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        # january
        month_graph = pynini.string_file(get_abs_path("data/date/month_name.tsv")).optimize()
        # January, JANUARY
        month_graph |= pynini.compose(
            TO_LOWER + pynini.closure(DAMO_CHAR), month_graph
        ) | pynini.compose(TO_LOWER ** (2, ...), month_graph)

        # jan
        month_abbr_graph = pynini.string_file(get_abs_path("data/date/month_abbr.tsv")).optimize()
        # jan, Jan, JAN
        month_abbr_graph = (
            month_abbr_graph
            | pynini.compose(TO_LOWER + pynini.closure(DAMO_LOWER, 1), month_abbr_graph).optimize()
            | pynini.compose(TO_LOWER ** (2, ...), month_abbr_graph).optimize()
        ) + pynini.closure(pynutil.delete("."), 0, 1)
        month_graph |= month_abbr_graph.optimize()

        month_numbers_labels = pynini.string_file(
            get_abs_path("data/date/month_number.tsv")
        ).optimize()
        cardinal_graph = cardinal.graph_hundred_component_at_least_one_none_zero_digit

        year_graph = _get_year_graph(cardinal_graph=cardinal_graph, deterministic=deterministic)

        # three_digit_year = (DAMO_DIGIT @ cardinal_graph) + insert_space + (DAMO_DIGIT ** 2) @ cardinal_graph
        # year_graph |= three_digit_year

        month_graph = pynutil.insert('month: "') + month_graph + pynutil.insert('"')
        month_numbers_graph = (
            pynutil.insert('month: "') + month_numbers_labels + pynutil.insert('"')
        )

        endings = ["rd", "th", "st", "nd"]
        endings += [x.upper() for x in endings]
        endings = pynini.union(*endings)

        day_graph = (
            pynutil.insert('day: "')
            + pynini.closure(pynutil.delete("the "), 0, 1)
            + (
                (
                    (pynini.union("1", "2") + DAMO_DIGIT)
                    | DAMO_DIGIT
                    | (pynini.accep("3") + pynini.union("0", "1"))
                )
                + pynini.closure(pynutil.delete(endings), 0, 1)
            )
            @ cardinal_graph
            + pynutil.insert('"')
        )

        two_digit_year = _get_two_digit_year(
            cardinal_graph=cardinal_graph, single_digits_graph=cardinal.single_digits_graph
        )
        two_digit_year = pynutil.insert('year: "') + two_digit_year + pynutil.insert('"')

        # if lm:
        #     two_digit_year = pynini.compose(pynini.difference(DAMO_DIGIT, "0") + DAMO_DIGIT ** (3), two_digit_year)
        #     year_graph = pynini.compose(pynini.difference(DAMO_DIGIT, "0") + DAMO_DIGIT ** (2), year_graph)
        #     year_graph |= pynini.compose(pynini.difference(DAMO_DIGIT, "0") + DAMO_DIGIT ** (4, ...), year_graph)

        graph_year = (
            pynutil.insert(' year: "') + pynutil.delete(" ") + year_graph + pynutil.insert('"')
        )
        graph_year |= (
            pynutil.insert(' year: "')
            + pynini.accep(",")
            + pynini.closure(pynini.accep(" "), 0, 1)
            + year_graph
            + pynutil.insert('"')
        )
        optional_graph_year = pynini.closure(graph_year, 0, 1)

        year_graph = pynutil.insert('year: "') + year_graph + pynutil.insert('"')

        graph_mdy = month_graph + (
            (delete_extra_space + day_graph)
            | (pynini.accep(" ") + day_graph)
            | graph_year
            | (delete_extra_space + day_graph + graph_year)
        )

        graph_mdy |= (
            month_graph
            + pynini.cross("-", " ")
            + day_graph
            + pynini.closure(((pynini.cross("-", " ") + DAMO_SIGMA) @ graph_year), 0, 1)
        )

        for x in ["-", "/", "."]:
            delete_sep = pynutil.delete(x)
            graph_mdy |= (
                month_numbers_graph
                + delete_sep
                + insert_space
                + pynini.closure(pynutil.delete("0"), 0, 1)
                + day_graph
                + delete_sep
                + insert_space
                + (year_graph | two_digit_year)
            )

        graph_dmy = day_graph + delete_extra_space + month_graph + optional_graph_year
        day_ex_month = (DAMO_DIGIT**2 - pynini.project(month_numbers_graph, "input")) @ day_graph
        for x in ["-", "/", "."]:
            delete_sep = pynutil.delete(x)
            graph_dmy |= (
                day_ex_month
                + delete_sep
                + insert_space
                + month_numbers_graph
                + delete_sep
                + insert_space
                + (year_graph | two_digit_year)
            )

        graph_ymd = pynini.accep("")
        for x in ["-", "/", "."]:
            delete_sep = pynutil.delete(x)
            graph_ymd |= (
                (year_graph | two_digit_year)
                + delete_sep
                + insert_space
                + month_numbers_graph
                + delete_sep
                + insert_space
                + pynini.closure(pynutil.delete("0"), 0, 1)
                + day_graph
            )

        final_graph = graph_mdy | graph_dmy

        if not deterministic or lm:
            final_graph += pynini.closure(pynutil.insert(" preserve_order: true"), 0, 1)
            m_sep_d = (
                month_numbers_graph
                + pynutil.delete(pynini.union("-", "/"))
                + insert_space
                + pynini.closure(pynutil.delete("0"), 0, 1)
                + day_graph
            )
            final_graph |= m_sep_d
        else:
            final_graph += pynutil.insert(" preserve_order: true")

        final_graph |= graph_ymd | year_graph

        if not deterministic or lm:
            ymd_to_mdy_graph = None
            ymd_to_dmy_graph = None
            mdy_to_dmy_graph = None
            md_to_dm_graph = None

            for month in [x[0] for x in load_labels(get_abs_path("data/date/month_name.tsv"))]:
                for day in [x[0] for x in load_labels(get_abs_path("data/date/day.tsv"))]:
                    ymd_to_mdy_curr = (
                        pynutil.insert('month: "' + month + '" day: "' + day + '" ')
                        + pynini.accep("year:")
                        + DAMO_SIGMA
                        + pynutil.delete(' month: "' + month + '" day: "' + day + '"')
                    )

                    # YY-MM-DD -> MM-DD-YY
                    ymd_to_mdy_curr = pynini.compose(graph_ymd, ymd_to_mdy_curr)
                    ymd_to_mdy_graph = (
                        ymd_to_mdy_curr
                        if ymd_to_mdy_graph is None
                        else pynini.union(ymd_to_mdy_curr, ymd_to_mdy_graph)
                    )

                    ymd_to_dmy_curr = (
                        pynutil.insert('day: "' + day + '" month: "' + month + '" ')
                        + pynini.accep("year:")
                        + DAMO_SIGMA
                        + pynutil.delete(' month: "' + month + '" day: "' + day + '"')
                    )

                    # YY-MM-DD -> MM-DD-YY
                    ymd_to_dmy_curr = pynini.compose(graph_ymd, ymd_to_dmy_curr).optimize()
                    ymd_to_dmy_graph = (
                        ymd_to_dmy_curr
                        if ymd_to_dmy_graph is None
                        else pynini.union(ymd_to_dmy_curr, ymd_to_dmy_graph)
                    )

                    mdy_to_dmy_curr = (
                        pynutil.insert('day: "' + day + '" month: "' + month + '" ')
                        + pynutil.delete('month: "' + month + '" day: "' + day + '" ')
                        + pynini.accep("year:")
                        + DAMO_SIGMA
                    ).optimize()
                    # MM-DD-YY -> verbalize as MM-DD-YY (February fourth 1991) or DD-MM-YY (the fourth of February 1991)
                    mdy_to_dmy_curr = pynini.compose(graph_mdy, mdy_to_dmy_curr).optimize()
                    mdy_to_dmy_graph = (
                        mdy_to_dmy_curr
                        if mdy_to_dmy_graph is None
                        else pynini.union(mdy_to_dmy_curr, mdy_to_dmy_graph).optimize()
                    ).optimize()

                    md_to_dm_curr = pynutil.insert(
                        'day: "' + day + '" month: "' + month + '"'
                    ) + pynutil.delete('month: "' + month + '" day: "' + day + '"')
                    md_to_dm_curr = pynini.compose(m_sep_d, md_to_dm_curr).optimize()

                    md_to_dm_graph = (
                        md_to_dm_curr
                        if md_to_dm_graph is None
                        else pynini.union(md_to_dm_curr, md_to_dm_graph).optimize()
                    ).optimize()

            final_graph |= mdy_to_dmy_graph | md_to_dm_graph | ymd_to_mdy_graph | ymd_to_dmy_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
