import pynini
from fun_text_processing.inverse_text_normalization.pt.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_SIGMA,
    DAMO_SPACE,
    DAMO_WHITE_SPACE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. menos veintitrés -> cardinal { negative: "-" integer: "23"}
    This class converts cardinals up to (but not including) "un cuatrillón",
    i.e up to "one septillion" in English (10^{24}).
    Cardinals below ten are not converted (in order to avoid
    "vivo em uma casa" --> "vivo em 1 casa" and any other odd conversions.)

    Although technically Portuguese grammar requires that "e" only comes after
    "10s" numbers (ie. "trinta", ..., "noventa"), these rules will convert
    numbers even with "e" in an ungrammatical place (because "e" is ignored
    inside cardinal numbers).
        e.g. "mil e uma" -> cardinal { integer: "1001"}
        e.g. "cento e uma" -> cardinal { integer: "101"}
    """

    def __init__(self, use_strict_e=False):
        """
        :param use_strict_e: When True forces to have the separator "e" in the right places
        """
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))
        graph_one_hundred = pynini.string_file(get_abs_path("data/numbers/onehundred.tsv"))
        graph_hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))

        graph = None

        if not use_strict_e:
            graph_hundred_component = graph_hundreds | pynutil.insert("0")
            graph_hundred_component += delete_space
            graph_hundred_component += pynini.union(
                graph_twenties | graph_teen | pynutil.insert("00"),
                (graph_ties | pynutil.insert("0"))
                + delete_space
                + (graph_digit | pynutil.insert("0")),
            )
            graph_hundred_component = pynini.union(graph_hundred_component, graph_one_hundred)

            graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
                pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT)
            )

            graph_thousands = pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit
                + delete_space
                + pynutil.delete("mil"),
                pynutil.insert("001")
                + pynutil.delete("mil"),  # because we say 'mil', not 'hum mil'
                pynutil.insert("000", weight=0.01),
            )

            graph_milhoes = pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit
                + delete_space
                + (pynutil.delete("milhão") | pynutil.delete("milhões")),
                pynutil.insert("000", weight=0.01),
            )

            graph_bilhoes = pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit
                + delete_space
                + (pynutil.delete("bilhão") | pynutil.delete("bilhões")),
                pynutil.insert("000", weight=0.01),
            )

            graph_trilhoes = pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit
                + delete_space
                + (pynutil.delete("trilhão") | pynutil.delete("trilhões")),
                pynutil.insert("000", weight=0.01),
            )

            graph_quatrilhoes = pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit
                + delete_space
                + (pynutil.delete("quatrilhão") | pynutil.delete("quatrilhões")),
                pynutil.insert("000", weight=0.01),
            )

            graph_quintilhoes = pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit
                + delete_space
                + (pynutil.delete("quintilhão") | pynutil.delete("quintilhões")),
                pynutil.insert("000", weight=0.01),
            )

            graph_sextilhoes = pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit
                + delete_space
                + (pynutil.delete("sextilhão") | pynutil.delete("sextilhões")),
                pynutil.insert("000", weight=0.01),
            )

            graph = pynini.union(
                graph_sextilhoes
                + delete_space
                + graph_quintilhoes
                + delete_space
                + graph_quatrilhoes
                + delete_space
                + graph_trilhoes
                + delete_space
                + graph_bilhoes
                + delete_space
                + graph_milhoes
                + delete_space
                + graph_thousands
                + delete_space
                + graph_hundred_component,
                graph_zero,
            )

            graph = graph @ pynini.union(
                pynutil.delete(pynini.closure("0"))
                + pynini.difference(DAMO_DIGIT, "0")
                + pynini.closure(DAMO_DIGIT),
                "0",
            )

            graph = (
                pynini.cdrewrite(pynutil.delete("e"), DAMO_SPACE, DAMO_SPACE, DAMO_SIGMA)
                @ (DAMO_ALPHA + DAMO_SIGMA)
                @ graph
            )

        else:
            graph_e = (
                pynutil.delete(DAMO_WHITE_SPACE.plus)
                + pynutil.delete("e")
                + pynutil.delete(DAMO_WHITE_SPACE.plus)
            )

            graph_ties_component = pynini.union(
                graph_teen | graph_twenties,
                graph_ties + ((graph_e + graph_digit) | pynutil.insert("0")),
                pynutil.add_weight(pynutil.insert("0") + graph_digit, 0.1),
            ) @ (pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT))

            graph_hundreds_except_hundred = (
                pynini.project(graph_hundreds, "input") - "cento"
            ) @ graph_hundreds

            graph_hundred_component_prefix_e = pynini.union(
                graph_one_hundred,
                pynutil.add_weight(graph_hundreds_except_hundred + pynutil.insert("00"), 0.1),
                pynutil.insert("0") + graph_ties_component,
            ) @ (pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT))
            graph_hundred_component_prefix_e = graph_hundred_component_prefix_e.optimize()

            graph_hundred_component_no_prefix = pynini.union(
                graph_hundreds + graph_e + graph_ties_component,
            ) @ (pynini.closure(DAMO_DIGIT) + (DAMO_DIGIT - "0") + pynini.closure(DAMO_DIGIT))
            graph_hundred_component_no_prefix = graph_hundred_component_no_prefix.optimize()

            graph_mil_prefix_e = pynini.union(
                # because we say 'mil', not 'hum mil'
                (
                    (graph_hundred_component_prefix_e + delete_space + pynutil.delete("mil"))
                    | (pynutil.insert("001", weight=0.1) + pynutil.delete("mil"))
                )
                + (
                    (graph_e + graph_hundred_component_prefix_e)
                    | (delete_space + graph_hundred_component_no_prefix)
                    | pynutil.insert("000", weight=0.1)
                )
            )

            graph_mil_no_prefix = pynini.union(
                (
                    (graph_hundred_component_no_prefix + delete_space + pynutil.delete("mil"))
                    | pynutil.insert("000", weight=0.1)
                )
                + (
                    (graph_e + graph_hundred_component_prefix_e)
                    | (delete_space + graph_hundred_component_no_prefix)
                    | pynutil.insert("000", weight=0.1)
                )
            )

            graph_milhao_prefix_e = pynini.union(
                (
                    graph_hundred_component_prefix_e
                    + delete_space
                    + (pynutil.delete("milhão") | pynutil.delete("milhões"))
                )
                + ((graph_e + graph_mil_prefix_e) | (delete_space + graph_mil_no_prefix))
            )

            graph_milhao_no_prefix = pynini.union(
                (
                    (
                        graph_hundred_component_no_prefix
                        + delete_space
                        + (pynutil.delete("milhão") | pynutil.delete("milhões"))
                    )
                    | pynutil.insert("000", weight=0.1)
                )
                + ((graph_e + graph_mil_prefix_e) | (delete_space + graph_mil_no_prefix))
            )

            graph_bilhao_prefix_e = pynini.union(
                (
                    graph_hundred_component_prefix_e
                    + delete_space
                    + (pynutil.delete("bilhão") | pynutil.delete("bilhões"))
                )
                + ((graph_e + graph_milhao_prefix_e) | (delete_space + graph_milhao_no_prefix))
            )

            graph_bilhao_no_prefix = pynini.union(
                (
                    (
                        graph_hundred_component_no_prefix
                        + delete_space
                        + (pynutil.delete("bilhão") | pynutil.delete("bilhões"))
                    )
                    | pynutil.insert("000", weight=0.1)
                )
                + ((graph_e + graph_milhao_prefix_e) | (delete_space + graph_milhao_no_prefix))
            )

            graph_trilhao_prefix_e = pynini.union(
                (
                    graph_hundred_component_prefix_e
                    + delete_space
                    + (pynutil.delete("trilhão") | pynutil.delete("trilhões"))
                )
                + ((graph_e + graph_bilhao_prefix_e) | (delete_space + graph_bilhao_no_prefix))
            )

            graph_trilhao_no_prefix = pynini.union(
                (
                    (
                        graph_hundred_component_no_prefix
                        + delete_space
                        + (pynutil.delete("trilhão") | pynutil.delete("trilhões"))
                    )
                    | pynutil.insert("000", weight=0.1)
                )
                + ((graph_e + graph_bilhao_prefix_e) | (delete_space + graph_bilhao_no_prefix))
            )

            graph_quatrilhao_prefix_e = pynini.union(
                (
                    graph_hundred_component_prefix_e
                    + delete_space
                    + (pynutil.delete("quatrilhão") | pynutil.delete("quatrilhões"))
                )
                + ((graph_e + graph_trilhao_prefix_e) | (delete_space + graph_trilhao_no_prefix))
            )

            graph_quatrilhao_no_prefix = pynini.union(
                (
                    (
                        graph_hundred_component_no_prefix
                        + delete_space
                        + (pynutil.delete("quatrilhão") | pynutil.delete("quatrilhões"))
                    )
                    | pynutil.insert("000", weight=0.1)
                )
                + ((graph_e + graph_trilhao_prefix_e) | (delete_space + graph_trilhao_no_prefix))
            )

            graph_quintilhao_prefix_e = pynini.union(
                (
                    graph_hundred_component_prefix_e
                    + delete_space
                    + (pynutil.delete("quintilhão") | pynutil.delete("quintilhões"))
                )
                + (
                    (graph_e + graph_quatrilhao_prefix_e)
                    | (delete_space + graph_quatrilhao_no_prefix)
                )
            )

            graph_quintilhao_no_prefix = pynini.union(
                (
                    (
                        graph_hundred_component_no_prefix
                        + delete_space
                        + (pynutil.delete("quintilhão") | pynutil.delete("quintilhões"))
                    )
                    | pynutil.insert("000", weight=0.1)
                )
                + (
                    (graph_e + graph_quatrilhao_prefix_e)
                    | (delete_space + graph_quatrilhao_no_prefix)
                )
            )

            graph_sextilhao_prefix_e = pynini.union(
                (
                    graph_hundred_component_prefix_e
                    + delete_space
                    + (pynutil.delete("sextilhão") | pynutil.delete("sextilhões"))
                )
                + (
                    (graph_e + graph_quintilhao_prefix_e)
                    | (delete_space + graph_quintilhao_no_prefix)
                )
            )

            graph_sextilhao_no_prefix = pynini.union(
                (
                    (
                        graph_hundred_component_no_prefix
                        + delete_space
                        + (pynutil.delete("sextilhão") | pynutil.delete("sextilhões"))
                    )
                    | pynutil.insert("000", weight=0.1)
                )
                + (
                    (graph_e + graph_quintilhao_prefix_e)
                    | (delete_space + graph_quintilhao_no_prefix)
                )
            )

            graph = pynini.union(
                graph_sextilhao_no_prefix,
                graph_sextilhao_prefix_e,
                graph_quintilhao_prefix_e,
                graph_quatrilhao_prefix_e,
                graph_trilhao_prefix_e,
                graph_bilhao_prefix_e,
                graph_milhao_prefix_e,
                graph_mil_prefix_e,
                graph_hundred_component_prefix_e,
                graph_ties_component,
                graph_zero,
            ).optimize()

            graph = graph @ pynini.union(
                pynutil.delete(pynini.closure("0"))
                + pynini.difference(DAMO_DIGIT, "0")
                + pynini.closure(DAMO_DIGIT),
                "0",
            )

        graph = graph.optimize()
        self.graph_no_exception = graph

        # save self.numbers_up_to_thousand for use in DecimalFst
        digits_up_to_thousand = DAMO_DIGIT | (DAMO_DIGIT**2) | (DAMO_DIGIT**3)
        numbers_up_to_thousand = pynini.compose(graph, digits_up_to_thousand).optimize()
        self.numbers_up_to_thousand = numbers_up_to_thousand

        # save self.numbers_up_to_million for use in DecimalFst
        digits_up_to_million = (
            DAMO_DIGIT
            | (DAMO_DIGIT**2)
            | (DAMO_DIGIT**3)
            | (DAMO_DIGIT**4)
            | (DAMO_DIGIT**5)
            | (DAMO_DIGIT**6)
        )
        numbers_up_to_million = pynini.compose(graph, digits_up_to_million).optimize()
        self.numbers_up_to_million = numbers_up_to_million

        # save self.digits_from_year for use in DateFst
        digits_1_2099 = [str(digits) for digits in range(1, 2100)]
        digits_from_year = (numbers_up_to_million @ pynini.union(*digits_1_2099)).optimize()
        self.digits_from_year = digits_from_year

        # don't convert cardinals from zero to nine inclusive
        graph_exception = pynini.project(pynini.union(graph_digit, graph_zero), "input")

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("menos", '"-"') + DAMO_SPACE, 0, 1
        )

        final_graph = (
            optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
