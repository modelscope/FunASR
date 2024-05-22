import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_space,
)
from fun_text_processing.text_normalization.es.graph_utils import roman_to_int, strip_accent
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

digit = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit.tsv")))
teens = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/teen.tsv")))
twenties = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/twenties.tsv")))
ties = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/ties.tsv")))
hundreds = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/hundreds.tsv")))


def get_one_to_one_thousand(cardinal: "pynini.FstLike") -> "pynini.FstLike":
    """
    Produces an acceptor for verbalizations of all numbers from 1 to 1000. Needed for ordinals and fractions.

    Args:
        cardinal: CardinalFst

    Returns:
        fst: A pynini.FstLike object
    """
    numbers = pynini.string_map([str(_) for _ in range(1, 1000)]) @ cardinal
    return pynini.project(numbers, "output").optimize()


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
                "21.º" -> ordinal { integer: "vigésimo primero" morphosyntactic_features: "gender_masc" }
    This class converts ordinal up to the millionth (millonésimo) order (exclusive).

    This FST also records the ending of the ordinal (called "morphosyntactic_features"):
    either as gender_masc, gender_fem, or apocope. Also introduces plural feature for non-deterministic graphs.

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify")
        cardinal_graph = cardinal.graph

        graph_digit = digit.optimize()
        graph_teens = teens.optimize()
        graph_ties = ties.optimize()
        graph_twenties = twenties.optimize()
        graph_hundreds = hundreds.optimize()

        if not deterministic:
            # Some alternative derivations
            graph_ties = graph_ties | pynini.cross("sesenta", "setuagésimo")

            graph_teens = graph_teens | pynini.cross("once", "decimoprimero")
            graph_teens |= pynini.cross("doce", "decimosegundo")

            graph_digit = graph_digit | pynini.cross("nueve", "nono")
            graph_digit |= pynini.cross("siete", "sétimo")

        graph_tens_component = (
            graph_teens
            | (graph_ties + pynini.closure(pynini.cross(" y ", DAMO_SPACE) + graph_digit, 0, 1))
            | graph_twenties
        )

        graph_hundred_component = pynini.union(
            graph_hundreds
            + pynini.closure(DAMO_SPACE + pynini.union(graph_tens_component, graph_digit), 0, 1),
            graph_tens_component,
            graph_digit,
        )

        # Need to go up to thousands for fractions
        self.one_to_one_thousand = get_one_to_one_thousand(cardinal_graph)

        thousands = pynini.cross("mil", "milésimo")

        graph_thousands = (
            strip_accent(self.one_to_one_thousand) + DAMO_SPACE + thousands
        )  # Cardinals become prefix for thousands series. Snce accent on the powers of ten we strip accent from leading words
        graph_thousands @= pynini.cdrewrite(
            delete_space, "", "milésimo", DAMO_SIGMA
        )  # merge as a prefix
        graph_thousands |= thousands

        self.multiples_of_thousand = (cardinal_graph @ graph_thousands).optimize()

        if (
            not deterministic
        ):  # Formally the words preceding the power of ten should be a prefix, but some maintain word boundaries.
            graph_thousands |= (
                (self.one_to_one_thousand @ graph_hundred_component) + DAMO_SPACE + thousands
            )

        graph_thousands += pynini.closure(DAMO_SPACE + graph_hundred_component, 0, 1)

        ordinal_graph = graph_thousands | graph_hundred_component
        ordinal_graph = cardinal_graph @ ordinal_graph

        if not deterministic:
            # The 10's and 20's series can also be two words
            split_words = pynini.cross("decimo", "décimo ") | pynini.cross("vigesimo", "vigésimo ")
            split_words = pynini.cdrewrite(split_words, "", DAMO_CHAR, DAMO_SIGMA)
            ordinal_graph |= ordinal_graph @ split_words

        # If "octavo" is preceeded by the "o" within string, it needs deletion
        ordinal_graph @= pynini.cdrewrite(pynutil.delete("o"), "", "octavo", DAMO_SIGMA)

        self.graph = ordinal_graph.optimize()

        masc = pynini.accep("gender_masc")
        fem = pynini.accep("gender_fem")
        apocope = pynini.accep("apocope")

        delete_period = pynini.closure(
            pynutil.delete("."), 0, 1
        )  # Sometimes the period is omitted f

        accept_masc = delete_period + pynini.cross("º", masc)
        accep_fem = delete_period + pynini.cross("ª", fem)
        accep_apocope = delete_period + pynini.cross("ᵉʳ", apocope)

        # Managing Romanization
        graph_roman = (
            pynutil.insert('integer: "') + roman_to_int(ordinal_graph) + pynutil.insert('"')
        )
        if not deterministic:
            # Introduce plural
            plural = pynini.closure(pynutil.insert("/plural"), 0, 1)
            accept_masc += plural
            accep_fem += plural

            # Romanizations have no morphology marker, so in non-deterministic case we provide option for all
            insert_morphology = pynutil.insert(pynini.union(masc, fem)) + plural
            insert_morphology |= pynutil.insert(apocope)
            insert_morphology = (
                pynutil.insert(' morphosyntactic_features: "')
                + insert_morphology
                + pynutil.insert('"')
            )

            graph_roman += insert_morphology

        else:
            # We insert both genders as default
            graph_roman += pynutil.insert(
                ' morphosyntactic_features: "gender_masc"'
            ) | pynutil.insert(' morphosyntactic_features: "gender_fem"')

        # Rest of graph
        convert_abbreviation = accept_masc | accep_fem | accep_apocope

        graph = (
            pynutil.insert('integer: "')
            + ordinal_graph
            + pynutil.insert('"')
            + pynutil.insert(' morphosyntactic_features: "')
            + convert_abbreviation
            + pynutil.insert('"')
        )
        graph = pynini.union(graph, graph_roman)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
