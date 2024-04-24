import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_ALPHA, DAMO_SIGMA, GraphFst
from fun_text_processing.text_normalization.en.utils import get_abs_path, load_labels
from pynini.lib import pynutil


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying roman numbers:
        e.g. "IV" -> tokens { roman { integer: "four" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        roman_dict = load_labels(get_abs_path("data/roman/roman_to_spoken.tsv"))
        default_graph = pynini.string_map(roman_dict).optimize()
        default_graph = pynutil.insert('integer: "') + default_graph + pynutil.insert('"')
        ordinal_limit = 19

        if deterministic:
            # exclude "I"
            start_idx = 1
        else:
            start_idx = 0

        graph_teens = pynini.string_map(
            [x[0] for x in roman_dict[start_idx:ordinal_limit]]
        ).optimize()

        # roman numerals up to ordinal_limit with a preceding name are converted to ordinal form
        names = get_names()
        graph = (
            pynutil.insert('key_the_ordinal: "')
            + names
            + pynutil.insert('"')
            + pynini.accep(" ")
            + graph_teens @ default_graph
        ).optimize()

        # single symbol roman numerals with preceding key words (multiple formats) are converted to cardinal form
        key_words = []
        for k_word in load_labels(get_abs_path("data/roman/key_word.tsv")):
            key_words.append(k_word)
            key_words.append([k_word[0][0].upper() + k_word[0][1:]])
            key_words.append([k_word[0].upper()])

        key_words = pynini.string_map(key_words).optimize()
        graph |= (
            pynutil.insert('key_cardinal: "')
            + key_words
            + pynutil.insert('"')
            + pynini.accep(" ")
            + default_graph
        ).optimize()

        if deterministic or lm:
            # two digit roman numerals up to 49
            roman_to_cardinal = pynini.compose(
                pynini.closure(DAMO_ALPHA, 2),
                (
                    pynutil.insert('default_cardinal: "default" ')
                    + (pynini.string_map([x[0] for x in roman_dict[:50]]).optimize())
                    @ default_graph
                ),
            )
            graph |= roman_to_cardinal
        elif not lm:
            # two or more digit roman numerals
            roman_to_cardinal = pynini.compose(
                pynini.difference(DAMO_SIGMA, "I"),
                (
                    pynutil.insert('default_cardinal: "default" integer: "')
                    + pynini.string_map(roman_dict).optimize()
                    + pynutil.insert('"')
                ),
            ).optimize()
            graph |= roman_to_cardinal

        # convert three digit roman or up with suffix to ordinal
        roman_to_ordinal = pynini.compose(
            pynini.closure(DAMO_ALPHA, 3),
            (
                pynutil.insert('default_ordinal: "default" ')
                + graph_teens @ default_graph
                + pynutil.delete("th")
            ),
        )

        graph |= roman_to_ordinal
        graph = self.add_tokens(graph.optimize())

        self.fst = graph.optimize()


def get_names():
    """
    Returns the graph that matched common male and female names.
    """
    male_labels = load_labels(get_abs_path("data/roman/male.tsv"))
    female_labels = load_labels(get_abs_path("data/roman/female.tsv"))
    male_labels.extend([[x[0].upper()] for x in male_labels])
    female_labels.extend([[x[0].upper()] for x in female_labels])
    names = pynini.string_map(male_labels).optimize()
    names |= pynini.string_map(female_labels).optimize()
    return names
