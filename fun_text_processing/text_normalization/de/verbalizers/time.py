import pynini
from fun_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_DIGIT,
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    delete_preserve_order,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic, e.g.
        time { hours: "2" minutes: "15"} -> "zwei uhr fünfzehn"
        time { minutes: "15" hours: "2" } -> "viertel nach zwei"
        time { minutes: "15" hours: "2" } -> "fünfzehn nach zwei"
        time { hours: "14" minutes: "15"} -> "vierzehn uhr fünfzehn"
        time { minutes: "15" hours: "14" } -> "viertel nach zwei"
        time { minutes: "15" hours: "14" } -> "fünfzehn nach drei"
        time { minutes: "45" hours: "14" } -> "viertel vor drei"

    Args:
        cardinal_tagger: cardinal_tagger tagger GraphFst
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal_tagger: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        # add weight so when using inverse text normalization this conversion is depriotized
        night_to_early = pynutil.add_weight(
            pynini.invert(
                pynini.string_file(get_abs_path("data/time/hour_to_night.tsv"))
            ).optimize(),
            weight=0.0001,
        )
        hour_to = pynini.invert(
            pynini.string_file(get_abs_path("data/time/hour_to.tsv"))
        ).optimize()
        minute_to = pynini.invert(
            pynini.string_file(get_abs_path("data/time/minute_to.tsv"))
        ).optimize()
        time_zone_graph = pynini.invert(
            convert_space(
                pynini.union(*[x[1] for x in load_labels(get_abs_path("data/time/time_zone.tsv"))])
            )
        )

        graph_zero = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        ).optimize()
        number_verbalization = graph_zero | cardinal_tagger.two_digit_non_zero
        hour = pynutil.delete('hours: "') + pynini.closure(DAMO_DIGIT, 1) + pynutil.delete('"')
        hour_verbalized = hour @ number_verbalization @ pynini.cdrewrite(
            pynini.cross("eins", "ein"), "[BOS]", "[EOS]", DAMO_SIGMA
        ) + pynutil.insert(" uhr")
        minute = pynutil.delete('minutes: "') + pynini.closure(DAMO_DIGIT, 1) + pynutil.delete('"')
        zone = pynutil.delete('zone: "') + time_zone_graph + pynutil.delete('"')
        optional_zone = pynini.closure(pynini.accep(" ") + zone, 0, 1)
        second = pynutil.delete('seconds: "') + pynini.closure(DAMO_DIGIT, 1) + pynutil.delete('"')
        graph_hms = (
            hour_verbalized
            + pynini.accep(" ")
            + minute @ number_verbalization
            + pynutil.insert(" minuten")
            + pynini.accep(" ")
            + second @ number_verbalization
            + pynutil.insert(" sekunden")
            + optional_zone
        )
        graph_hms @= pynini.cdrewrite(
            pynini.cross("eins minuten", "eine minute")
            | pynini.cross("eins sekunden", "eine sekunde"),
            pynini.union(" ", "[BOS]"),
            "",
            DAMO_SIGMA,
        )

        min_30 = [str(x) for x in range(1, 31)]
        min_30 = pynini.union(*min_30)
        min_29 = [str(x) for x in range(1, 30)]
        min_29 = pynini.union(*min_29)

        graph_h = hour_verbalized
        graph_hm = hour_verbalized + pynini.accep(" ") + minute @ number_verbalization

        graph_m_past_h = (
            minute @ min_30 @ (number_verbalization | pynini.cross("15", "viertel"))
            + pynini.accep(" ")
            + pynutil.insert("nach ")
            # + hour @ number_verbalization
            + hour
            @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", DAMO_SIGMA)
            @ number_verbalization
        )
        graph_m30_h = (
            minute @ pynini.cross("30", "halb")
            + pynini.accep(" ")
            + hour
            @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", DAMO_SIGMA)
            @ hour_to
            @ number_verbalization
        )
        graph_m_to_h = (
            minute @ minute_to @ min_29 @ (number_verbalization | pynini.cross("15", "viertel"))
            + pynini.accep(" ")
            + pynutil.insert("vor ")
            + hour
            @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", DAMO_SIGMA)
            @ hour_to
            @ number_verbalization
        )

        self.graph = (
            graph_hms
            | graph_h
            | graph_hm
            | pynutil.add_weight(graph_m_past_h, weight=0.0001)
            | pynutil.add_weight(graph_m30_h, weight=0.0001)
            | pynutil.add_weight(graph_m_to_h, weight=0.0001)
        ) + optional_zone
        delete_tokens = self.delete_tokens(self.graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
