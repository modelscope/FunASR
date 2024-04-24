import pynini
from fun_text_processing.inverse_text_normalization.pt.utils import get_abs_path
from fun_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. quinze pro meio dia -> time { hours: "11" minutes: "45" }
        e.g. quinze pra meia noite -> time { hours: "23" minutes: "45" }
        e.g. quinze pra uma -> time { hours: "12" minutes: "45" }
        e.g. dez pras duas -> time { hours: "1" minutes: "50" }
        e.g. quinze pras duas -> time { hours: "1" minutes: "45" }
        e.g. ao meio dia -> time { hours: "12" minutes: "00" morphosyntactic_features: "ao" }
        e.g. ao meio dia e meia -> time { hours: "12" minutes: "30" morphosyntactic_features: "ao" }
        e.g. ao meio dia e meio -> time { hours: "12" minutes: "30" morphosyntactic_features: "ao" }
        e.g. à meia noite e quinze -> time { hours: "0" minutes: "15" morphosyntactic_features: "à" }
        e.g. à meia noite e meia -> time { hours: "0" minutes: "30" morphosyntactic_features: "à" }
        e.g. à uma e trinta -> time { hours: "1" minutes: "30" morphosyntactic_features: "à" }
        e.g. às onze e trinta -> time { hours: "11" minutes: "30" morphosyntactic_features: "às" }
        e.g. às três horas e trinta minutos -> time { hours: "3" minutes: "30" morphosyntactic_features: "às" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")

        # graph_hour_to_am = pynini.string_file(get_abs_path("data/time/hour_to_am.tsv"))
        # graph_hour_to_pm = pynini.string_file(get_abs_path("data/time/hour_to_pm.tsv"))
        graph_hours_to = pynini.string_file(get_abs_path("data/time/hours_to.tsv"))
        graph_minutes_to = pynini.string_file(get_abs_path("data/time/minutes_to.tsv"))
        graph_suffix_am = pynini.string_file(get_abs_path("data/time/time_suffix_am.tsv"))
        graph_suffix_pm = pynini.string_file(get_abs_path("data/time/time_suffix_pm.tsv"))

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))

        graph_1_to_100 = pynini.union(
            graph_digit,
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + pynutil.delete(" e ") + graph_digit),
        )

        # note that graph_hour will start from 2 hours
        # "1 o'clock" will be treated differently because it
        # is singular
        digits_2_to_23 = [str(digits) for digits in range(2, 24)]
        digits_1_to_59 = [str(digits) for digits in range(1, 60)]

        graph_2_to_23 = graph_1_to_100 @ pynini.union(*digits_2_to_23)
        graph_1_to_59 = graph_1_to_100 @ pynini.union(*digits_1_to_59)
        graph_uma = pynini.cross("uma", "1")

        # Mapping 'horas'
        graph_hour = pynutil.delete(pynini.accep("hora") + pynini.accep("s").ques)
        graph_minute = pynutil.delete(pynini.accep("minuto") + pynini.accep("s").ques)

        # Mapping 'meio dia' and 'meia noite'
        graph_meio_dia = pynini.cross("meio dia", "12")
        graph_meia_noite = pynini.cross("meia noite", "0")

        # Mapping 'e meia'
        graph_e = delete_space + pynutil.delete(" e ") + delete_space
        graph_e_meia = graph_e + pynini.cross("meia", "30")
        graph_e_meio = graph_e + pynini.cross("meio", "30")

        # à uma e meia -> 1:30
        # às três e meia -> 3:30
        graph_hours_at_prefix_singular = (
            pynutil.insert('morphosyntactic_features: "')
            + (pynini.cross("à", "à") | pynini.cross("a", "à"))
            + pynutil.insert('" ')
            + delete_space
        )
        graph_hours_at_singular = (
            graph_hours_at_prefix_singular
            + pynutil.insert('hours: "')
            + graph_uma
            + pynutil.insert('"')
            + (delete_space + graph_hour).ques
        )
        graph_hours_at_prefix_plural = (
            pynutil.insert('morphosyntactic_features: "')
            + (pynini.cross("às", "às") | pynini.cross("as", "às"))
            + pynutil.insert('" ')
            + delete_space
        )
        graph_hours_at_plural = (
            graph_hours_at_prefix_plural
            + pynutil.insert('hours: "')
            + graph_2_to_23
            + pynutil.insert('"')
            + (delete_space + graph_hour).ques
        )
        final_graph_hour_at = graph_hours_at_singular | graph_hours_at_plural

        graph_minutes_component_without_zero = (
            graph_e + graph_1_to_59 + (delete_space + graph_minute).ques
        )
        graph_minutes_component_without_zero |= (
            graph_e_meia + pynutil.delete(delete_space + pynini.accep("hora")).ques
        )
        final_graph_minute = (
            pynutil.insert(' minutes: "')
            + graph_minutes_component_without_zero
            + pynutil.insert('"')
        )

        graph_hm = final_graph_hour_at + final_graph_minute

        # à uma hora -> 1:00
        graph_hours_at_singular_with_hour = (
            graph_hours_at_prefix_singular
            + pynutil.insert('hours: "')
            + graph_uma
            + pynutil.insert('"')
            + delete_space
            + graph_hour
        )

        graph_hours_at_plural_with_hour = (
            graph_hours_at_prefix_plural
            + pynutil.insert('hours: "')
            + graph_2_to_23
            + pynutil.insert('"')
            + delete_space
            + graph_hour
        )

        graph_hm |= (
            graph_hours_at_singular_with_hour | graph_hours_at_plural_with_hour
        ) + pynutil.insert(' minutes: "00"', weight=0.2)

        # meio dia e meia -> 12:30
        # meia noite e meia -> 0:30
        graph_minutes_without_zero = (
            pynutil.insert(' minutes: "')
            + graph_minutes_component_without_zero
            + pynutil.insert('"')
        )
        graph_meio_min = (
            pynutil.insert('hours: "')
            + (graph_meio_dia | graph_meia_noite)
            + pynutil.insert('"')
            + graph_minutes_without_zero
        )
        graph_meio_min |= (
            pynutil.insert('hours: "')
            + graph_meio_dia
            + pynutil.insert('" minutes: "')
            + graph_e_meio
            + pynutil.insert('"')
        )
        graph_hm |= graph_meio_min

        # às quinze para as quatro -> às 3:45
        # NOTE: case 'para à uma' ('to one') could be either 0:XX or 12:XX
        #       leading to wrong reading ('meio dia e ...' or 'meia noite e ...')
        graph_para_a = (
            pynutil.delete("para")
            | pynutil.delete("para a")
            | pynutil.delete("para as")
            | pynutil.delete("pra")
            | pynutil.delete("pras")
        )
        graph_para_o = pynutil.delete("para") | pynutil.delete("para o") | pynutil.delete("pro")

        graph_pra_min = (
            pynutil.insert('morphosyntactic_features: "')
            + (
                pynini.cross("à", "à")
                | pynini.cross("às", "às")
                | pynini.cross("a", "à")
                | pynini.cross("as", "às")
            )
            + pynutil.insert('" ')
            + delete_space
        )
        graph_pra_min += (
            pynutil.insert('minutes: "')
            + (graph_1_to_59 @ graph_minutes_to)
            + pynutil.insert('" ')
            + (delete_space + graph_minute).ques
        )
        graph_pra_hour = (
            pynutil.insert('hours: "')
            + (graph_2_to_23 @ graph_hours_to)
            + pynutil.insert('"')
            + (delete_space + graph_hour).ques
        )
        graph_pra_hour |= (
            pynutil.insert('hours: "') + (graph_meia_noite @ graph_hours_to) + pynutil.insert('"')
        )

        graph_pra = graph_pra_min + delete_space + graph_para_a + delete_space + graph_pra_hour

        # às quinze pro meio dia -> às 11:45
        graph_pro = graph_pra_min + delete_space + graph_para_o + delete_space
        graph_pro += (
            pynutil.insert(' hours: "') + (graph_meio_dia @ graph_hours_to) + pynutil.insert('"')
        )

        graph_mh = graph_pra | graph_pro

        # optional suffix
        final_suffix = (
            pynutil.insert('suffix: "') + (graph_suffix_am | graph_suffix_pm) + pynutil.insert('"')
        )
        final_suffix_optional = pynini.closure(delete_space + insert_space + final_suffix, 0, 1)

        final_graph = pynini.union((graph_hm | graph_mh) + final_suffix_optional).optimize()

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
