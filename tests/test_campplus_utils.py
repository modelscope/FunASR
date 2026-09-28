"""Speaker assignment uses the actual overlap with each sentence."""

from itertools import permutations

import numpy as np
import pytest

from funasr.models.campplus.utils import distribute_spk


def test_distribute_spk_does_not_double_count_the_first_overlap():
    sentences = [{"start": 0, "end": 10000, "text": "first and second"}]

    result = distribute_spk(sentences, [(0, 4, 0), (4, 10, 1)])

    assert result == [{"start": 0, "end": 10000, "text": "first and second", "spk": 1}]


@pytest.mark.parametrize(
    "timeline",
    list(permutations([(0, 2, 7), (2, 5, 9), (5, 7, 7), (7, 10, 9)])),
)
def test_distribute_spk_totals_nonconsecutive_turns_independent_of_order(timeline):
    sentences = [{"start": 0, "end": 10000}]

    assert distribute_spk(sentences, timeline) == [{"start": 0, "end": 10000, "spk": 9}]


@pytest.mark.parametrize(
    ("timeline", "expected"),
    [
        ([(0, 2, 7), (2, 7, 9), (7, 10, 7)], 7),
        ([(2, 7, 9), (0, 2, 7), (7, 10, 7)], 9),
        ([(-2, 0, 9), (0, 5, 7), (5, 10, 9)], 7),
    ],
)
def test_distribute_spk_breaks_total_ties_by_first_positive_overlap(timeline, expected):
    assert distribute_spk([{"start": 0, "end": 10000}], timeline) == [
        {"start": 0, "end": 10000, "spk": expected}
    ]


def test_distribute_spk_clips_to_each_sentence_and_preserves_objects():
    first = {"start": 4000, "end": 6000, "text": "clipped", "spk": 99}
    second = {"start": 6000, "end": 7000, "timestamp": [[6000, 7000]]}
    sentences = [first, second]
    timeline = [(0, 4, 8), (4, 5.5, np.int64(3)), (5.5, 100, np.int64(9))]

    result = distribute_spk(sentences, timeline)

    assert result is sentences
    assert result[0] is first
    assert result[1] is second
    assert result == [
        {"start": 4000, "end": 6000, "text": "clipped", "spk": 3},
        {"start": 6000, "end": 7000, "timestamp": [[6000, 7000]], "spk": 9},
    ]
    assert type(first["spk"]) is int
    assert type(second["spk"]) is int


@pytest.mark.parametrize("timeline", [[], [(0, 1, 8), (2, 3, 9)]])
def test_distribute_spk_defaults_to_zero_without_positive_overlap(timeline):
    assert distribute_spk([{"start": 1000, "end": 2000}], timeline) == [
        {"start": 1000, "end": 2000, "spk": 0}
    ]


def test_distribute_spk_accepts_empty_sentences():
    sentences = []
    assert distribute_spk(sentences, [(0, 1, 7)]) is sentences
