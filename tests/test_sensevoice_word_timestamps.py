"""Word timestamps use the same SentencePiece boundaries for the first word."""

from copy import deepcopy

import pytest

from funasr.models.sense_voice.model import SenseVoiceSmall


@pytest.mark.parametrize(
    ("tokens", "words", "timestamps"),
    [
        pytest.param(
            ["\u2581Hello", "\u2581world", "."],
            ["Hello", "world", "."],
            [[0, 125], [125, 250], [250, 375]],
            id="first-word-marker",
        ),
        pytest.param(
            ["\u2581Hel", "lo", "\u2581world", "."],
            ["Hello", "world", "."],
            [[0, 250], [250, 375], [375, 500]],
            id="first-word-subpieces",
        ),
        pytest.param(
            ["Hel", "lo", "\u2581wor", "ld", "."],
            ["Hello", "world", "."],
            [[0, 250], [250, 500], [500, 625]],
            id="unmarked-first-word-and-later-boundary",
        ),
        pytest.param(
            ["\u2581", "\u2581Hel", "lo"],
            ["Hello"],
            [[125, 375]],
            id="standalone-leading-marker",
        ),
        pytest.param(
            ["\u2581", "Hel", "lo"],
            ["Hello"],
            [[125, 375]],
            id="standalone-marker-before-unmarked-word",
        ),
        pytest.param(
            ["\u2581\u4f60", "\u597d", "\u3002"],
            ["\u4f60", "\u597d", "\u3002"],
            [[0, 125], [125, 250], [250, 375]],
            id="chinese-first-word",
        ),
        pytest.param(
            ["\u2581Hello", "\u4e16", "\u754c"],
            ["Hello", "\u4e16", "\u754c"],
            [[0, 125], [125, 250], [250, 375]],
            id="mixed-script-boundaries",
        ),
        pytest.param(["\u2581", "\u2581"], [], [], id="only-markers"),
        pytest.param([], [], [], id="empty"),
        pytest.param(
            ["under\u2581score"], ["under\u2581score"], [[0, 125]], id="nonleading-marker"
        ),
        pytest.param(
            ["\u2581I'm", "\u2581here", "."],
            ["I'm", "here", "."],
            [[0, 125], [125, 250], [250, 375]],
            id="preserve-punctuation",
        ),
        pytest.param(
            ["A", "\u2581B", "C"], ["A", "BC"], [[0, 125], [125, 375]], id="word-boundary"
        ),
        pytest.param(
            ["Hello", "\u2581", "world"],
            ["Hello", "world"],
            [[0, 125], [250, 375]],
            id="standalone-word-boundary",
        ),
        pytest.param(
            ["\u2581Hel", "lo", "\u2581", "wor", "ld"],
            ["Hello", "world"],
            [[0, 250], [375, 625]],
            id="subpieces-around-standalone-boundary",
        ),
        pytest.param(
            ["Hello", "\u2581", "\u2581", "world"],
            ["Hello", "world"],
            [[0, 125], [375, 500]],
            id="repeated-standalone-boundaries",
        ),
    ],
)
def test_post_preserves_word_boundaries_and_timestamp_alignment(tokens, words, timestamps):
    token_timestamps = [[token, index / 8, (index + 1) / 8] for index, token in enumerate(tokens)]
    original = deepcopy(token_timestamps)

    # post uses no model state; importing the real class requires no checkpoint.
    result_timestamps, result_words = SenseVoiceSmall.post(None, token_timestamps)

    assert result_words == words
    assert result_timestamps == timestamps
    assert len(result_words) == len(result_timestamps)
    assert token_timestamps == original
