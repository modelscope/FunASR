import pytest

from funasr.utils.timestamp_tools import timestamp_sentence, timestamp_sentence_en


def test_timestamp_sentence_en_handles_whitespace_only_segment():
    result = timestamp_sentence_en(
        punc_id_list=[3],
        timestamp_postprocessed=[[0, 100]],
        text_postprocessed=" ",
        return_raw_text=True,
    )

    assert result == [
        {
            "text": ".",
            "start": 0,
            "end": 100,
            "timestamp": [[0, 100]],
            "raw_text": "",
        }
    ]


def test_timestamp_sentence_en_preserves_normal_sentence_output():
    result = timestamp_sentence_en(
        punc_id_list=[1, 3],
        timestamp_postprocessed=[[0, 100], [100, 220]],
        text_postprocessed="hello world",
        return_raw_text=True,
    )

    assert result == [
        {
            "text": "hello world.",
            "start": 0,
            "end": 220,
            "timestamp": [[0, 100], [100, 220]],
            "raw_text": "hello world",
        }
    ]


def test_timestamp_sentence_en_missing_trailing_timestamp_matches_zh_fallback():
    # timestamp_postprocessed is one entry shorter than punc_id_list/text_postprocessed,
    # the mismatch the function itself warns about. The second sentence ("bar.") has no
    # timestamp for its only word, so its start must be reported as unknown (None),
    # not silently reused from the previous, unrelated sentence.
    punc_id_list = [1, 3, 3]
    timestamp_postprocessed = [[0, 100], [100, 200]]
    text_postprocessed = "hello world bar"

    en_result = timestamp_sentence_en(punc_id_list, timestamp_postprocessed, text_postprocessed)
    zh_result = timestamp_sentence(punc_id_list, timestamp_postprocessed, text_postprocessed)

    assert en_result[1]["start"] is None
    assert en_result[1]["start"] == zh_result[1]["start"]


def test_timestamp_sentence_raw_text_keeps_token_separators():
    result = timestamp_sentence(
        punc_id_list=[1, 2, 1, 1, 3],
        timestamp_postprocessed=[[0, 100], [100, 200], [200, 300], [300, 400], [400, 500]],
        text_postprocessed="我 用 iphone pro max",
        return_raw_text=True,
    )

    assert [sentence["raw_text"] for sentence in result] == ["我 用", "iphone pro max"]


@pytest.mark.parametrize(
    ("splitter", "words", "first", "last"),
    [
        (timestamp_sentence, "你 好 世界", "你好。", "世界"),
        (timestamp_sentence_en, "hello world again", "hello world.", "again"),
    ],
)
def test_timestamp_sentence_keeps_unpunctuated_tail(splitter, words, first, last):
    result = splitter(
        [1, 3, 1],
        [[0, 100], [100, 200], [200, 300]],
        words,
        return_raw_text=True,
    )

    assert [sentence["text"].strip() for sentence in result] == [first, last]
    assert result[-1] == {
        "text": last,
        "start": 200,
        "end": 300,
        "timestamp": [[200, 300]],
        "raw_text": last,
    }


@pytest.mark.parametrize("splitter", [timestamp_sentence, timestamp_sentence_en])
def test_timestamp_sentence_does_not_emit_tail_without_its_timestamp(splitter):
    result = splitter(
        [1, 3, 1],
        [[0, 100], [100, 200]],
        "hello world again",
    )

    assert len(result) == 1
    assert result[0]["end"] == 200
