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
