from funasr.utils.timestamp_tools import timestamp_sentence_en


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
