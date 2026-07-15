from fun_text_processing.text_normalization.data_loader_utils import post_process_punct


def test_post_process_punct_preserves_ascii_punctuation_spacing():
    assert post_process_punct("test' example", "test 'example") == "test' example"


def test_post_process_punct_handles_unicode_punctuation():
    assert (
        post_process_punct(
            "\u4f60\u597d\uff0c\u4e16\u754c\uff01",
            "\u4f60\u597d \uff0c \u4e16\u754c \uff01",
            add_unicode_punct=True,
        )
        == "\u4f60\u597d\uff0c\u4e16\u754c\uff01"
    )
