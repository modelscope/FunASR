from pathlib import Path
from typing import Iterable
from typing import Union


def build_tokenizer(
    token_type: str,
    bpemodel: Union[Path, str, Iterable[str]] = None,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
    g2p_type: str = None,
    p_word2phn: float = 0.5,
):
    if "whisper_rich_ttsfrd" in token_type:
        from funasr.models.llm_asr.tts_text_tokenizer.whisper_tokenizer import WhisperRichTtsFrdTokenizer
        return WhisperRichTtsFrdTokenizer(
            token_path="multilingual_zh_ja_yue_char_del",
            num_languages=105,
            task=None,
            language=None,
            ttsfrd_type="ttsfrd_rich",
            ttsfrd_model=bpemodel,
            p_word2phn=p_word2phn,
        )

    else:
        raise ValueError(
            f"token_mode must be one of bpe, word, char or phn: " f"{token_type}"
        )
