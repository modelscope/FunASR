from funasr.register import tables


@tables.register("tokenizer_classes", "WhisperTokenizer")
def WhisperTokenizer(**kwargs):
    try:
        from whisper.tokenizer import get_tokenizer
    except:
        print("Notice: If you want to use whisper, please `pip install -U openai-whisper`")

    language = kwargs.get("language", None)
    task = kwargs.get("task", "transcribe")
    is_multilingual = kwargs.get("is_multilingual", True)
    num_languages = kwargs.get("num_languages", 99)
    tokenizer = get_tokenizer(
        multilingual=is_multilingual,
        num_languages=num_languages,
        language=language,
        task=task,
    )

    return tokenizer


@tables.register("tokenizer_classes", "SenseVoiceTokenizer")
def SenseVoiceTokenizer(**kwargs):
    try:
        from funasr.models.sense_voice.whisper_lib.tokenizer import get_tokenizer
    except:
        print("Notice: If you want to use whisper, please `pip install -U openai-whisper`")

    language = kwargs.get("language", None)
    task = kwargs.get("task", None)
    is_multilingual = kwargs.get("is_multilingual", True)
    num_languages = kwargs.get("num_languages", 8749)
    vocab_path = kwargs.get("vocab_path", None)
    tokenizer = get_tokenizer(
        multilingual=is_multilingual,
        num_languages=num_languages,
        language=language,
        task=task,
        vocab_path=vocab_path,
    )

    return tokenizer
