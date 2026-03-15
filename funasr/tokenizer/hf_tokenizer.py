from funasr.register import tables


@tables.register("tokenizer_classes", "HuggingfaceTokenizer")
def HuggingfaceTokenizer(init_param_path, **kwargs):
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        raise ImportError(
            "HuggingfaceTokenizer requires 'transformers'. "
            "Please install it with: pip install -U transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(init_param_path)

    return tokenizer
