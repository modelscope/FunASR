"""Write runtime transcripts and tokens for external CER/WER scoring."""


def main(argv=None):
    if __package__:
        from .runtime_eval import parse_args, transcribe
    else:
        from runtime_eval import parse_args, transcribe

    transcribe(parse_args(argv, cer=True))


if __name__ == "__main__":
    main()
