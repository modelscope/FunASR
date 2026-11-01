"""Measure serial end-to-end runtime RTF, including frontend and audio loading."""


def main(argv=None):
    if __package__:
        from .runtime_eval import benchmark, parse_args
    else:
        from runtime_eval import benchmark, parse_args

    benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
