"""Measure batched GPU runtime RTF without silently benchmarking CPU fallback."""


def main(argv=None):
    if __package__:
        from .runtime_eval import benchmark, parse_args
    else:
        from runtime_eval import benchmark, parse_args

    benchmark(parse_args(argv, gpu=True), batched=True)


if __name__ == "__main__":
    main()
