#!/usr/bin/env python
import sys


def get_commandline_args(no_executable=True):
    extra_chars = [
        " ",
        ";",
        "&",
        "|",
        "<",
        ">",
        "?",
        "*",
        "~",
        "`",
        '"',
        "'",
        "\\",
        "{",
        "}",
        "(",
        ")",
    ]

    # Escape the extra characters for shell
    argv = [
        arg.replace("'", "'\\''")
        if all(char not in arg for char in extra_chars)
        else "'" + arg.replace("'", "'\\''") + "'"
        for arg in sys.argv
    ]

    if no_executable:
        return " ".join(argv[1:])
    else:
        return sys.executable + " " + " ".join(argv)


def main():
    print(get_commandline_args())


if __name__ == "__main__":
    main()
