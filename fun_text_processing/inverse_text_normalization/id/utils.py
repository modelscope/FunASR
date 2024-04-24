import os
from typing import Union

import inflect

_inflect = inflect.engine()


def num_to_word(x: Union[str, int]):
    """
    converts integer to spoken representation

    Args
        x: integer

    Returns: spoken representation
    """
    if isinstance(x, int):
        x = str(x)
        x = _inflect.number_to_words(str(x)).replace("-", " ").replace(",", "")
    return x


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    return os.path.dirname(os.path.abspath(__file__)) + "/" + rel_path
