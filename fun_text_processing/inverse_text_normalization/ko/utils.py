import csv
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


def load_labels(abs_path):
    """
    loads relative path file as dictionary

    Args:
        abs_path: absolute path

    Returns dictionary of mappings
    """
    label_tsv = open(abs_path, encoding="utf-8")
    labels = list(csv.reader(label_tsv, delimiter="\t"))
    return labels


def augment_labels_with_punct_at_end(labels):
    """
    augments labels: if key ends on a punctuation that value does not have, add a new label
    where the value maintains the punctuation

    Args:
        labels : input labels
    Returns:
        additional labels
    """
    res = []
    for label in labels:
        if len(label) > 1:
            if label[0][-1] == "." and label[1][-1] != ".":
                res.append([label[0], label[1] + "."] + label[2:])
    return res
