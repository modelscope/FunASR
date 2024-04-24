import csv
import os

import logging


def get_abs_path(rel_path):
    """
    Get absolute path

    Args:
        rel_path: relative path to this file

    Returns absolute path
    """
    abs_path = os.path.dirname(os.path.abspath(__file__)) + os.sep + rel_path

    if not os.path.exists(abs_path):
        logging.warning(f"{abs_path} does not exist")
    return abs_path


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
