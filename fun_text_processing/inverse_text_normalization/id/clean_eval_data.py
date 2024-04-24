from argparse import ArgumentParser
from typing import List

import regex as re
from fun_text_processing.text_normalization.data_loader_utils import (
    EOS_TYPE,
    Instance,
    load_files,
    training_data_to_sentences,
)


"""
This file is for evaluation purposes.
filter_loaded_data() cleans data (list of instances) for inverse text normalization. Filters and cleaners can be specified for each semiotic class individually.
For example, normalized text should only include characters and whitespace characters but no punctuation. 
            Cardinal unnormalized instances should contain at least one integer and all other characters are removed.
"""


class Filter:
    """
    Filter class

    Args:
        class_type: semiotic class used in dataset
        process_func: function to transform text
        filter_func:  function to filter text

    """

    def __init__(self, class_type: str, process_func: object, filter_func: object):
        self.class_type = class_type
        self.process_func = process_func
        self.filter_func = filter_func

    def filter(self, instance: Instance) -> bool:
        """
        filter function

        Args:
            filters given instance with filter function

        Returns: True if given instance fulfills criteria or does not belong to class type
        """
        if instance.token_type != self.class_type:
            return True
        return self.filter_func(instance)

    def process(self, instance: Instance) -> Instance:
        """
        process function

        Args:
            processes given instance with process function

        Returns: processed instance if instance belongs to expected class type or original instance
        """
        if instance.token_type != self.class_type:
            return instance
        return self.process_func(instance)


def filter_cardinal_1(instance: Instance) -> bool:
    ok = re.search(r"[0-9]", instance.un_normalized)
    return ok


def process_cardinal_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    un_normalized = re.sub(r"[^0-9]", "", un_normalized)
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_ordinal_1(instance: Instance) -> bool:
    ok = re.search(r"(st|nd|rd|th)\s*$", instance.un_normalized)
    return ok


def process_ordinal_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    un_normalized = re.sub(r"[,\s]", "", un_normalized)
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_decimal_1(instance: Instance) -> bool:
    ok = re.search(r"[0-9]", instance.un_normalized)
    return ok


def process_decimal_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    un_normalized = re.sub(r",", "", un_normalized)
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_measure_1(instance: Instance) -> bool:
    ok = True
    return ok


def process_measure_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    un_normalized = re.sub(r",", "", un_normalized)
    un_normalized = re.sub(r"m2", "m²", un_normalized)
    un_normalized = re.sub(r"(\d)([^\d.\s])", r"\1 \2", un_normalized)
    normalized = re.sub(r"[^a-z\s]", "", normalized)
    normalized = re.sub(r"per ([a-z\s]*)s$", r"per \1", normalized)
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_money_1(instance: Instance) -> bool:
    ok = re.search(r"[0-9]", instance.un_normalized)
    return ok


def process_money_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    un_normalized = re.sub(r",", "", un_normalized)
    un_normalized = re.sub(r"a\$", r"$", un_normalized)
    un_normalized = re.sub(r"us\$", r"$", un_normalized)
    un_normalized = re.sub(r"(\d)m\s*$", r"\1 juta", un_normalized)
    un_normalized = re.sub(r"(\d)bn?\s*$", r"\1 milyar", un_normalized)
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_time_1(instance: Instance) -> bool:
    ok = re.search(r"[0-9]", instance.un_normalized)
    return ok


def process_time_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    un_normalized = re.sub(r": ", ":", un_normalized)
    un_normalized = re.sub(r"(\d)\s?a\s?m\s?", r"\1 a.m.", un_normalized)
    un_normalized = re.sub(r"(\d)\s?p\s?m\s?", r"\1 p.m.", un_normalized)
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_plain_1(instance: Instance) -> bool:
    ok = True
    return ok


def process_plain_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_punct_1(instance: Instance) -> bool:
    ok = True
    return ok


def process_punct_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_date_1(instance: Instance) -> bool:
    ok = True
    return ok


def process_date_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    un_normalized = re.sub(r",", "", un_normalized)
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_letters_1(instance: Instance) -> bool:
    ok = True
    return ok


def process_letters_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_verbatim_1(instance: Instance) -> bool:
    ok = True
    return ok


def process_verbatim_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_digit_1(instance: Instance) -> bool:
    ok = re.search(r"[0-9]", instance.un_normalized)
    return ok


def process_digit_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_telephone_1(instance: Instance) -> bool:
    ok = re.search(r"[0-9]", instance.un_normalized)
    return ok


def process_telephone_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_electronic_1(instance: Instance) -> bool:
    ok = re.search(r"[0-9]", instance.un_normalized)
    return ok


def process_electronic_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_fraction_1(instance: Instance) -> bool:
    ok = re.search(r"[0-9]", instance.un_normalized)
    return ok


def process_fraction_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


def filter_address_1(instance: Instance) -> bool:
    ok = True
    return ok


def process_address_1(instance: Instance) -> Instance:
    un_normalized = instance.un_normalized
    normalized = instance.normalized
    normalized = re.sub(r"[^a-z ]", "", normalized)
    return Instance(
        token_type=instance.token_type, un_normalized=un_normalized, normalized=normalized
    )


filters = []
filters.append(
    Filter(class_type="CARDINAL", process_func=process_cardinal_1, filter_func=filter_cardinal_1)
)
filters.append(
    Filter(class_type="ORDINAL", process_func=process_ordinal_1, filter_func=filter_ordinal_1)
)
filters.append(
    Filter(class_type="DECIMAL", process_func=process_decimal_1, filter_func=filter_decimal_1)
)
filters.append(
    Filter(class_type="MEASURE", process_func=process_measure_1, filter_func=filter_measure_1)
)
filters.append(Filter(class_type="MONEY", process_func=process_money_1, filter_func=filter_money_1))
filters.append(Filter(class_type="TIME", process_func=process_time_1, filter_func=filter_time_1))

filters.append(Filter(class_type="DATE", process_func=process_date_1, filter_func=filter_date_1))
filters.append(Filter(class_type="PLAIN", process_func=process_plain_1, filter_func=filter_plain_1))
filters.append(Filter(class_type="PUNCT", process_func=process_punct_1, filter_func=filter_punct_1))
filters.append(
    Filter(class_type="LETTERS", process_func=process_letters_1, filter_func=filter_letters_1)
)
filters.append(
    Filter(class_type="VERBATIM", process_func=process_verbatim_1, filter_func=filter_verbatim_1)
)
filters.append(Filter(class_type="DIGIT", process_func=process_digit_1, filter_func=filter_digit_1))
filters.append(
    Filter(class_type="TELEPHONE", process_func=process_telephone_1, filter_func=filter_telephone_1)
)
filters.append(
    Filter(
        class_type="ELECTRONIC", process_func=process_electronic_1, filter_func=filter_electronic_1
    )
)
filters.append(
    Filter(class_type="FRACTION", process_func=process_fraction_1, filter_func=filter_fraction_1)
)
filters.append(
    Filter(class_type="ADDRESS", process_func=process_address_1, filter_func=filter_address_1)
)
filters.append(Filter(class_type=EOS_TYPE, process_func=lambda x: x, filter_func=lambda x: True))


def filter_loaded_data(data: List[Instance], verbose: bool = False) -> List[Instance]:
    """
    Filters list of instances

    Args:
        data: list of instances

    Returns: filtered and transformed list of instances
    """
    updates_instances = []
    for instance in data:
        updated_instance = False
        for fil in filters:
            if fil.class_type == instance.token_type and fil.filter(instance):
                instance = fil.process(instance)
                updated_instance = True
        if updated_instance:
            if verbose:
                print(instance)
            updates_instances.append(instance)
    return updates_instances


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input", help="input file path", type=str, default="./id_with_types/output-00001-of-00100"
    )
    parser.add_argument("--verbose", help="print filtered instances", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.input

    print("Loading training data: " + file_path)
    instance_list = load_files([file_path])  # List of instances
    filtered_instance_list = filter_loaded_data(instance_list, args.verbose)
    training_data_to_sentences(filtered_instance_list)
