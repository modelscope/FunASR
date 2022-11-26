from distutils.util import strtobool
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


def remove_parenthesis(value: str):
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1]
    elif value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value


def remove_quotes(value: str):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value


def int_or_none(value: str) -> Optional[int]:
    """int_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=int_or_none)
        >>> parser.parse_args(['--foo', '456'])
        Namespace(foo=456)
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return int(value)


def float_or_none(value: str) -> Optional[float]:
    """float_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=float_or_none)
        >>> parser.parse_args(['--foo', '4.5'])
        Namespace(foo=4.5)
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return float(value)


def humanfriendly_parse_size_or_none(value) -> Optional[float]:
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return humanfriendly.parse_size(value)


def str_or_int(value: str) -> Union[str, int]:
    try:
        return int(value)
    except ValueError:
        return value


def str_or_none(value: str) -> Optional[str]:
    """str_or_none.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=str_or_none)
        >>> parser.parse_args(['--foo', 'aaa'])
        Namespace(foo='aaa')
        >>> parser.parse_args(['--foo', 'none'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'null'])
        Namespace(foo=None)
        >>> parser.parse_args(['--foo', 'nil'])
        Namespace(foo=None)

    """
    if value.strip().lower() in ("none", "null", "nil"):
        return None
    return value


def str2pair_str(value: str) -> Tuple[str, str]:
    """str2pair_str.

    Examples:
        >>> import argparse
        >>> str2pair_str('abc,def ')
        ('abc', 'def')
        >>> parser = argparse.ArgumentParser()
        >>> _ = parser.add_argument('--foo', type=str2pair_str)
        >>> parser.parse_args(['--foo', 'abc,def'])
        Namespace(foo=('abc', 'def'))

    """
    value = remove_parenthesis(value)
    a, b = value.split(",")

    # Workaround for configargparse issues:
    # If the list values are given from yaml file,
    # the value givent to type() is shaped as python-list,
    # e.g. ['a', 'b', 'c'],
    # so we need to remove double quotes from it.
    return remove_quotes(a), remove_quotes(b)


def str2triple_str(value: str) -> Tuple[str, str, str]:
    """str2triple_str.

    Examples:
        >>> str2triple_str('abc,def ,ghi')
        ('abc', 'def', 'ghi')
    """
    value = remove_parenthesis(value)
    a, b, c = value.split(",")

    # Workaround for configargparse issues:
    # If the list values are given from yaml file,
    # the value givent to type() is shaped as python-list,
    # e.g. ['a', 'b', 'c'],
    # so we need to remove quotes from it.
    return remove_quotes(a), remove_quotes(b), remove_quotes(c)
