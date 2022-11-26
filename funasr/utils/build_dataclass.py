import argparse
import dataclasses

from typeguard import check_type


def build_dataclass(dataclass, args: argparse.Namespace):
    """Helper function to build dataclass from 'args'."""
    kwargs = {}
    for field in dataclasses.fields(dataclass):
        if not hasattr(args, field.name):
            raise ValueError(
                f"args doesn't have {field.name}. You need to set it to ArgumentsParser"
            )
        check_type(field.name, getattr(args, field.name), field.type)
        kwargs[field.name] = getattr(args, field.name)
    return dataclass(**kwargs)
