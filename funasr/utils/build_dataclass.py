import argparse
import dataclasses



def build_dataclass(dataclass, args: argparse.Namespace):
    """Helper function to build dataclass from 'args'."""
    kwargs = {}
    for field in dataclasses.fields(dataclass):
        if not hasattr(args, field.name):
            raise ValueError(
                f"args doesn't have {field.name}. You need to set it to ArgumentsParser"
            )
        kwargs[field.name] = getattr(args, field.name)
    return dataclass(**kwargs)
