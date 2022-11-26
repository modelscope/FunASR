import inspect


class Invalid:
    """Marker object for not serializable-object"""


def get_default_kwargs(func):
    """Get the default values of the input function.

    Examples:
        >>> def func(a, b=3):  pass
        >>> get_default_kwargs(func)
        {'b': 3}

    """

    def yaml_serializable(value):
        # isinstance(x, tuple) includes namedtuple, so type is used here
        if type(value) is tuple:
            return yaml_serializable(list(value))
        elif isinstance(value, set):
            return yaml_serializable(list(value))
        elif isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                return Invalid
            retval = {}
            for k, v in value.items():
                v2 = yaml_serializable(v)
                # Register only valid object
                if v2 not in (Invalid, inspect.Parameter.empty):
                    retval[k] = v2
            return retval
        elif isinstance(value, list):
            retval = []
            for v in value:
                v2 = yaml_serializable(v)
                # If any elements in the list are invalid,
                # the list also becomes invalid
                if v2 is Invalid:
                    return Invalid
                else:
                    retval.append(v2)
            return retval
        elif value in (inspect.Parameter.empty, None):
            return value
        elif isinstance(value, (float, int, complex, bool, str, bytes)):
            return value
        else:
            return Invalid

    # params: An ordered mapping of inspect.Parameter
    params = inspect.signature(func).parameters
    data = {p.name: p.default for p in params.values()}
    # Remove not yaml-serializable object
    data = yaml_serializable(data)
    return data
