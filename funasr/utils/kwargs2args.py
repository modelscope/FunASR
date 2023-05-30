import inspect


def func(a: int, b, *, c, **kwargs):
    pass

def kwargs2args(func, kwargs):
    parameters = inspect.signature(func).parameters
    d = {k: i for i, k in enumerate(parameters)}
    args = [None for i in range(len(parameters))]
    for k, v in kwargs.items():
        if k in d:
            args[d[k]] = v

    for i, v in enumerate(args):
        if v is None:
            break

    return tuple(args[:i])
