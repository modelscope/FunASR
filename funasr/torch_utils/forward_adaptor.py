import torch
from typeguard import check_argument_types


class ForwardAdaptor(torch.nn.Module):
    """Wrapped module to parallelize specified method

    torch.nn.DataParallel parallelizes only "forward()"
    and, maybe, the method having the other name can't be applied
    except for wrapping the module just like this class.

    Examples:
        >>> class A(torch.nn.Module):
        ...     def foo(self, x):
        ...         ...
        >>> model = A()
        >>> model = ForwardAdaptor(model, "foo")
        >>> model = torch.nn.DataParallel(model, device_ids=[0, 1])
        >>> x = torch.randn(2, 10)
        >>> model(x)
    """

    def __init__(self, module: torch.nn.Module, name: str):
        assert check_argument_types()
        super().__init__()
        self.module = module
        self.name = name
        if not hasattr(module, name):
            raise ValueError(f"{module} doesn't have {name}")

    def forward(self, *args, **kwargs):
        func = getattr(self.module, self.name)
        return func(*args, **kwargs)
