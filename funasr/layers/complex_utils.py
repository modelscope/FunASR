"""Beamformer module."""
from distutils.version import LooseVersion
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor


EPS = torch.finfo(torch.double).eps
is_torch_1_8_plus = LooseVersion(torch.__version__) >= LooseVersion("1.8.0")
is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


def new_complex_like(
    ref: Union[torch.Tensor, ComplexTensor],
    real_imag: Tuple[torch.Tensor, torch.Tensor],
):
    if isinstance(ref, ComplexTensor):
        return ComplexTensor(*real_imag)
    elif is_torch_complex_tensor(ref):
        return torch.complex(*real_imag)
    else:
        raise ValueError(
            "Please update your PyTorch version to 1.9+ for complex support."
        )


def is_torch_complex_tensor(c):
    return (
        not isinstance(c, ComplexTensor) and is_torch_1_9_plus and torch.is_complex(c)
    )


def is_complex(c):
    return isinstance(c, ComplexTensor) or is_torch_complex_tensor(c)


def to_double(c):
    if not isinstance(c, ComplexTensor) and is_torch_1_9_plus and torch.is_complex(c):
        return c.to(dtype=torch.complex128)
    else:
        return c.double()


def to_float(c):
    if not isinstance(c, ComplexTensor) and is_torch_1_9_plus and torch.is_complex(c):
        return c.to(dtype=torch.complex64)
    else:
        return c.float()


def cat(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            "cat(): argument 'tensors' (position 1) must be tuple of Tensors, "
            "not Tensor"
        )
    if isinstance(seq[0], ComplexTensor):
        return FC.cat(seq, *args, **kwargs)
    else:
        return torch.cat(seq, *args, **kwargs)


def complex_norm(
    c: Union[torch.Tensor, ComplexTensor], dim=-1, keepdim=False
) -> torch.Tensor:
    if not is_complex(c):
        raise TypeError("Input is not a complex tensor.")
    if is_torch_complex_tensor(c):
        return torch.norm(c, dim=dim, keepdim=keepdim)
    else:
        return torch.sqrt(
            (c.real**2 + c.imag**2).sum(dim=dim, keepdim=keepdim) + EPS
        )


def einsum(equation, *operands):
    # NOTE: Do not mix ComplexTensor and torch.complex in the input!
    # NOTE (wangyou): Until PyTorch 1.9.0, torch.einsum does not support
    # mixed input with complex and real tensors.
    if len(operands) == 1:
        if isinstance(operands[0], (tuple, list)):
            operands = operands[0]
        complex_module = FC if isinstance(operands[0], ComplexTensor) else torch
        return complex_module.einsum(equation, *operands)
    elif len(operands) != 2:
        op0 = operands[0]
        same_type = all(op.dtype == op0.dtype for op in operands[1:])
        if same_type:
            _einsum = FC.einsum if isinstance(op0, ComplexTensor) else torch.einsum
            return _einsum(equation, *operands)
        else:
            raise ValueError("0 or More than 2 operands are not supported.")
    a, b = operands
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        return FC.einsum(equation, a, b)
    elif is_torch_1_9_plus and (torch.is_complex(a) or torch.is_complex(b)):
        if not torch.is_complex(a):
            o_real = torch.einsum(equation, a, b.real)
            o_imag = torch.einsum(equation, a, b.imag)
            return torch.complex(o_real, o_imag)
        elif not torch.is_complex(b):
            o_real = torch.einsum(equation, a.real, b)
            o_imag = torch.einsum(equation, a.imag, b)
            return torch.complex(o_real, o_imag)
        else:
            return torch.einsum(equation, a, b)
    else:
        return torch.einsum(equation, a, b)


def inverse(
    c: Union[torch.Tensor, ComplexTensor]
) -> Union[torch.Tensor, ComplexTensor]:
    if isinstance(c, ComplexTensor):
        return c.inverse2()
    else:
        return c.inverse()


def matmul(
    a: Union[torch.Tensor, ComplexTensor], b: Union[torch.Tensor, ComplexTensor]
) -> Union[torch.Tensor, ComplexTensor]:
    # NOTE: Do not mix ComplexTensor and torch.complex in the input!
    # NOTE (wangyou): Until PyTorch 1.9.0, torch.matmul does not support
    # multiplication between complex and real tensors.
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        return FC.matmul(a, b)
    elif is_torch_1_9_plus and (torch.is_complex(a) or torch.is_complex(b)):
        if not torch.is_complex(a):
            o_real = torch.matmul(a, b.real)
            o_imag = torch.matmul(a, b.imag)
            return torch.complex(o_real, o_imag)
        elif not torch.is_complex(b):
            o_real = torch.matmul(a.real, b)
            o_imag = torch.matmul(a.imag, b)
            return torch.complex(o_real, o_imag)
        else:
            return torch.matmul(a, b)
    else:
        return torch.matmul(a, b)


def trace(a: Union[torch.Tensor, ComplexTensor]):
    # NOTE (wangyou): until PyTorch 1.9.0, torch.trace does not
    # support bacth processing. Use FC.trace() as fallback.
    return FC.trace(a)


def reverse(a: Union[torch.Tensor, ComplexTensor], dim=0):
    if isinstance(a, ComplexTensor):
        return FC.reverse(a, dim=dim)
    else:
        return torch.flip(a, dims=(dim,))


def solve(b: Union[torch.Tensor, ComplexTensor], a: Union[torch.Tensor, ComplexTensor]):
    """Solve the linear equation ax = b."""
    # NOTE: Do not mix ComplexTensor and torch.complex in the input!
    # NOTE (wangyou): Until PyTorch 1.9.0, torch.solve does not support
    # mixed input with complex and real tensors.
    if isinstance(a, ComplexTensor) or isinstance(b, ComplexTensor):
        if isinstance(a, ComplexTensor) and isinstance(b, ComplexTensor):
            return FC.solve(b, a, return_LU=False)
        else:
            return matmul(inverse(a), b)
    elif is_torch_1_9_plus and (torch.is_complex(a) or torch.is_complex(b)):
        if torch.is_complex(a) and torch.is_complex(b):
            return torch.linalg.solve(a, b)
        else:
            return matmul(inverse(a), b)
    else:
        if is_torch_1_8_plus:
            return torch.linalg.solve(a, b)
        else:
            return torch.solve(b, a)[0]


def stack(seq: Sequence[Union[ComplexTensor, torch.Tensor]], *args, **kwargs):
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            "stack(): argument 'tensors' (position 1) must be tuple of Tensors, "
            "not Tensor"
        )
    if isinstance(seq[0], ComplexTensor):
        return FC.stack(seq, *args, **kwargs)
    else:
        return torch.stack(seq, *args, **kwargs)
