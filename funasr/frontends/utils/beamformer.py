import torch
from torch_complex import functional as FC
from torch_complex.tensor import ComplexTensor


def get_power_spectral_density_matrix(
    xs: ComplexTensor, mask: torch.Tensor, normalization=True, eps: float = 1e-15
) -> ComplexTensor:
    """Return cross-channel power spectral density (PSD) matrix

    Args:
        xs (ComplexTensor): (..., F, C, T)
        mask (torch.Tensor): (..., F, C, T)
        normalization (bool):
        eps (float):
    Returns
        psd (ComplexTensor): (..., F, C, C)

    """
    # outer product: (..., C_1, T) x (..., C_2, T) -> (..., T, C, C_2)
    psd_Y = FC.einsum("...ct,...et->...tce", [xs, xs.conj()])

    # Averaging mask along C: (..., C, T) -> (..., T)
    mask = mask.mean(dim=-2)

    # Normalized mask along T: (..., T)
    if normalization:
        # If assuming the tensor is padded with zero, the summation along
        # the time axis is same regardless of the padding length.
        mask = mask / (mask.sum(dim=-1, keepdim=True) + eps)

    # psd: (..., T, C, C)
    psd = psd_Y * mask[..., None, None]
    # (..., T, C, C) -> (..., C, C)
    psd = psd.sum(dim=-3)

    return psd


def get_mvdr_vector(
    psd_s: ComplexTensor,
    psd_n: ComplexTensor,
    reference_vector: torch.Tensor,
    eps: float = 1e-15,
) -> ComplexTensor:
    """Return the MVDR(Minimum Variance Distortionless Response) vector:

        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u

    Reference:
        On optimal frequency-domain multichannel linear filtering
        for noise reduction; M. Souden et al., 2010;
        https://ieeexplore.ieee.org/document/5089420

    Args:
        psd_s (ComplexTensor): (..., F, C, C)
        psd_n (ComplexTensor): (..., F, C, C)
        reference_vector (torch.Tensor): (..., C)
        eps (float):
    Returns:
        beamform_vector (ComplexTensor)r: (..., F, C)
    """
    # Add eps
    C = psd_n.size(-1)
    eye = torch.eye(C, dtype=psd_n.dtype, device=psd_n.device)
    shape = [1 for _ in range(psd_n.dim() - 2)] + [C, C]
    eye = eye.view(*shape)
    psd_n += eps * eye

    # numerator: (..., C_1, C_2) x (..., C_2, C_3) -> (..., C_1, C_3)
    numerator = FC.einsum("...ec,...cd->...ed", [psd_n.inverse(), psd_s])
    # ws: (..., C, C) / (...,) -> (..., C, C)
    ws = numerator / (FC.trace(numerator)[..., None, None] + eps)
    # h: (..., F, C_1, C_2) x (..., C_2) -> (..., F, C_1)
    beamform_vector = FC.einsum("...fec,...c->...fe", [ws, reference_vector])
    return beamform_vector


def apply_beamforming_vector(beamform_vector: ComplexTensor, mix: ComplexTensor) -> ComplexTensor:
    # (..., C) x (..., C, T) -> (..., T)
    es = FC.einsum("...c,...ct->...t", [beamform_vector.conj(), mix])
    return es
