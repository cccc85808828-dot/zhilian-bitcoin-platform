from __future__ import annotations

import torch


def scatter_add(
    source: torch.Tensor,
    index: torch.Tensor,
    *,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Native PyTorch fallback for the dim-0 torch_scatter operation used here."""
    if dim < 0:
        dim += source.ndim
    if dim != 0:
        source = source.movedim(dim, 0)
    size = (
        int(dim_size)
        if dim_size is not None
        else (int(index.max()) + 1 if index.numel() else 0)
    )
    output = source.new_zeros((size, *source.shape[1:]))
    if index.numel():
        output.index_add_(0, index.to(dtype=torch.long), source)
    if dim != 0:
        output = output.movedim(0, dim)
    return output


def scatter_mean(
    source: torch.Tensor,
    index: torch.Tensor,
    *,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    if dim < 0:
        dim += source.ndim
    if dim != 0:
        source = source.movedim(dim, 0)
    size = (
        int(dim_size)
        if dim_size is not None
        else (int(index.max()) + 1 if index.numel() else 0)
    )
    output = scatter_add(source, index, dim=0, dim_size=size)
    counts = source.new_zeros(size)
    if index.numel():
        counts.index_add_(
            0,
            index.to(dtype=torch.long),
            torch.ones(index.numel(), dtype=source.dtype, device=source.device),
        )
    divisor = counts.clamp_min(1).reshape(size, *([1] * (source.ndim - 1)))
    output = output / divisor
    if dim != 0:
        output = output.movedim(0, dim)
    return output
