"""Pack unsigned 3-bit and 2-bit codes.

INT3: eight codes ``v0..v7`` in three bytes. The codes form one little-endian 24-bit
word, three bits each, and the word is stored low byte first::

    word   = v0 | v1 << 3 | v2 << 6 | ... | v7 << 21
    byte_k = (word >> 8k) & 0xFF                       k = 0, 1, 2

So a code can straddle two bytes (``v2`` uses bits 6 to 8). Unpacking rebuilds the word
and shifts each code back out. 3 bits per weight, no padding inside a group of eight.

INT2: four codes per byte, element ``4k + i`` at bit offset ``2i``, the same layout as
the 2-bit ternary format.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _check(codes: Tensor, bits: int, multiple: int, validate: bool) -> Tensor:
    if codes.shape[-1] % multiple:
        raise ValueError(f"last dimension must be a multiple of {multiple}, got {codes.shape[-1]}")
    c = codes.to(torch.int32)
    if validate and c.numel() and (int(c.min()) < 0 or int(c.max()) >= 1 << bits):
        raise ValueError(f"int{bits} codes must lie in [0, {(1 << bits) - 1}]")
    return c


def pack_int3(codes: Tensor, *, validate: bool = True) -> Tensor:
    """Pack codes in ``[0, 7]`` (last dim a multiple of 8) into ``3/8`` as many bytes.

    ``validate=False`` skips the range check (a host read), as in :func:`pack_int4`.
    """
    c = _check(codes, 3, 8, validate)
    groups = c.reshape(*c.shape[:-1], -1, 8)
    shifts = torch.arange(0, 24, 3, device=c.device, dtype=torch.int32)
    word = (groups << shifts).sum(dim=-1)  # fields do not overlap, so the sum is an OR
    bytes3 = torch.stack([(word >> (8 * k)) & 0xFF for k in range(3)], dim=-1)
    return bytes3.reshape(*c.shape[:-1], -1).to(torch.uint8)


def unpack_int3(packed: Tensor) -> Tensor:
    """Inverse of :func:`pack_int3`. Returns uint8 codes."""
    if packed.shape[-1] % 3:
        raise ValueError(f"last dimension must be a multiple of 3, got {packed.shape[-1]}")
    b = packed.to(torch.int32).reshape(*packed.shape[:-1], -1, 3)
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    shifts = torch.arange(0, 24, 3, device=packed.device, dtype=torch.int32)
    codes = (word.unsqueeze(-1) >> shifts) & 0b111
    return codes.reshape(*packed.shape[:-1], -1).to(torch.uint8)


def pack_int2(codes: Tensor, *, validate: bool = True) -> Tensor:
    """Pack codes in ``[0, 3]`` (last dim a multiple of 4), four per byte."""
    c = _check(codes, 2, 4, validate)
    groups = c.reshape(*c.shape[:-1], -1, 4)
    shifts = torch.arange(0, 8, 2, device=c.device, dtype=torch.int32)
    return (groups << shifts).sum(dim=-1).to(torch.uint8)


def unpack_int2(packed: Tensor) -> Tensor:
    """Inverse of :func:`pack_int2`. Returns uint8 codes."""
    shifts = torch.arange(0, 8, 2, device=packed.device, dtype=torch.int32)
    codes = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0b11
    return codes.reshape(*packed.shape[:-1], -1).to(torch.uint8)
