"""Quantized key/value cache storage.

Keys and values arrive as ``(batch, heads, tokens, head_dim)``. Each is quantized with an
asymmetric min/max grid per group, stored as packed codes plus an fp16 scale and an fp16
minimum per group::

    s = (max - min) / (2^b - 1),   q = clamp(round((x - min) / s), 0, 2^b - 1),   x' = q s + min

Two group shapes:

- **per token** (``axis="token"``): one group is ``group_size`` consecutive channels of one
  token. Each token is quantized the moment it is written, independently of the others.
- **per channel** (``axis="channel"``): one group is one channel over ``group_size``
  consecutive tokens. Keys have a few channels that are large for every token; a per-token
  grid must stretch over them and loses the small channels, while a per-channel grid
  gives each channel its own range. A group can only be quantized once all its tokens
  exist, so the last ``n mod group_size`` tokens wait in full precision (the residual).

Codes are packed along ``head_dim``: one byte for 8 bits, two codes per byte for 4 bits,
eight codes per three bytes for 3 bits, four per byte for 2 bits. ``bits=16`` stores the
tensor unchanged, which is useful to quantize only keys or only values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from bitsmith.packing import pack_int2, pack_int3, pack_int4, unpack_int2, unpack_int3, unpack_int4

Axis = Literal["token", "channel"]
BITS = (2, 3, 4, 8, 16)
_SPEC_RE = re.compile(r"^k(\d+)(c?)v(\d+)(c?)(?:-g(\d+))?$")


@dataclass(frozen=True)
class KVSpec:
    """How to quantize the cache: bits and group axis for keys and for values."""

    key_bits: int = 4
    value_bits: int = 4
    key_axis: Axis = "channel"
    value_axis: Axis = "token"
    group_size: int = 64

    def __post_init__(self) -> None:
        for bits in (self.key_bits, self.value_bits):
            if bits not in BITS:
                raise ValueError(f"cache bits must be one of {BITS}, got {bits}")
        for axis in (self.key_axis, self.value_axis):
            if axis not in ("token", "channel"):
                raise ValueError(f"axis must be 'token' or 'channel', got {axis!r}")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")

    @property
    def name(self) -> str:
        """Short name, the inverse of :func:`parse_kv_spec`: ``k4cv4`` = 4-bit keys per channel."""
        k = f"k{self.key_bits}{'c' if self.key_axis == 'channel' else ''}"
        v = f"v{self.value_bits}{'c' if self.value_axis == 'channel' else ''}"
        return k + v + ("" if self.group_size == 64 else f"-g{self.group_size}")

    def bits_per_element(self, which: Literal["key", "value"]) -> float:
        """Stored bits per cached number, counting the fp16 scale and minimum of each group."""
        bits = self.key_bits if which == "key" else self.value_bits
        return 16.0 if bits == 16 else bits + 32 / self.group_size

    @property
    def chunk(self) -> int:
        """Token chunk that keeps per-channel groups aligned (1 if nothing is per channel)."""
        pairs = ((self.key_bits, self.key_axis), (self.value_bits, self.value_axis))
        return self.group_size if any(b != 16 and a == "channel" for b, a in pairs) else 1


def parse_kv_spec(name: str) -> KVSpec:
    """Parse ``k{bits}[c]v{bits}[c][-g{group}]``; ``c`` means per channel, otherwise per token.

    ``k8v8``: INT8 keys and values per token. ``k4cv4``: 4-bit keys per channel and values per
    token (the KIVI layout). ``k2cv2-g32``: the same at 2 bits with groups of 32.
    """
    m = _SPEC_RE.match(name)
    if m is None:
        raise ValueError(f"bad cache spec {name!r}: expected k<bits>[c]v<bits>[c][-g<group>], e.g. k4cv4")
    kb, kc, vb, vc, g = m.groups()
    return KVSpec(
        key_bits=int(kb),
        value_bits=int(vb),
        key_axis="channel" if kc else "token",
        value_axis="channel" if vc else "token",
        group_size=int(g) if g else 64,
    )


# codes <-> bytes ----------------------------------------------------------------------
def _pack(codes: Tensor, bits: int) -> Tensor:
    if bits == 8:
        return codes
    # codes are clamped into range, so skip the packers' range check (a GPU to host sync)
    if bits == 4:
        return pack_int4(codes, validate=False)
    return pack_int3(codes, validate=False) if bits == 3 else pack_int2(codes, validate=False)


def _unpack(packed: Tensor, bits: int) -> Tensor:
    if bits == 8:
        return packed
    if bits == 4:
        return unpack_int4(packed, signed=False)
    return unpack_int3(packed) if bits == 3 else unpack_int2(packed)


@dataclass
class QBlock:
    """Quantized tokens ``(B, H, T, D)``: packed codes, fp16 scale and fp16 minimum per group."""

    packed: Tensor
    scale: Tensor
    low: Tensor
    bits: int
    axis: Axis
    group_size: int

    @property
    def tokens(self) -> int:
        return self.packed.shape[2] * (self.group_size if self.axis == "channel" else 1)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scale, self.low))


def quantize(x: Tensor, bits: int, axis: Axis, group_size: int) -> QBlock:
    """Quantize ``x`` of shape ``(B, H, T, D)``.

    Per token, ``D`` must be a multiple of ``group_size``; per channel, ``T`` must be.
    """
    b, h, t, d = x.shape
    if axis == "token":
        if d % group_size:
            raise ValueError(f"head_dim {d} is not a multiple of the group size {group_size}")
        groups = x.reshape(b, h, t, d // group_size, group_size)
        dim = -1
    else:
        if t % group_size:
            raise ValueError(f"{t} tokens is not a whole number of groups of {group_size}")
        groups = x.reshape(b, h, t // group_size, group_size, d)
        dim = -2
    xf = groups.float()
    low = xf.amin(dim, keepdim=True).to(torch.float16)
    high = xf.amax(dim, keepdim=True)
    qmax = (1 << bits) - 1
    scale = ((high - low.float()) / qmax).to(torch.float16)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))  # constant group: every code is 0
    codes = torch.round((xf - low.float()) / scale.float()).clamp_(0, qmax).to(torch.uint8)
    if axis == "token":
        codes = codes.reshape(b, h, t, d)
    return QBlock(_pack(codes, bits), scale, low, bits, axis, group_size)


def dequantize(q: QBlock, dtype: torch.dtype) -> Tensor:
    """Inverse of :func:`quantize`, up to rounding. Returns ``(B, H, T, D)`` in ``dtype``."""
    codes = _unpack(q.packed, q.bits).float()
    if q.axis == "token":
        b, h, t, d = codes.shape
        codes = codes.reshape(b, h, t, d // q.group_size, q.group_size)
    out = codes * q.scale.float() + q.low.float()
    b, h = out.shape[:2]
    d = out.shape[-1] if q.axis == "channel" else out.shape[-2] * out.shape[-1]
    return out.reshape(b, h, -1, d).to(dtype)


def _concat(a: QBlock, b: QBlock) -> QBlock:
    """Append the tokens of ``b`` to ``a`` (same layout), so the cache stays one block."""
    return QBlock(
        torch.cat([a.packed, b.packed], 2),
        torch.cat([a.scale, b.scale], 2),
        torch.cat([a.low, b.low], 2),
        a.bits,
        a.axis,
        a.group_size,
    )


@dataclass
class KVStream:
    """The stored keys (or values) of one attention layer.

    :meth:`append` adds new tokens and returns every stored token as the attention will
    read it: quantized tokens dequantized, followed by the full-precision residual. For
    the per-channel axis the new tokens are returned *before* their group is quantized,
    as in token-by-token decoding, where a group is quantized only once it is complete.
    """

    bits: int
    axis: Axis
    group_size: int
    block: QBlock | None = None
    residual: Tensor | None = None
    length: int = 0
    _dtype: torch.dtype = field(default=torch.float32, repr=False)

    def append(self, x: Tensor) -> Tensor:
        self._dtype = x.dtype
        self.length += x.shape[2]
        if self.bits == 16:
            self.residual = x if self.residual is None else torch.cat([self.residual, x], 2)
            return self.residual
        if self.axis == "token":
            self._store(quantize(x, self.bits, "token", self.group_size))
            return self.read()
        pending = x if self.residual is None else torch.cat([self.residual, x], 2)
        self.residual = pending
        out = self.read()
        whole = pending.shape[2] // self.group_size * self.group_size
        if whole:
            self._store(quantize(pending[:, :, :whole], self.bits, "channel", self.group_size))
            self.residual = pending[:, :, whole:] if whole < pending.shape[2] else None
        return out

    def _store(self, q: QBlock) -> None:
        self.block = q if self.block is None else _concat(self.block, q)

    def read(self) -> Tensor:
        parts = []
        if self.block is not None:
            parts.append(dequantize(self.block, self._dtype))
        if self.residual is not None:
            parts.append(self.residual)
        return parts[0] if len(parts) == 1 else torch.cat(parts, 2)

    @property
    def nbytes(self) -> int:
        """Bytes actually held: packed codes, scales, minimums and the residual."""
        res = 0 if self.residual is None else self.residual.numel() * self.residual.element_size()
        return res + (0 if self.block is None else self.block.nbytes)
