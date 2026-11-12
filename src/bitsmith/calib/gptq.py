"""GPTQ: quantize one column at a time and push the error onto the remaining columns.

With ``H = 2 X^T X / n`` the layer-wise objective ``||X W^T - X Q^T||^2`` has the closed
form update from Optimal Brain Surgeon. After quantizing column ``i`` with error
``e = (w_i - q_i) / [H^-1]_ii`` every later column ``j`` is corrected by
``w_j -= e * [H^-1]_ij``. Using the upper Cholesky factor of ``H^-1`` makes each step a
row slice, and updates are applied lazily per block for speed.

Group parameters (scale, zero)
------------------------------
* Default: computed when the first column of a group is reached, on the already corrected
  weights. ``block_size`` is rounded to a multiple of ``group_size`` so a group never
  straddles two lazy-update blocks.
* ``act_order=True``: columns are processed in order of decreasing ``diag(H)`` (the inputs
  with the most energy first, while the most columns are left to absorb their error).
  Group parameters are then computed up front from the original weights ("static
  groups"), so the storage layout stays contiguous groups and needs no index map.

Grid formats (NF4, MX)
----------------------
:func:`gptq_quantize_table` runs the same column loop for codecs whose weights are a
per-block scale times a value from a fixed grid. Block scales are fixed up front from the
original weights (as with static groups), and each column is rounded with the codec's own
``snap``, so the result is exactly what the codec stores. See ADR 0012.

See ADR 0005 and ADR 0009.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from bitsmith.nn.codecs import TableCodec, round_scale
from bitsmith.quant import QTensor, QuantSpec, compute_qparams


@dataclass
class GPTQResult:
    qtensor: QTensor
    proxy_loss: float  # sum of (w - q)^2 / [H^-1]_ii / 2, the OBS objective


@dataclass
class TableGPTQResult:
    buffers: dict[str, Tensor]  # ready for QLinear, as produced by the codec
    weight: Tensor  # the dequantized result, float32
    proxy_loss: float


def _upper_inverse_factor(h: Tensor) -> Tensor | None:
    """``V`` with ``h^-1 = V^T V``, or None if ``h`` is not positive definite at this precision."""
    lower, info = torch.linalg.cholesky_ex(h.flip(0, 1))
    if int(info) != 0:
        return None
    eye = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    upper = torch.linalg.solve_triangular(lower, eye, upper=False).flip(0, 1)
    return upper if bool(torch.isfinite(upper).all()) else None


def damped_inverse_cholesky(
    hessian: Tensor, damp: float = 0.01, *, dtype: torch.dtype | None = None
) -> tuple[Tensor, Tensor]:
    """Return ``(upper Cholesky factor U of H^-1, dead-column mask)``, so ``H^-1 = U^T U``.

    With ``P`` the order-reversing permutation and ``P H P = L L^T``:
    ``H^-1 = P L^-T L^-1 P = V^T V`` where ``V = P L^-1 P`` is upper triangular with a
    positive diagonal, so ``V`` is exactly the Cholesky factor. That needs one
    factorization and one triangular inverse instead of factorize, invert, factorize.

    ``dtype`` defaults to float32 on CUDA (consumer GPUs run float64 at 1/64 speed; the
    GPTQ output error is unchanged, see ADR 0013) and float64 elsewhere. A float32
    factorization that fails or overflows is redone in float64.
    """
    if dtype is None:
        dtype = torch.float32 if hessian.is_cuda else torch.float64
    h = hessian.to(torch.float64).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    h += damp * torch.diag(h).mean() * torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
    upper = _upper_inverse_factor(h.to(dtype))
    if upper is None and dtype != torch.float64:
        upper = _upper_inverse_factor(h)
    if upper is None:
        raise torch.linalg.LinAlgError("damped Hessian is not positive definite; increase damp")
    return upper.to(torch.float32), dead


def _group_params(w: Tensor, group_size: int, spec: QuantSpec) -> tuple[Tensor, Tensor]:
    """Static ``(scale, zero)`` of shape ``(rows, n_groups)`` from the given weights."""
    rows, cols = w.shape
    n_groups = -(-cols // group_size)
    padded = torch.nn.functional.pad(w, (0, n_groups * group_size - cols))
    s, z = compute_qparams(padded.reshape(rows, n_groups, group_size), spec)
    return round_scale(s[..., 0]), z[..., 0]


# round(x, *params) -> quantized values of one column; params are that column's rows of
# the per-column parameter tables (for example scale and zero point)
RoundFn = Callable[..., Tensor]


def _inner_block(wb: Tensor, hb: Tensor, params: Sequence[Tensor], qb: Tensor, eb: Tensor, rnd: RoundFn) -> None:
    """Quantize the columns of one inner block in order, with rank-1 updates inside it.

    ``wb`` is ``(n, rows)`` and is updated in place; results go to ``qb`` and ``eb``.
    Only tensor operations, no host synchronization, so the block can be a CUDA graph.
    """
    n = wb.shape[0]
    for k in range(n):
        x = wb[k]
        qb[k] = rnd(x, *(p[k] for p in params))
        torch.sub(x, qb[k], out=eb[k])
        eb[k].div_(hb[k, k])
        if k + 1 < n:
            wb[k + 1 :].addr_(hb[k, k + 1 :], eb[k], alpha=-1.0)


class _BlockRunner:
    """Runs :func:`_inner_block` eagerly, or on CUDA as a captured graph that is replayed.

    A 32-column block issues about 300 small kernels; replaying a graph issues one. The
    graph runs on fixed buffers, so each block is copied in and its results copied out.
    Partial blocks (the last one of a layer) run eagerly. Both paths execute the same
    kernels in the same order, so their results are identical.
    """

    def __init__(self, rnd: RoundFn, use_graph: bool) -> None:
        self.rnd = rnd
        self.use_graph = use_graph and torch.cuda.is_available()
        self.graph: torch.cuda.CUDAGraph | None = None
        self.size = 0
        self.bufs: tuple[Tensor, Tensor, list[Tensor], Tensor, Tensor] | None = None

    def __call__(self, wb: Tensor, hb: Tensor, params: Sequence[Tensor], qb: Tensor, eb: Tensor) -> None:
        if not (self.use_graph and wb.is_cuda and (self.graph is None or wb.shape[0] == self.size)):
            _inner_block(wb, hb, params, qb, eb, self.rnd)
            return
        if self.graph is None:
            self._capture(wb, hb, params)
        assert self.bufs is not None and self.graph is not None
        swb, shb, sparams, sqb, seb = self.bufs
        swb.copy_(wb)
        shb.copy_(hb)
        for s, p in zip(sparams, params, strict=True):
            s.copy_(p)
        self.graph.replay()
        qb.copy_(sqb)
        eb.copy_(seb)

    def _capture(self, wb: Tensor, hb: Tensor, params: Sequence[Tensor]) -> None:
        swb, shb = wb.clone(), hb.clone()
        sparams = [p.clone() for p in params]
        sqb, seb = torch.empty_like(wb), torch.empty_like(wb)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm up outside the graph, as CUDA graphs require
            _inner_block(swb.clone(), shb, sparams, sqb, seb, self.rnd)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _inner_block(swb, shb, sparams, sqb, seb, self.rnd)
        self.graph, self.size, self.bufs = graph, wb.shape[0], (swb, shb, sparams, sqb, seb)


def _obs_loop(
    wt: Tensor,
    hinv: Tensor,
    rnd: RoundFn,
    params: Sequence[Tensor],
    *,
    block_size: int,
    inner_size: int,
    on_inner: Callable[[int, int], None] | None = None,
    use_cuda_graph: bool = True,
) -> tuple[Tensor, Tensor]:
    """The GPTQ column loop on a transposed weight ``wt`` of shape ``(cols, rows)``.

    Row ``j`` of ``wt`` is input column ``j`` (already in processing order), so every
    column is a contiguous slice; ``params`` are ``(cols, rows)`` tables in the same order.
    Updates are lazy on two levels: rank-1 updates only touch the rest of the current
    inner block, one matmul updates the rest of the outer block when an inner block ends,
    and one more updates all later columns when an outer block ends. This is the same
    arithmetic as updating every later column after every step, reordered.

    ``on_inner(i1, i2)`` runs when inner block ``[i1, i2)`` starts; at that point those
    columns have received the updates of every earlier column, and it may fill ``params``
    for them. ``wt`` is modified in place. Returns the quantized columns ``(cols, rows)``
    and the OBS loss per output row.
    """
    cols = wt.shape[0]
    out = torch.empty_like(wt)
    errs = torch.empty_like(wt)  # scaled errors e_j = (w_j - q_j) / U_jj
    runner = _BlockRunner(rnd, use_cuda_graph)
    for b1 in range(0, cols, block_size):
        b2 = min(b1 + block_size, cols)
        for i1 in range(b1, b2, inner_size):
            i2 = min(i1 + inner_size, b2)
            if on_inner is not None:
                on_inner(i1, i2)
            runner(wt[i1:i2], hinv[i1:i2, i1:i2], [p[i1:i2] for p in params], out[i1:i2], errs[i1:i2])
            if i2 < b2:
                wt[i2:b2].addmm_(hinv[i1:i2, i2:b2].T, errs[i1:i2], alpha=-1.0)
        if b2 < cols:
            wt[b2:].addmm_(hinv[b1:b2, b2:].T, errs[b1:b2], alpha=-1.0)
    loss = errs.pow(2).sum(dim=0) / 2
    return out, loss


def _prepare(weight: Tensor, hessian: Tensor, act_order: bool, damp: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Float32 weight with dead inputs zeroed, column order, permuted transposed weight, U."""
    w = weight.detach().to(torch.float32).clone()
    h = hessian.to(device=w.device, dtype=torch.float32)
    w[:, torch.diag(h) == 0] = 0.0
    cols = w.shape[1]
    perm = torch.argsort(torch.diag(h), descending=True) if act_order else torch.arange(cols, device=w.device)
    hinv, _ = damped_inverse_cholesky(h[perm][:, perm], damp)
    return w, perm, w[:, perm].T.contiguous(), hinv


def _unpermute(x: Tensor, perm: Tensor) -> Tensor:
    """Columns of ``x`` (rows, cols) back from processing order to the original order."""
    unperm = torch.empty_like(perm)
    unperm[perm] = torch.arange(perm.numel(), device=perm.device)
    return x[:, unperm]


@torch.no_grad()
def gptq_quantize(
    weight: Tensor,
    hessian: Tensor,
    *,
    bits: int = 4,
    group_size: int = 128,
    symmetric: bool = False,
    block_size: int = 128,
    damp: float = 0.01,
    act_order: bool = False,
    inner_size: int = 32,
    use_cuda_graph: bool = True,
) -> GPTQResult:
    spec = QuantSpec(bits=bits, symmetric=symmetric, granularity="group", group_size=group_size)
    w, perm, wt, hinv = _prepare(weight, hessian, act_order, damp)
    rows, cols = w.shape
    n_groups = -(-cols // group_size)
    qmin, qmax = float(spec.qmin), float(spec.qmax)

    def rnd(x: Tensor, s: Tensor, z: Tensor) -> Tensor:
        return ((torch.round(x / s) + z).clamp_(qmin, qmax) - z) * s

    if act_order:
        # static groups: parameters from the original weights, gathered per processed column
        scales, zeros = _group_params(w, group_size, spec)
        group_of = perm // group_size
        col_s = scales.T[group_of].contiguous()  # (cols, rows) in processing order
        col_z = zeros.T[group_of].contiguous()
        on_inner = None
        inner = inner_size
    else:
        # dynamic groups: parameters from the corrected weights when a group starts. An inner
        # block is exactly one group, so its columns are up to date at that moment.
        scales = torch.zeros(rows, n_groups, device=w.device)
        zeros = torch.zeros(rows, n_groups, device=w.device)
        col_s = torch.empty_like(wt)
        col_z = torch.empty_like(wt)
        inner = group_size

        def on_inner(i1: int, i2: int) -> None:
            s, z = compute_qparams(wt[i1:i2].T.unsqueeze(1), spec)
            g = i1 // group_size
            scales[:, g], zeros[:, g] = round_scale(s[:, 0, 0]), z[:, 0, 0]
            col_s[i1:i2] = scales[:, g]
            col_z[i1:i2] = zeros[:, g]

    block = max(group_size, (block_size // group_size) * group_size)
    out_t, loss = _obs_loop(
        wt,
        hinv,
        rnd,
        (col_s, col_z),
        block_size=block,
        inner_size=min(inner, block),
        on_inner=on_inner,
        use_cuda_graph=use_cuda_graph,
    )
    # out = (q - z) * s exactly, so the integer codes are recovered by one more rounding
    codes = _unpermute((torch.round(out_t / col_s) + col_z).T, perm)

    # Padded columns (beyond ``cols``) get the zero point so they dequantize to exactly 0.
    pad = n_groups * group_size - cols
    codes = torch.nn.functional.pad(codes, (0, pad))
    if pad:
        codes[:, cols:] = zeros[:, -1:].expand(rows, pad)
    dtype = torch.int8 if symmetric else torch.uint8
    qt = QTensor(
        codes=codes.reshape(rows, n_groups, group_size).to(dtype),
        scale=scales.unsqueeze(-1),
        zero=zeros.unsqueeze(-1),
        spec=spec,
        shape=weight.shape,
    )
    return GPTQResult(qtensor=qt, proxy_loss=float(loss.sum()))


@torch.no_grad()
def gptq_quantize_table(
    weight: Tensor,
    hessian: Tensor,
    codec: TableCodec,
    *,
    block_size: int = 128,
    damp: float = 0.01,
    act_order: bool = True,
    inner_size: int = 32,
    use_cuda_graph: bool = True,
) -> TableGPTQResult:
    """GPTQ onto the grid of ``codec`` (NF4, MXFP4, MXINT8) with scales fixed from ``weight``."""
    w, perm, wt, hinv = _prepare(weight, hessian, act_order, damp)
    cols = w.shape[1]
    scale = codec.scale_buffer(w)
    col_s = codec.scale_values(scale).repeat_interleave(codec.block_size, dim=1)[:, :cols]
    col_s = col_s[:, perm].T.contiguous()  # (cols, rows) in processing order

    def rnd(x: Tensor, s: Tensor) -> Tensor:
        return codec.snap(x / s) * s

    out_t, loss = _obs_loop(
        wt, hinv, rnd, (col_s,), block_size=block_size, inner_size=inner_size, use_cuda_graph=use_cuda_graph
    )
    out = _unpermute(out_t.T, perm).contiguous()
    return TableGPTQResult(codec.encode_with_scale(out, scale), out, float(loss.sum()))


def output_error(weight: Tensor, approx: Tensor, hessian: Tensor) -> float:
    """``tr((W - Q) H (W - Q)^T)``: expected squared output error under the calibration data."""
    d = (weight.to(torch.float32) - approx.to(torch.float32)).to(torch.float64)
    h = hessian.to(device=d.device, dtype=torch.float64)
    return float(torch.einsum("ij,jk,ik->", d, h, d))
