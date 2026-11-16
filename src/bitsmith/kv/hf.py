"""Plug the quantized KV store into Hugging Face models and measure its cost.

``quantized_cache(spec)`` returns a ``transformers`` cache object; pass it as
``past_key_values`` to a forward pass or to ``generate``. Requires the ``hf`` extra
(tested with transformers 4.57 and 5.2).

Perplexity (:func:`kv_perplexity`) runs each evaluation window through a fresh cache in
chunks of ``spec.chunk`` tokens. Every query then attends to keys and values exactly as
token-by-token decoding would store them: per-token groups quantized on write, per-channel
groups quantized once complete, so the chunked pass gives the same numbers as decoding the
window one token at a time, without the cost.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from bitsmith.bench.pareto import SweepConfig
from bitsmith.eval.perplexity import PerplexityResult, sliding_window_perplexity
from bitsmith.kv.quant import KVSpec, KVStream


@functools.cache
def _classes() -> tuple[type, type]:
    from transformers.cache_utils import Cache, DynamicLayer

    class QuantizedKVLayer(DynamicLayer):  # type: ignore[misc, valid-type]
        """One attention layer's cache: a key stream and a value stream."""

        def __init__(self, spec: KVSpec) -> None:
            super().__init__()
            self.spec = spec
            self._new_streams()

        def _new_streams(self) -> None:
            s = self.spec
            self.key_stream = KVStream(s.key_bits, s.key_axis, s.group_size)
            self.value_stream = KVStream(s.value_bits, s.value_axis, s.group_size)

        def lazy_initialization(self, key_states: Tensor, value_states: Tensor | None = None) -> None:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.is_initialized = True

        def update(
            self, key_states: Tensor, value_states: Tensor, cache_kwargs: dict[str, Any] | None = None
        ) -> tuple[Tensor, Tensor]:
            if not self.is_initialized:
                self.lazy_initialization(key_states)
            return self.key_stream.append(key_states), self.value_stream.append(value_states)

        def get_seq_length(self) -> int:
            return self.key_stream.length

        def reset(self) -> None:
            self._new_streams()

        def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
            raise NotImplementedError("beam search is not supported by the quantized cache")

        def crop(self, max_length: int) -> None:
            raise NotImplementedError("cropping is not supported by the quantized cache")

        @property
        def nbytes(self) -> int:
            return self.key_stream.nbytes + self.value_stream.nbytes

    class QuantizedKVCache(Cache):  # type: ignore[misc, valid-type]
        """A cache whose layers store quantized keys and values (see :class:`KVSpec`)."""

        def __init__(self, spec: KVSpec) -> None:
            # the cache only calls this with no arguments, so a partial works where a class is typed
            super().__init__(layer_class_to_replicate=functools.partial(QuantizedKVLayer, spec))  # type: ignore[arg-type]
            self.spec = spec

        @property
        def nbytes(self) -> int:
            return sum(layer.nbytes for layer in self.layers)  # type: ignore[attr-defined]

    return QuantizedKVLayer, QuantizedKVCache


def quantized_cache(spec: KVSpec) -> Any:
    """A new, empty Hugging Face cache that stores keys and values as ``spec`` says."""
    return _classes()[1](spec)


def kv_bytes_per_token(config: Any, spec: KVSpec | None) -> float:
    """Cache bytes per token for a model config (all layers, keys and values), groups complete.

    ``spec=None`` means an unquantized cache in 16 bits.
    """
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    heads = getattr(text, "num_key_value_heads", None) or text.num_attention_heads
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    per_head = heads * head_dim * text.num_hidden_layers
    if spec is None:
        return per_head * 2 * 2.0
    return per_head * (spec.bits_per_element("key") + spec.bits_per_element("value")) / 8


def kv_logits_fn(model: nn.Module, spec: KVSpec | None) -> Any:
    """``ids -> logits`` running each window through a fresh cache, chunked for exactness."""

    def logits(ids: Tensor) -> Tensor:
        if spec is None:
            return model(ids).logits  # type: ignore[no-any-return]
        cache = quantized_cache(spec)
        step = spec.chunk if spec.chunk > 1 else ids.shape[1]
        outs = [
            model(ids[:, s : s + step], past_key_values=cache, use_cache=True).logits
            for s in range(0, ids.shape[1], step)
        ]
        return torch.cat(outs, 1)

    return logits


@torch.no_grad()
def kv_perplexity(
    model: nn.Module,
    tokens: Tensor,
    spec: KVSpec | None,
    *,
    window: int = 1024,
    stride: int = 512,
) -> PerplexityResult:
    """Sliding-window perplexity with the cache quantized as ``spec`` (``None`` = 16-bit cache)."""
    device = next(model.parameters()).device
    return sliding_window_perplexity(kv_logits_fn(model, spec), tokens, window=window, stride=stride, device=device)


_SHORT = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}


def model_bytes(model: nn.Module) -> int:
    """Bytes held by a model's parameters and buffers (packed weights included), tied tensors once."""
    seen: set[int] = set()
    total = 0
    for t in [*model.parameters(), *model.buffers()]:
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total


@dataclass
class KVPoint:
    label: str
    key_bits: float  # stored bits per cached key element, scales included
    value_bits: float
    kib_per_token: float  # all layers, keys and values
    compression: float  # cache size against a 16-bit cache
    perplexity: float
    delta_pct: float  # against the unquantized model with a 16-bit cache
    seconds: float
    weights: str = ""  # weight config, or the dtype when the weights are not quantized
    model_mib: float = 0.0  # parameters and buffers
    total_mib: float = 0.0  # model plus the cache at ``context`` tokens

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass
class KVSweep:
    model_id: str
    n_params: int
    eval_tokens: int
    points: list[KVPoint]
    context: int = 32768


def run_kv_sweep(
    model_id: str,
    specs: Sequence[KVSpec],
    *,
    weights: SweepConfig | None = None,
    eval_tokens: int | None = 8192,
    window: int = 1024,
    stride: int = 512,
    n_calib: int = 128,
    calib_len: int = 512,
    context: int = 32768,
    device: str = "cpu",
    dtype: str = "float32",
    progress: Callable[[str], None] | None = None,
) -> KVSweep:
    """WikiText-2 perplexity of a Hugging Face model with each cache spec, plus a 16-bit cache.

    With ``weights``, the unquantized model is measured first as the reference, then its
    weights are quantized (and calibrated) as that config says and every cache spec runs on
    the quantized model. ``total_mib`` adds the cache for one sequence of ``context`` tokens.
    """
    from bitsmith.bench.hf import _require_hf, calibration_batches, wikitext2_tokens

    transformers, _ = _require_hf()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    tokens = wikitext2_tokens(tok, "test", eval_tokens)
    base_bytes = kv_bytes_per_token(model.config, None)
    short = _SHORT.get(dtype, dtype)
    points: list[KVPoint] = []

    def measure(spec: KVSpec | None, weight_label: str, ref_nll: float | None) -> float:
        label = spec.name if spec else f"{short} cache"
        if progress:
            progress(f"{weight_label} weights, {label}")
        t0 = time.perf_counter()
        res = kv_perplexity(model, tokens, spec, window=window, stride=stride)
        ref = res.nll if ref_nll is None else ref_nll
        nbytes = kv_bytes_per_token(model.config, spec)
        bits = (16.0, 16.0) if spec is None else (spec.bits_per_element("key"), spec.bits_per_element("value"))
        mib = model_bytes(model) / 2**20
        points.append(
            KVPoint(
                label,
                *bits,
                kib_per_token=nbytes / 1024,
                compression=base_bytes / nbytes,
                perplexity=res.perplexity,
                delta_pct=100 * (math.exp(res.nll - ref) - 1),
                seconds=time.perf_counter() - t0,
                weights=weight_label,
                model_mib=mib,
                total_mib=mib + nbytes * context / 2**20,
            )
        )
        return res.nll

    ref_nll = measure(None, short, None)
    if weights is not None:
        from bitsmith.calib import calibrate_and_quantize

        if progress:
            progress(f"quantizing weights: {weights.label}")
        calib = calibration_batches(wikitext2_tokens(tok, "train", 200_000), n_calib, calib_len)
        calibrate_and_quantize(
            model,
            [c.to(device) for c in calib],
            method=weights.method,  # type: ignore[arg-type]
            scheme=weights.scheme,
            group_size=weights.group_size,
            act_bits=weights.act_bits,
        )
        measure(None, weights.label, ref_nll)
    for spec in specs:
        measure(spec, weights.label if weights else short, ref_nll)
    return KVSweep(model_id, n_params, int(tokens.numel()), points, context)
