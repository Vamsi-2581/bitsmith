"""Greedy decoding with the one-token step replayed from a CUDA graph.

At batch 1 a decode step of a small model is a few hundred tiny kernels, and launching
them costs more than running them: Qwen2.5-1.5B decodes at about 40 tokens/s through
``generate`` while its weights could be read about 400 times per second. Capturing the
step once as a CUDA graph and replaying it turns those launches into one.

A graph replays fixed kernels on fixed memory, so everything the step touches must keep
its address: the cache is a ``transformers.StaticCache`` (preallocated to ``max_len``),
and the input token and its position live in small buffers that are overwritten before
each replay. The next token is picked on the GPU (``argmax``) and copied straight into the
input buffer, so the loop never waits for the host. On the CPU, or with
``use_graph=False``, the same step runs eagerly, which keeps the logic testable anywhere.

Requires the ``hf`` extra. Batch size 1, greedy only.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch
from torch import Tensor, nn


class GraphDecoder:
    """Greedy decoder for a Hugging Face causal LM with a static cache of ``max_len`` tokens."""

    def __init__(self, model: nn.Module, max_len: int, *, use_graph: bool | None = None) -> None:
        from transformers import StaticCache

        self.model = model
        self.device = next(model.parameters()).device
        self.max_len = max_len
        self.use_graph = self.device.type == "cuda" if use_graph is None else use_graph
        if self.use_graph and self.device.type != "cuda":
            raise ValueError("CUDA graphs need the model on a CUDA device")
        self.cache = StaticCache(config=model.config, max_cache_len=max_len)  # type: ignore[arg-type]
        self._ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._logits: Tensor | None = None

    def _step(self) -> Tensor:
        pos = self._pos
        cache = self.cache
        out = self.model(self._ids, cache_position=pos, position_ids=pos[None], past_key_values=cache, use_cache=True)
        return out.logits[:, -1]  # type: ignore[no-any-return]

    def _capture(self) -> None:
        # warm up on a side stream (lazy init, Triton autotuning), then record one step
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(side)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._logits = self._step()

    @torch.no_grad()
    def generate(self, prompt: Tensor, new_tokens: int) -> Tensor:
        """Greedy continuation of ``prompt`` (shape ``(1, T)``): returns ``(1, new_tokens)`` ids."""
        if prompt.dim() != 2 or prompt.shape[0] != 1:
            raise ValueError("prompt must have shape (1, T)")
        n = prompt.shape[1]
        if new_tokens < 1 or n + new_tokens > self.max_len:
            raise ValueError(f"need 1 <= new_tokens and prompt + new_tokens <= max_len ({self.max_len})")
        self.cache.reset()  # zeroes the buffers in place, so captured addresses stay valid
        prompt = prompt.to(self.device)
        out = self.model(prompt, cache_position=torch.arange(n, device=self.device), past_key_values=self.cache)
        tokens = torch.empty(1, new_tokens, dtype=torch.long, device=self.device)
        tokens[:, 0] = out.logits[:, -1].argmax(-1)
        if self.use_graph and self._graph is None and new_tokens > 1:
            self._ids.copy_(tokens[:, :1])
            self._pos.fill_(n)
            self._capture()
            self.cache.reset()  # the warm-up steps wrote into the cache: redo the prompt
            self.model(prompt, cache_position=torch.arange(n, device=self.device), past_key_values=self.cache)
        for i in range(1, new_tokens):
            self._ids.copy_(tokens[:, i - 1 : i])
            self._pos.fill_(n + i - 1)
            if self._graph is not None:
                self._graph.replay()
                logits = self._logits
                assert logits is not None
            else:
                logits = self._step()
            tokens[:, i] = logits.argmax(-1)
        return tokens


def tokens_per_s(fn: Any, new_tokens: int, repeats: int = 3, cuda: bool = True) -> float:
    """Median tokens/s of ``fn()`` producing ``new_tokens`` tokens; the first run is a warm-up."""
    rates = []
    for i in range(repeats + 1):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        if i:
            rates.append(new_tokens / (time.perf_counter() - t0))
    return statistics.median(rates)
