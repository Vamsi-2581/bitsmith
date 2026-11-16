# 0016. Decode with a static cache and a CUDA graph per token

- Status: Accepted
- Date: 2026-11-16

## Context

The fused INT4 kernel (ADR 0010) beats fp16 cuBLAS per layer at batch 1, but end to end
`generate` stayed slower than bf16: on Qwen2.5-1.5B, 25 tokens/s for INT4 GPTQ on the
fused kernel against 42 tokens/s for bf16. A decode step of a model this size is a few
hundred small kernels, and launching them from Python costs more than running them. A
faster GEMV cannot help while the step is launch bound.

## Decision

- **`GraphDecoder`** (`bitsmith.decode`): greedy decoding at batch 1 with a
  `transformers.StaticCache` preallocated to `max_len`. The prompt is prefilled eagerly;
  one decode step is captured as a CUDA graph and replayed for every new token.
- **Fixed addresses**: the input token and its position are small device buffers written
  before each replay, the cache is reset in place, and the logits buffer is reused. The
  next token is chosen with `argmax` on the GPU and copied into the input buffer, so the
  loop never synchronizes with the host until the end.
- **Warm-up before capture**: two eager steps on a side stream initialize lazy state and
  run Triton's autotuning, which cannot happen inside a capture. They write into the
  cache, so the prompt is prefilled again afterwards.
- **Same code without graphs**: `use_graph=False` (and every CPU run) executes the same
  step eagerly. A CPU test checks the eager path against `generate`, token for token; a
  GPU test checks graph replay against the eager path for bf16, the reference INT4
  runtime and the fused kernel.
- **Scope**: batch 1, greedy, a 16-bit static cache. The quantized KV cache (ADR 0015)
  grows dynamically and cannot be captured as is.

Considered: `torch.compile(mode="reduce-overhead")`, which also uses CUDA graphs. It
recompiles for new shapes, compiles slowly on first use, and hides what is captured; a
hand-written capture of one step is under 100 lines and easy to test.

## Consequences

- Measured results are in `results/decode_rtx3090.txt` and the README.
- Captured graphs pin the memory of the step; the decoder keeps one graph per instance.
- Anything that synchronizes with the host inside the model breaks capture. The
  bitsmith kernels avoid that by design (see also ADR 0013).
