# 0015. KV-cache quantization: per-channel keys, per-token values, exact chunked evaluation

- Status: Accepted
- Date: 2026-11-13

## Context

Everything before this record shrinks the weights. During decoding each layer also keeps
the keys and values of every earlier token, `2 x layers x kv_heads x head_dim` numbers per
token, growing with context length and batch size. For Qwen2.5-1.5B in bf16 that is 28 KiB
per token: 0.9 GiB for one 32k-token sequence, and more than the 2.4 GiB of bf16 weights
from a batch of 3 such sequences.

Three questions had to be settled: how to group the cache for quantization, how to store
it, and how to measure its effect without cheating.

## Decision

- **Asymmetric min/max grid per group**, fp16 scale and fp16 minimum (`x' = q s + m`), at
  8, 4, 3 or 2 bits, packed with the existing INT4, INT3 and INT2 packers. 16 bits leaves a
  tensor unquantized, so keys and values can be studied separately.
- **Group axis chosen per tensor**: per token (a group is `group` channels of one token,
  quantized on write) or per channel (a group is one channel over `group` tokens). The
  default spec `k4cv4` puts keys per channel and values per token, the KIVI layout.
  Measured on WikiText-2 (`scripts/kv_channel_stats.py`), the largest key channel of a
  typical head is 6x to 10x the median channel, and up to 270x in Qwen2.5-1.5B; for values
  it is 2.5x to 3.8x. A per-token grid has to cover the large key channels and leaves few
  levels for the rest.
- **Per-channel groups wait in full precision until complete.** `KVStream.append` returns
  what attention should read before it quantizes a group that the new tokens complete,
  which is what token-by-token decoding sees. At most `group - 1` tokens per layer stay in
  full precision.
- **A `transformers` cache class** (`quantized_cache(spec)`), built from the per-layer
  cache API that transformers 4.57 and 5.x share, so it works in plain forward passes and
  in `generate`. Beam search and cropping raise `NotImplementedError`.
- **Exact evaluation in chunks.** `kv_perplexity` runs each window through a fresh cache
  in chunks of `group` tokens. A test checks that this matches token-by-token decoding to
  float rounding. Two shortcuts were rejected: Hugging Face's `QuantizedCache` returns the
  prompt unquantized on the first call, so a one-pass evaluation would measure a
  full-precision cache; and quantizing the whole window up front would let each query see
  a group whose statistics include tokens after it.
- **The cache stores packed codes; attention reads a dequantized copy** rebuilt each step.
  This measures accuracy and memory. Speed needs a fused attention kernel (roadmap).

## Consequences

- Full WikiText-2 test split, bf16 weights (`results/kv_*.txt`), SmolLM2-135M /
  Qwen2.5-0.5B / Qwen2.5-1.5B:

  | Cache | Bits/element | Perplexity increase |
  |-------|-------------:|---------------------|
  | `k8v8` | 8.5 | +0.05% / +3.5% / +0.5% |
  | `k4v4` | 4.5 | +8.3% / +878% / +245,592% |
  | `k4cv4` | 4.5 | +1.2% / +1.1% / +0.8% |
  | `k3cv3` | 3.5 | +6.1% / +6.3% / +4.3% |
  | `k2cv2` | 2.5 | +93% / +115% / +67% |

  Per-channel keys are what make 4 bits work: per-token 4-bit keys break both Qwen models,
  whose key outliers are the largest. On Qwen2.5-0.5B, 4-bit per-channel keys even beat
  8-bit per-token keys.
- The chunked evaluation is launch-bound: about 1 s per 1024-token window on a shared RTX
  3090 against 0.03 s for one plain pass, so a per-channel config takes about 10 minutes
  on the full test split. Per-token configs need no chunks and run in one pass.
- `pack_int4`, `pack_int3` and `pack_int2` gained `validate=False`. Their range check reads
  the codes back to the host, and on a GPU shared with other jobs every such sync waited
  for a time slice, which dominated the run time.
