# 0018. SmoothQuant chooses per layer, and may choose not to smooth

- Status: Accepted (builds on ADR 0007)
- Date: 2026-11-16

## Context

SmoothQuant moved every layer's activation range into its weights with one `alpha = 0.5`.
That worked on SmolLM2 and Qwen2.5, but on GPT-2 W8A8 + SmoothQuant cost +73% perplexity
while plain W8A8 cost +0.7%: smoothing made things a hundred times worse.

Per-layer output errors inside the quantized model (WikiText-2 windows) found the cause:
the first MLP down projection, `h.0.mlp.c_proj`, went from 4.7% error without smoothing
to 37% with it, and every `mlp.c_proj` was somewhat worse. The input of a down
projection is a GELU output in which some channels rarely fire. Such a channel shows a
small maximum during calibration, so its factor `s` is small and `1 / s` large (13 here);
when it fires on new text, that one channel takes over the token's INT8 range and the
other channels round to almost nothing.

The SmoothQuant paper smooths only layers fed by a LayerNorm (attention QKV and the first
MLP projection), whose smoothing can be folded into the norm. bitsmith has no model graph
and, because the scale is an explicit input buffer (ADR 0007), smoothed every linear layer.

## Decision

- **Per-layer search on calibration data** (`smoothquant_search`): for each layer, try no
  smoothing and `alpha` in {0.25, 0.5, 0.75}; keep the option with the lowest mean squared
  output error of the quantized layer (INT8 weights, per-token INT8 activations, `1 / s`
  rounded to fp16 as stored) on the calibration samples.
- This is the default of `calibrate_and_quantize(method="smoothquant")`. Passing `alpha`
  keeps the old behaviour: one fixed value, every layer smoothed.
- Only calibration data drives the choice; the test split is never consulted.

Considered: smoothing only layers whose names look like QKV or up projections. Simple, but
it ties the library to naming conventions, and the search reaches the same decision from
the data alone.

## Consequences

- GPT-2: W8A8 + SmoothQuant goes from +73% to +0.50%, better than plain W8A8 (+0.71%).
- The SmolLM2 and Qwen2.5 results were re-measured with the new default
  (`results/gpu_smoothquant_*.txt`).
- Calibration keeps activation samples for SmoothQuant now (as AWQ does), and runs four
  quantized matmuls per layer; on these models that adds seconds.
