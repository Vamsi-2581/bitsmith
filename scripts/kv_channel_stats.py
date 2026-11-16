"""Measure how uneven the key and value channels of a model's KV cache are.

For each layer and KV head: the mean magnitude of every channel over 1024 WikiText-2
tokens, then the ratio of the largest channel to the median channel. A large ratio means
a per-token grid must stretch over a few big channels. Prints the median and range of the
ratio over all layers and heads, for keys and for values.

    python scripts/kv_channel_stats.py --device cuda > results/kv_channels.txt
"""

from __future__ import annotations

import argparse

import torch

from bitsmith.bench.hf import _require_hf, wikitext2_tokens
from bitsmith.report import table

MODELS = ["HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]


@torch.no_grad()
def channel_ratios(model_id: str, device: str, tokens: int) -> tuple[list[float], list[float]]:
    transformers, _ = _require_hf()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    ids = wikitext2_tokens(tok, "test", tokens).to(device)[None]
    cache = transformers.DynamicCache()
    model(ids, past_key_values=cache, use_cache=True)
    ratios: tuple[list[float], list[float]] = ([], [])
    for layer in cache.layers:
        for t, out in zip((layer.keys, layer.values), ratios, strict=True):
            mag = t.float().abs().mean(2)[0]  # (heads, head_dim): mean |x| of each channel
            out.extend((mag.amax(-1) / mag.median(-1).values).tolist())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=1024)
    args = parser.parse_args()
    rows: list[list[object]] = []
    for model_id in MODELS:
        keys, values = channel_ratios(model_id, args.device, args.tokens)
        for name, r in (("keys", keys), ("values", values)):
            rows.append([model_id, name, len(r), float(torch.tensor(r).median()), min(r), max(r)])
    print(f"Largest channel / median channel, mean |x| over {args.tokens} WikiText-2 tokens, per layer and KV head")
    print(table(["model", "tensor", "heads", "median ratio", "min", "max"], rows))


if __name__ == "__main__":
    main()
