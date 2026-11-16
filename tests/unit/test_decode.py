"""Greedy decoding with a static cache: the eager path matches ``generate`` exactly."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from bitsmith.decode import GraphDecoder, tokens_per_s  # noqa: E402


def _tiny_llama() -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def test_eager_static_decode_matches_generate() -> None:
    m = _tiny_llama()
    prompt = torch.randint(0, 64, (1, 7), generator=torch.Generator().manual_seed(1))
    ref = m.generate(prompt, max_new_tokens=12, min_new_tokens=12, do_sample=False, pad_token_id=0)[:, 7:]
    dec = GraphDecoder(m, max_len=32)
    assert not dec.use_graph
    assert torch.equal(dec.generate(prompt, 12), ref)
    assert torch.equal(dec.generate(prompt, 12), ref)  # the cache is reset between calls
    assert torch.equal(dec.generate(prompt, 1), ref[:, :1])


def test_decoder_validation() -> None:
    m = _tiny_llama()
    with pytest.raises(ValueError, match="CUDA"):
        GraphDecoder(m, 16, use_graph=True)
    dec = GraphDecoder(m, 16)
    with pytest.raises(ValueError, match="shape"):
        dec.generate(torch.zeros(2, 3, dtype=torch.long), 2)
    with pytest.raises(ValueError, match="max_len"):
        dec.generate(torch.zeros(1, 10, dtype=torch.long), 7)


def test_tokens_per_s_counts_runs() -> None:
    calls = []
    rate = tokens_per_s(lambda: calls.append(1), new_tokens=10, repeats=2, cuda=False)
    assert len(calls) == 3 and rate > 0
