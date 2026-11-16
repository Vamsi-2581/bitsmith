"""KV-cache quantization: grids, packed storage, streaming semantics, Hugging Face cache."""

from __future__ import annotations

import pytest
import torch

from bitsmith.eval import sqnr_db
from bitsmith.kv import KVSpec, KVStream, dequantize, parse_kv_spec, quantize


def _kv(t: int = 32, d: int = 64, heads: int = 2, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, heads, t, d, generator=g)


@pytest.mark.parametrize(
    ("name", "spec"),
    [
        ("k8v8", KVSpec(8, 8, "token", "token")),
        ("k4cv4", KVSpec(4, 4, "channel", "token")),
        ("k2cv2-g32", KVSpec(2, 2, "channel", "token", 32)),
        ("k16v4c", KVSpec(16, 4, "token", "channel")),
    ],
)
def test_parse_and_name_round_trip(name: str, spec: KVSpec) -> None:
    assert parse_kv_spec(name) == spec
    assert spec.name == name


@pytest.mark.parametrize("bad", ["k4", "kv4", "k5v4", "k4v4-g0", "K4V4"])
def test_parse_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match=r"bits|group|spec"):
        parse_kv_spec(bad)


def test_spec_validation() -> None:
    with pytest.raises(ValueError, match="axis"):
        KVSpec(key_axis="row")  # type: ignore[arg-type]


def test_bits_per_element_and_chunk() -> None:
    spec = parse_kv_spec("k4cv2")
    assert spec.bits_per_element("key") == 4.5  # 4 bits + fp16 scale and min over 64
    assert spec.bits_per_element("value") == 2.5
    assert parse_kv_spec("k16v8").bits_per_element("key") == 16
    assert spec.chunk == 64
    assert parse_kv_spec("k4v4").chunk == 1
    assert parse_kv_spec("k16cv4").chunk == 1  # an unquantized axis needs no alignment


@pytest.mark.parametrize("bits", [8, 4, 3, 2])
@pytest.mark.parametrize("axis", ["token", "channel"])
def test_round_trip_error_within_half_step(bits: int, axis: str) -> None:
    x = _kv()
    q = quantize(x, bits, axis, 16)  # type: ignore[arg-type]
    y = dequantize(q, torch.float32)
    assert y.shape == x.shape
    step = q.scale.float()
    if axis == "token":
        err = (x - y).reshape(*x.shape[:3], -1, 16).abs()
    else:
        err = (x - y).reshape(1, 2, -1, 16, 64).abs()
    assert bool((err <= step / 2 + 1e-3).all())


def test_packed_bytes_match_bits() -> None:
    x = _kv(t=64, d=64)
    for bits in (8, 4, 3, 2):
        q = quantize(x, bits, "channel", 64)
        assert q.packed.numel() == x.numel() * bits // 8
        assert q.nbytes * 8 == pytest.approx(x.numel() * KVSpec(bits, bits).bits_per_element("key"))


def test_constant_group_is_exact() -> None:
    x = torch.full((1, 1, 4, 16), 0.75)
    assert torch.equal(dequantize(quantize(x, 4, "token", 16), torch.float32), x)


def test_shape_checks() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        quantize(_kv(d=48), 4, "token", 32)
    with pytest.raises(ValueError, match="groups"):
        quantize(_kv(t=10), 4, "channel", 4)


def test_per_channel_wins_on_outlier_channels() -> None:
    """Keys have a few channels large for every token; per-channel groups isolate them."""
    x = _kv(t=128)
    big = [3, 17]
    x[..., big] *= 20
    tok = dequantize(quantize(x, 4, "token", 64), torch.float32)
    ch = dequantize(quantize(x, 4, "channel", 64), torch.float32)
    small = [c for c in range(64) if c not in big]  # per token, these share the outliers' step
    assert sqnr_db(x[..., small], ch[..., small]) > sqnr_db(x[..., small], tok[..., small]) + 12


def test_token_stream_quantizes_on_write_and_ignores_chunking() -> None:
    x = _kv(t=24)
    one = KVStream(4, "token", 32)
    out = one.append(x)
    pieces = KVStream(4, "token", 32)
    for s in range(0, 24, 5):
        last = pieces.append(x[:, :, s : s + 5])
    assert torch.equal(out, last)
    assert torch.equal(out, dequantize(quantize(x, 4, "token", 32), torch.float32))
    assert one.residual is None and one.length == 24


def test_channel_stream_keeps_partial_group_exact() -> None:
    x = _kv(t=20)
    s = KVStream(4, "channel", 8)
    out = s.append(x)
    assert torch.equal(out, x)  # returned before quantizing, as in decoding
    assert s.length == 20 and s.residual is not None and s.residual.shape[2] == 4
    later = s.read()
    assert torch.equal(later[:, :, 16:], x[:, :, 16:])  # residual stays exact
    assert not torch.equal(later[:, :, :16], x[:, :, :16])  # two full groups were quantized
    assert torch.equal(later[:, :, :16], dequantize(quantize(x[:, :, :16], 4, "channel", 8), torch.float32))


def test_channel_stream_matches_token_by_token() -> None:
    x = _kv(t=24)
    chunked = KVStream(3, "channel", 8)
    for s in range(0, 24, 8):
        chunked.append(x[:, :, s : s + 8])
    single = KVStream(3, "channel", 8)
    for t in range(24):
        single.append(x[:, :, t : t + 1])
    assert torch.equal(chunked.read(), single.read())
    assert single.nbytes == chunked.nbytes


def test_stream_16_bit_passthrough_and_bytes() -> None:
    x = _kv(t=8).half()
    s = KVStream(16, "token", 64)
    s.append(x[:, :, :5])
    assert torch.equal(s.append(x[:, :, 5:]), x)
    assert s.nbytes == x.numel() * 2
    empty = KVStream(4, "token", 64)
    assert empty.nbytes == 0


def test_stream_keeps_input_dtype() -> None:
    s = KVStream(4, "token", 64)
    assert s.append(_kv(t=4).to(torch.bfloat16)).dtype == torch.bfloat16


# Hugging Face cache: run only where transformers is installed --------------------------
def _tiny_llama() -> torch.nn.Module:
    transformers = pytest.importorskip("transformers")
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
def test_hf_unquantized_spec_matches_plain_forward() -> None:
    from bitsmith.kv.hf import kv_logits_fn

    m = _tiny_llama()
    ids = torch.randint(0, 64, (1, 32), generator=torch.Generator().manual_seed(1))
    assert torch.equal(kv_logits_fn(m, KVSpec(16, 16))(ids), m(ids).logits)
    assert torch.equal(kv_logits_fn(m, None)(ids), m(ids).logits)


@torch.no_grad()
@pytest.mark.parametrize("name", ["k4cv4-g8", "k4v4-g8", "k2cv3c-g8"])
def test_hf_chunked_pass_equals_token_by_token_decoding(name: str) -> None:
    from bitsmith.kv.hf import kv_bytes_per_token, kv_logits_fn, quantized_cache

    m = _tiny_llama()
    spec = parse_kv_spec(name)
    ids = torch.randint(0, 64, (1, 32), generator=torch.Generator().manual_seed(2))
    cache = quantized_cache(spec)
    stepwise = torch.cat([m(ids[:, t : t + 1], past_key_values=cache, use_cache=True).logits for t in range(32)], 1)
    chunked = kv_logits_fn(m, spec)(ids)
    torch.testing.assert_close(chunked, stepwise, atol=1e-5, rtol=1e-5)
    assert (chunked - m(ids).logits).abs().max() > 1e-3  # and the cache really is quantized
    assert cache.get_seq_length() == 32
    assert cache.nbytes == kv_bytes_per_token(m.config, spec) * 32  # 32 tokens = whole groups


@torch.no_grad()
def test_hf_generate_and_perplexity() -> None:
    from bitsmith.kv.hf import kv_bytes_per_token, kv_perplexity, quantized_cache

    m = _tiny_llama()
    ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(3))
    out = m.generate(ids, past_key_values=quantized_cache(parse_kv_spec("k8v8-g16")), max_new_tokens=4, do_sample=False)
    assert out.shape == (1, 12)
    tokens = torch.randint(0, 64, (200,), generator=torch.Generator().manual_seed(4))
    base = kv_perplexity(m, tokens, None, window=64, stride=32)
    k8 = kv_perplexity(m, tokens, parse_kv_spec("k8v8-g16"), window=64, stride=32)
    k2 = kv_perplexity(m, tokens, parse_kv_spec("k2v2-g16"), window=64, stride=32)
    assert base.n_tokens == k8.n_tokens == 199
    assert abs(k8.nll - base.nll) < abs(k2.nll - base.nll)
    assert kv_bytes_per_token(m.config, None) == 2 * 2 * 16 * 2 * 2  # layers x heads x dim x (k, v) x 2 bytes


def test_hf_cache_rejects_beam_search_and_crop() -> None:
    from bitsmith.kv.hf import quantized_cache

    m = _tiny_llama()
    cache = quantized_cache(parse_kv_spec("k4v4-g16"))
    with torch.no_grad():
        m(torch.zeros(1, 4, dtype=torch.long), past_key_values=cache, use_cache=True)
    layer = cache.layers[0]
    with pytest.raises(NotImplementedError):
        layer.reorder_cache(torch.tensor([0]))
    with pytest.raises(NotImplementedError):
        layer.crop(2)
    layer.reset()
    assert layer.get_seq_length() == 0


def test_model_bytes_counts_tied_tensors_once() -> None:
    from bitsmith.kv.hf import model_bytes

    lin = torch.nn.Linear(4, 4, bias=False)
    m = torch.nn.Sequential(lin, torch.nn.Linear(4, 4, bias=False))
    m[1].weight = lin.weight  # tied
    m.register_buffer("codes", torch.zeros(10, dtype=torch.uint8))
    assert model_bytes(m) == 16 * 4 + 10


@torch.no_grad()
def test_hf_sweep_with_quantized_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """The combined sweep: reference first, then quantized weights with each cache."""
    transformers = pytest.importorskip("transformers")
    import bitsmith.bench.hf as bhf
    from bitsmith.bench import SweepConfig
    from bitsmith.kv.hf import run_kv_sweep

    model = _tiny_llama()
    tokens = torch.randint(0, 64, (300,), generator=torch.Generator().manual_seed(5))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda _id: None))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(lambda _id, **kw: model))
    monkeypatch.setattr(bhf, "wikitext2_tokens", lambda tok, split, n: tokens)
    res = run_kv_sweep(
        "tiny",
        [parse_kv_spec("k4cv4-g16")],
        weights=SweepConfig("int4-g16", "int4", group_size=16),
        eval_tokens=None,
        window=64,
        stride=32,
        n_calib=2,
        calib_len=32,
        context=1024,
    )
    ref, wq, both = res.points
    assert (ref.weights, ref.label, ref.delta_pct) == ("fp32", "fp32 cache", 0.0)
    assert (wq.weights, wq.label) == ("int4-g16", "fp32 cache")
    assert both.label == "k4cv4-g16" and both.weights == "int4-g16"
    assert wq.model_mib < ref.model_mib  # packed weights are smaller
    assert both.total_mib < wq.total_mib  # and so is the quantized cache
    assert wq.delta_pct != 0.0
