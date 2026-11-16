"""GPT-2 style Conv1D layers are converted like nn.Linear."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bitsmith.calib import calibrate_and_quantize
from bitsmith.nn import QLinear, linearize_conv1d, quantize_model


class Conv1D(nn.Module):
    """Same layout as ``transformers.pytorch_utils.Conv1D``: weight ``(in, out)``."""

    def __init__(self, nf: int, nx: int) -> None:
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.randn(nx, nf) * 0.05)
        self.bias = nn.Parameter(torch.randn(nf) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).reshape(*x.shape[:-1], self.nf)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = Conv1D(64, 32)
        self.c_proj = Conv1D(32, 64)
        self.lm_head = nn.Linear(32, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.c_proj(torch.relu(self.c_fc(x))))


@torch.no_grad()
def test_linearize_keeps_outputs() -> None:
    torch.manual_seed(0)
    m = _Mlp()
    x = torch.randn(2, 5, 32)
    ref = m(x)
    assert linearize_conv1d(m) == ["c_fc", "c_proj"]
    assert isinstance(m.c_fc, nn.Linear) and m.c_fc.weight.shape == (64, 32)
    torch.testing.assert_close(m(x), ref)
    assert linearize_conv1d(m) == []  # nothing left to convert


@torch.no_grad()
def test_quantize_model_converts_conv1d() -> None:
    torch.manual_seed(0)
    m = _Mlp()
    x = torch.randn(4, 32)
    ref = m(x)
    report = quantize_model(m, "int8")
    assert report.converted == ["c_fc", "c_proj"] and report.skipped == ["lm_head"]
    assert isinstance(m.c_fc, QLinear)
    assert (m(x) - ref).norm() / ref.norm() < 0.02


def test_gptq_calibrates_conv1d() -> None:
    torch.manual_seed(0)
    m = _Mlp()
    calib = [torch.randn(1, 16, 32) for _ in range(4)]
    report = calibrate_and_quantize(m, calib, method="gptq", scheme="int4", group_size=32)
    assert sorted(report.converted) == ["c_fc", "c_proj"]


@torch.no_grad()
def test_real_gpt2_config() -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2)
    m = transformers.GPT2LMHeadModel(cfg).eval()
    ids = torch.randint(0, 64, (1, 16))
    ref = m(ids).logits
    report = quantize_model(m, "int8")
    assert len(report.converted) == 8  # c_attn, attn.c_proj, c_fc, mlp.c_proj per block
    assert (m(ids).logits - ref).norm() / ref.norm() < 0.02


def test_sweep_counts_conv1d_weights() -> None:
    from bitsmith.bench import SweepConfig, run_sweep

    class _LM(_Mlp):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 32)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return super().forward(self.embed(ids))

    torch.manual_seed(0)
    tokens = torch.randint(0, 8, (40,))
    configs = [SweepConfig("fp32", "baseline"), SweepConfig("int8", "int8")]
    pts = run_sweep(_LM(), tokens, [], configs, window=16, stride=8)
    assert pts[1].bits_per_weight == pytest.approx(8.0, abs=0.6)
