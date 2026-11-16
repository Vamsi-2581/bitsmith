"""Drive the command line interface the way a user would."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bitsmith import __version__
from bitsmith.cli import main


def test_version_via_module_entry_point() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "bitsmith", "--version"], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == f"bitsmith {__version__}"


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "int4" in out and "rtx3050-laptop" in out


@pytest.mark.parametrize("fmt", ["int4", "int3", "int2", "ternary", "ternary-b3", "sparse24"])
def test_pack(fmt: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pack", "--format", fmt, "--seed", "3"]) == 0
    assert "round trip" in capsys.readouterr().out


def test_sqnr(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sqnr", "--shape", "64x256", "--outliers", "2", "--group-size", "64"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n|") >= 12  # header, separator and 11 schemes
    assert "| nf4 " in out and "| mxfp4 " in out


def test_sqnr_rejects_bad_shape() -> None:
    with pytest.raises(SystemExit):
        main(["sqnr", "--shape", "64by64"])


def test_roofline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["roofline", "--params", "1e9", "--hardware", "t4"]) == 0
    out = capsys.readouterr().out
    assert "Tesla T4" in out and "memory" in out


@pytest.mark.slow
def test_sweep_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_file = tmp_path / "sweep.json"
    args = [
        "sweep",
        "--vocab",
        "32",
        "--dim",
        "32",
        "--layers",
        "1",
        "--steps",
        "40",
        "--train-tokens",
        "4000",
        "--eval-tokens",
        "400",
        "--window",
        "64",
        "--group-size",
        "32",
        "--json",
        str(out_file),
    ]
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "Pareto frontier" in text
    data = json.loads(out_file.read_text())
    labels = [p["label"] for p in data["points"]]
    assert labels[0] == "fp32" and "int4-g32+gptq" in labels and "ternary" in labels
    assert data["optimal_perplexity"] > 1
    assert all(p["perplexity"] > 1 for p in data["points"])


def test_kv_command_prints_table_and_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import bitsmith.kv.hf as kvhf

    seen: list[object] = []

    def fake_sweep(model_id: str, specs: list, **kw: object) -> kvhf.KVSweep:
        seen.extend(s.name for s in specs)
        seen.append(kw["weights"])
        pts = [
            kvhf.KVPoint("fp32 cache", 16, 16, 9.0, 1.0, 10.0, 0.0, 1.0),
            kvhf.KVPoint("k4cv4", 4.5, 4.5, 2.53, 3.56, 10.2, 2.0, 2.0),
        ]
        return kvhf.KVSweep(model_id, 1000, 512, pts)

    monkeypatch.setattr(kvhf, "run_kv_sweep", fake_sweep)
    out_file = tmp_path / "kv.json"
    assert main(["kv", "--configs", "k4cv4", "--weights", "nf4+gptq", "--json", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert seen[0] == "k4cv4"
    assert getattr(seen[1], "label", None) == "nf4-g64+gptq"
    assert "MiB at 32k" in out
    assert "| k4cv4 " in out and "KiB/token" in out
    data = json.loads(out_file.read_text())
    assert data["kind"] == "kv-sweep" and data["points"][1]["compression"] == 3.56


def test_kv_command_rejects_bad_spec() -> None:
    with pytest.raises(ValueError, match="cache spec"):
        main(["kv", "--configs", "k4"])
