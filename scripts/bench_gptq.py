"""Time GPTQ per layer for the layer shapes of Qwen2.5-1.5B.

Each GPTQ call has a fixed part (factoring the inverse Hessian: float64 shown for
reference, the default is float32 on CUDA) and the column loop.
The loop runs either eagerly (one kernel per tensor operation) or with every 32-column
inner block captured as a CUDA graph and replayed (ADR 0013). Reports the median of
``--repeats`` runs after one warm-up run.

Usage:
    python scripts/bench_gptq.py --device cuda --json results/gptq_speed_rtx3090.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from bitsmith.calib import damped_inverse_cholesky, gptq_quantize, gptq_quantize_table
from bitsmith.nn import make_codec
from bitsmith.report import table

# (name, out_features, in_features) of the linear layers in one Qwen2.5-1.5B block
SHAPES = [
    ("q_proj / o_proj", 1536, 1536),
    ("k_proj / v_proj", 256, 1536),
    ("gate_proj / up_proj", 8960, 1536),
    ("down_proj", 1536, 8960),
]


def _median_ms(fn: Callable[[], object], device: str, repeats: int) -> float:
    def sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    fn()
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _measure(name: str, out_f: int, in_f: int, dev: str, repeats: int) -> dict[str, Any]:
    x = torch.randn(2048, in_f, device=dev)
    h = 2 * x.T @ x / x.shape[0]
    w = torch.randn(out_f, in_f, device=dev) * 0.02
    nf4 = make_codec("nf4", 64)

    def int4(graph: bool) -> object:
        return gptq_quantize(w, h, group_size=64, act_order=True, use_cuda_graph=graph)

    return {
        "layer": name,
        "shape": [out_f, in_f],
        "hessian_fp64_ms": _median_ms(lambda: damped_inverse_cholesky(h, dtype=torch.float64), dev, repeats),
        "hessian_ms": _median_ms(lambda: damped_inverse_cholesky(h), dev, repeats),
        "int4_eager_ms": _median_ms(lambda: int4(False), dev, repeats),
        "int4_graph_ms": _median_ms(lambda: int4(True), dev, repeats),
        "nf4_graph_ms": _median_ms(lambda: gptq_quantize_table(w, h, nf4), dev, repeats),  # type: ignore[arg-type]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.manual_seed(0)
    dev = args.device
    rows, records = [], []
    for name, out_f, in_f in SHAPES:
        rec = _measure(name, out_f, in_f, dev, args.repeats)
        records.append(rec)
        speedup = rec["int4_eager_ms"] / rec["int4_graph_ms"]
        times = [rec[k] for k in ("hessian_fp64_ms", "hessian_ms", "int4_eager_ms", "int4_graph_ms")]
        rows.append([name, f"{out_f}x{in_f}", *times, speedup, rec["nf4_graph_ms"]])
    gpu = torch.cuda.get_device_name() if dev.startswith("cuda") else "cpu"
    print(f"GPTQ per layer, act-order, group 64, {gpu}, median of {args.repeats} (ms)")
    headers = ["layer", "shape", "H^-1 fp64", "H^-1 default", "int4 eager", "int4 graph", "graph speedup", "nf4 graph"]
    print(table(headers, rows))
    if args.json:
        payload = {"device": gpu, "torch": torch.__version__, "repeats": args.repeats, "records": records}
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
