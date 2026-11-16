"""Time real W8A8 (INT8 x INT8 on tensor cores) against fp16 and simulated W8A8.

For each layer shape and row count:

* ``fp16``           ``F.linear`` with an fp16 weight (cuBLAS)
* ``w8a8-simulated`` the reference runtime: round activations to INT8 codes, dequantize
                     weights and activations, fp16 matmul (what the accuracy results use)
* ``w8a8-int8``      the ``int8`` backend: per-token INT8 activations, ``torch._int_mm``

Every implementation is replayed from a CUDA graph, so the times are GPU time without
Python launch overhead. The int8 time includes quantizing the activations.

    python scripts/bench_int8.py --json results/int8_rtx3090.json > results/int8_rtx3090.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from bench_kernels import SHAPES, bench_graph
from torch import nn

from bitsmith.nn import QLinear
from bitsmith.report import table

ROWS = (1, 16, 128, 512, 2048)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write results to this JSON file")
    args = parser.parse_args()
    torch.manual_seed(0)
    records = []
    for name, (out_f, in_f) in SHAPES.items():
        lin = nn.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.float16)
        q = QLinear.from_linear(lin, "int8", act_bits=8)
        w16 = lin.weight
        for rows in ROWS:
            x = torch.randn(rows, in_f, device="cuda", dtype=torch.float16)
            q.set_backend("reference")
            sim = bench_graph(lambda q=q, x=x: q(x))
            q.set_backend("int8")
            real = bench_graph(lambda q=q, x=x: q(x))
            fp16 = bench_graph(lambda w=w16, x=x: F.linear(x, w))
            err = (q(x).float() - F.linear(x, w16).float()).norm() / F.linear(x, w16).float().norm()
            records.append(
                {
                    "layer": name,
                    "rows": rows,
                    "fp16_us": fp16,
                    "w8a8_simulated_us": sim,
                    "w8a8_int8_us": real,
                    "speedup_vs_fp16": fp16 / real,
                    "rel_error_vs_fp16": float(err),
                }
            )
    gpu = torch.cuda.get_device_name()
    print(f"W8A8 linear layers on {gpu}: CUDA graph replay, median microseconds")
    headers = ["layer", "rows", "fp16", "w8a8 simulated", "w8a8 int8", "int8 vs fp16", "rel. error"]
    rows_out = [
        [
            r["layer"],
            r["rows"],
            r["fp16_us"],
            r["w8a8_simulated_us"],
            r["w8a8_int8_us"],
            f"{r['speedup_vs_fp16']:.2f}x",
            f"{r['rel_error_vs_fp16']:.4f}",
        ]
        for r in records
    ]
    print(table(headers, rows_out))
    if args.json:
        payload = {"gpu": gpu, "torch": torch.__version__, "records": records}
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
