# 0013. Fast GPTQ: transposed two-level loop, CUDA graphs, cheaper Hessian factor

- Status: Accepted
- Date: 2026-11-12

## Context

GPTQ walks the input columns one at a time; each step is about a dozen small tensor
operations on one column. Profiling the v0.2.0 loop showed three costs:

1. On the CPU, 80% of the time was the rank-1 update of the rest of a 128-column block,
   on a row-major matrix where each column is a strided slice.
2. On the GPU, every operation is a kernel launch (about 20 us each on a shared RTX 3090)
   and `int(perm[col])` forced a GPU to CPU synchronization per column. The math per
   column is tiny, so the GPU sat mostly idle.
3. The inverse Hessian factor took three float64 factorizations. Consumer GPUs run
   float64 at 1/64 of the float32 rate: 2 to 3.5 s for `down_proj` (8960 inputs).

## Decision

- **One shared loop** (`_obs_loop`) for INT4 and the grid formats, on the transposed
  weight so each column is a contiguous row, with **two levels of lazy updates**: rank-1
  updates only inside a 32-column inner block, a matmul to the rest of the outer block
  when an inner block ends, and a matmul to all later columns when an outer block ends.
  It is the same arithmetic in a different order. INT4 codes are recovered after the
  loop from the dequantized values, so the loop needs no per-column bookkeeping and no
  host synchronization.
- **CUDA graphs**: on CUDA, the first full inner block is captured as a graph and every
  later block is copied into its fixed buffers and replayed, so one launch replaces about
  300. Partial blocks run eagerly. Both paths run the same kernels in the same order,
  and a GPU test checks they agree bit for bit. `use_cuda_graph=False` turns it off.
- **Reversed-order factorization**: `U = P L^-1 P` with `P H P = L L^T`, one Cholesky
  plus one triangular inverse instead of factorize, invert, factorize. Tested equal to
  the textbook factor.
- **Float32 factorization on CUDA**, float64 on the CPU. A float32 factorization that
  fails or produces non-finite values is redone in float64. On a 1536x8960 layer with
  outlier channels, the float32 factor gives the same GPTQ output error as float64 to a
  ratio of 1.00001; 1.5% of codes differ, which are equally good rounding choices.

## Consequences

- Measured on an RTX 3090 shared with other jobs (`scripts/bench_gptq.py`,
  `results/gptq_speed_rtx3090.txt`): the factor is 4x to 37x faster, the column loop with
  graphs is 2x to 10x faster than eager, and a whole layer is about 3.5x to 15x faster
  than the v0.2.0 implementation, depending on shape and load.
- Results are unchanged beyond float rounding: against v0.2.0, 99.7% to 99.99% of codes
  are identical and the expected output error matches to 0.01%. End to end, INT4 GPTQ on
  Qwen2.5-0.5B gives WikiText-2 perplexity 13.13 against 13.12 before
  (`results/gptq_recheck_qwen2.5_0.5b.txt`), and the whole run (calibration, both evaluations and decode timing) took 4.5 minutes.
- CPU results stay on the float64 path, so CPU tests and committed CPU results do not
  move. GPU runs may differ from v0.2.0 in a small fraction of codes.
- The graph path depends on every operation in the round function being capturable (no
  host synchronization). `round_to_table` now clamps with a tensor bound for this reason.
