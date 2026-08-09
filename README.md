# How to optimize in GPU -- FlyDSL / AMD CDNA edition

A port of [Liu-xiandong/How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU)
from CUDA to **FlyDSL** on **AMD Instinct (CDNA4 / gfx950)**.

The original is a set of optimization *ladders*: for each kernel, a sequence of
files where each one adds exactly one idea to the previous one, so the delta
between rungs is the lesson. That structure is preserved here rung for rung.
What changes is the machine underneath -- a 64-lane wavefront instead of a 32-lane
warp, 256 CUs instead of 80 SMs, matrix cores instead of SASS tuning -- and every
place the change forces a different answer is called out in the source and in
[`docs/porting-notes.md`](docs/porting-notes.md).

Every number below was measured on the hardware named in the table header. Every
rung is verified against a reference before it is timed; a wrong kernel is
reported `FAIL` and its time is suppressed.

```
AMD Instinct MI350X VF (gfx950, wave64) | 256 CU @ 2.2 GHz | LDS 160 KB/CU
HBM 8.0 TB/s | FlyDSL 0.2.4 | ROCm 7.2 | 90/90 rows correct
```

> **Disclaimer.** These are my own measurements, taken on one virtualised MI350X
> partition, as a personal project. They are **not** produced, reviewed,
> validated, or endorsed by AMD, and they are not official AMD performance data.
> Treat them as one person's reproducible data point, not as a benchmark result
> for the hardware. Peak figures quoted as denominators come from AMD's published
> CDNA 4 whitepaper; everything measured is mine.

---

## Quick start

```bash
export PYTHONPATH=$PWD
PY=/root/flydsl-wgrad-ragged/.venv/bin/python   # the venv with the FlyDSL wheel

$PY -m bench --list                 # every op and its rungs
$PY -m bench                        # run everything
$PY -m bench sgemm --shape 4096     # one op, one shape
$PY -m bench reduce --json out.json # machine-readable results
$PY -m pytest tests -q              # correctness only, fast
```

Or `make list` / `make bench` / `make test`.

---

## 1. elementwise -- how wide is one lane's transaction?

`C = A + B`. Pure HBM traffic, so the metric is bandwidth. The CUDA original's
`float` / `float2` / `float4` reinterpret casts become copy *atoms* --
`BufferCopy32b/64b/128b`, lowering to `buffer_load_dword/x2/x4`.

| rung | | N=32M | N=256M | vs rocBLAS |
|---|---|---|---|---|
| `v0_float` | one f32 per lane | 5283 GB/s | 5127 GB/s | 0.88x |
| `v1_float2` | two f32 per lane | 5649 GB/s | 5381 GB/s | 0.92x |
| `v2_float4` | four f32 per lane | **5923 GB/s** | **5869 GB/s** | **1.00x** |
| `v3_float4_x4` | 4x float4 per lane | 5574 GB/s | 5597 GB/s | 0.94x |

Same conclusion as the original -- wider is better, and `float4` matches the
vendor library exactly. `v3` is a **negative result kept on purpose**: more loads
in flight per lane does not help once the transaction is already 128 bits wide.

## 2. reduce -- the eight-rung classic, on 64 lanes

Block-wise sum. Each rung consumes a different number of elements per block (the
CUDA files change their grid as the optimization progresses), so the comparison
is bytes read per second, as in the original README.

| rung | | N=32M | speedup |
|---|---|---|---|
| `v0_baseline` | LDS tree, `tid % (2s) == 0` -- divergent | 1704 GB/s | 1.00x |
| `v1_no_divergence` | `index = 2*s*tid`, active lanes packed | 1739 GB/s | 1.02x |
| `v2_no_bank_conflict` | `s` halving, `tid < s` | 1928 GB/s | 1.13x |
| `v3_add_during_load` | fold 2 globals before the tree | 3637 GB/s | 2.13x |
| `v4_unroll_last_wave` | last 64 lanes finish in registers | 4001 GB/s | 2.35x |
| `v5_full_unroll` | LDS levels emitted straight-line | 4021 GB/s | 2.36x |
| `v6_multi_add` | serial accumulate, then the tree | 6861 GB/s | 4.03x |
| `v7_shuffle` | serial accumulate + wave shuffle | **6940 GB/s** | **4.07x** |
| `v8_shuffle_vec4` | v7 with dwordx4 loads | 6862 GB/s | 4.03x |
| `v9_vec4_wide_grid` | v8 on 32 blocks/CU | 6825 GB/s | 4.01x |

**6.94 TB/s is 87% of the 8 TB/s HBM peak, and 1.33x PyTorch's segmented sum.**

Two rungs deviate from a literal transcription. `v4`/`v5` "unroll the last warp"
becomes "the last *wavefront*" -- 64 lanes, one level deeper -- and since FlyDSL has
no `volatile` memref the barrier-free tail runs in registers through shuffles
rather than through LDS. `v8` and `v9` are additions that **did not work**: once a
thread accumulates serially the kernel is at the memory roof, and neither a wider
transaction nor a bigger grid moves it.

## 3. sgemv -- shaping rows onto the wavefront

`y = A x`. One FMA per 4 bytes, so memory-bound at every size; the whole game is
mapping rows onto lanes without idling any.

| shape | best rung | GB/s | vs rocBLAS |
|---|---|---|---|
| M=16384, N=16 | `v2_subwave_per_row` (4 rows per wavefront) | 154 | **1.81x** |
| M=16384, N=64 | `v0_wave_per_row` | 588 | **1.80x** |
| M=16384, N=256 | `v0_wave_per_row` | 2298 | **1.89x** |
| M=16384, N=4096 | `v3_block_per_row` | 7419 | **1.56x** |
| M=1024, N=16384 | `v3_block_per_row` | 6237 | **1.95x** |

Every shape beats rocBLAS's `sgemv`, and the long-row case reaches **7.4 TB/s --
93% of HBM peak**. The CUDA `N == 32` case becomes `N == 64` here, and the
"pack several rows per warp" case divides 64 rather than 32, so N=16 puts **four**
rows in a wavefront where CUDA put two.

## 4. sgemm -- two levels of blocking, then the matrix cores

`C = A B` in f32.

| rung | | 1024^3 | 2048^3 | 4096^3 |
|---|---|---|---|---|
| `v0_naive` | thread per C element, all from global | 8.6 | 8.7 | 8.1 |
| `v1_lds_tile` | 16x16x16 LDS tile | 14.2 | 15.1 | 13.8 |
| `v2_thread_tile` | 128x128x8, 8x8 per thread, float4 | 10.5 | 42.1 | 68.7 |
| `v3_prefetch` | + next-tile global prefetch | 12.0 | 47.6 | 68.8 |
| `v4_double_buffer` | + LDS ping-pong, one barrier/tile | 11.5 | 45.7 | 68.8 |
| `v5_mfma` | same blocking, `v_mfma_f32_16x16x4_f32` | **16.9** | **66.5** | **113.8** |
| *rocBLAS* | | *95.3* | *120.8* | *135.2* |

TFLOP/s. At 4096^3 the ladder spans **14x** end to end and finishes at **84% of
rocBLAS**.

Three things worth reading off this table:

- **v2-v4 land within 1% of each other at 4096^3** -- about **97% of the FP32
  vector peak** (~69.5 of ~72 TFLOP/s). Prefetching and double-buffering are not
  wrong; there is nothing left for them to hide.
- **The only move left is the one the hardware asks for.** The CUDA repo's final
  chapter is SASS-level register remapping with CuAssembler to squeeze the vector
  FMA pipe. CDNA's answer is a different functional unit: FP32 MFMA retires 256
  FLOP/clock/CU against the vector pipe's 128. `v5` takes it and gets 1.65x the
  best vector rung.
- **At 1024^3 the register-tiled kernel loses to the naive LDS tile.** A 128x128
  tile gives an 8x8 = 64-block grid on a 256-CU part. The V100-tuned tile shape
  is not wrong; it is sized for a machine with 80 SMs.

## 5. spmv -- the lanes-per-row knob

`y = A_csr x`. The CUDA original is one kernel with one knob,
`THREADS_PER_VECTOR`; the ladder is that knob swept. On CDNA the useful settings
are divisors of 64.

| lanes/row | 32 nnz/row | 8 nnz/row | skewed (power-law) |
|---|---|---|---|
| 1 | 50.9 | 160.7 | 5.1 |
| 4 | 142.7 | 308.2 | 13.1 |
| 8 | 255.0 | **312.7** | 18.9 |
| 16 | **308.7** | 246.8 | 29.8 |
| 64 (full wave) | 235.3 | 84.7 | **70.3** |

GFLOP/s. The knob has a real optimum and it tracks the average row length -- and
on a power-law matrix a **full wavefront per row is 13.8x the scalar kernel**,
because the wavefront is what absorbs the load imbalance. All three shapes beat
PyTorch's CSR `mv` (up to 48x; it is not a tuned path).

## 6. spmm -- reusing the sparse row across dense columns

`C = A_csr B`, dense `B`.

| rung | | n=256 | n=1024 | skewed, n=256 |
|---|---|---|---|---|
| `v0_scalar` | 1 output column per thread | 2418 | 2569 | 82.6 |
| `v1_lds_row` | stage the row's (col,val) in LDS | 1754 | 1849 | **376.5** |
| `v2_vec4` | 4 output columns per thread, float4 B | **4182** | **6202** | 93.6 |

GFLOP/s. `v1` is the kernel the CUDA source itself labels `// useless optimize`.
**On a uniform matrix that verdict holds exactly** (0.72x, pure overhead). **On a
power-law matrix the same kernel is 4.6x faster than v0** -- and the win is not the
LDS, it is the `CHUNK`-sized outer loop the staging forces, which turns one
thread's unbounded row walk into barrier-synchronised passes the workgroup
advances through together. The original's verdict was measured on balanced
matrices.

---

## Layout

```
flyopt/
  env.py            hardware detection + the peak figures everything is measured against
  dsl.py            shared device-side helpers (copy atoms, shuffles, MFMA, fast launch)
  registry.py       the Op / Variant model the bench and tests both drive
  elementwise.py reduce.py sgemv.py sgemm.py spmv.py spmm.py
  sparse_common.py  reproducible CSR generation (the original's matrices are not in-repo)
bench/              `python -m bench` -- verify, then time, then report
tests/              correctness for every rung at its smallest shape
docs/porting-notes.md   the CUDA->CDNA translation table, the FlyDSL sharp edges,
                        and the results that contradicted expectation
```

Each kernel module's docstring says what it ports, what changed, and why.
`docs/porting-notes.md` is the one to read before writing a new FlyDSL kernel --
in particular Sec. 2.2 (host dispatch masquerading as a bandwidth roof), Sec. 2.3
(shuffles inside `scf.if` are silently wrong), and Sec. 2.7 (`a*b + c` is not an FMA
and costs 6x).

## Caveats

- **Unofficial.** See the disclaimer at the top: personal measurements, not
  AMD-verified or AMD-endorsed performance data.
- Measured on an **MI350X VF** (a virtualised partition), one GPU, ROCm 7.2.
  Efficiency percentages use the MI350X clock (2.2 GHz); an MI355X would move the
  denominators up ~9%.
- Everything is f32, to stay comparable with the CUDA original. On CDNA4 that is
  the *worst* datatype in relative terms -- the matrix cores do 2.5 PF of FP16 and
  5 PF of FP8 against 157 TF of FP32. Nothing here is a statement about what the
  hardware can do.
- `torch`/rocBLAS columns are what the installed PyTorch dispatches to, not a
  hand-tuned rocBLAS call. The sparse ones in particular are not tuned paths.

## Credits

This repository exists because of
**[Liu-xiandong/How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU)**
by **Xiandong Liu** (`xiandong_liu@foxmail.com`), released under Apache 2.0.

Everything structural here is theirs: the choice of kernels, the idea of teaching
each one as a ladder where every file adds exactly one optimization, the specific
sequence of ideas in each ladder (divergence -> bank conflicts -> add-during-load ->
unrolling -> shuffles; global->LDS->register blocking; the lanes-per-row knob), and
the discipline of reporting every rung against the vendor library rather than
against the previous rung alone. The V100 measurements those ladders were derived
from are in the original README and are worth reading alongside these.

This port contributes the translation to a different machine -- 64-lane
wavefronts, 256 CUs, matrix cores, FlyDSL instead of CUDA C -- plus the CDNA4
measurements, the rungs marked as CDNA additions, and the negative results. Where
a conclusion of the original does not survive the move (the `spmm` "useless
optimize" comment, the V100-sized GEMM tile), that is noted as a difference in
hardware, not a correction: each verdict is right for the machine it was measured
on.

The CDNA4 hardware facts this port budgets against (LDS capacity, HBM bandwidth,
matrix-core throughput -- all reproduced in `flyopt/env.py`) come from AMD's
*Introducing AMD CDNA 4 Architecture* whitepaper and the CDNA4 machine-readable
ISA published at [gpuopen.com](https://gpuopen.com/download/machine-readable-isa/).

## License

**Apache License 2.0** -- the same license as the original repository this ports,
whose `LICENSE` file is carried over here unchanged. See [`LICENSE`](LICENSE) for
the full text and [`NOTICE`](NOTICE) for the derivative-work attribution required
by section 4(d): upstream copyright Xiandong Liu, ported work copyright the
contributors to this repository.
