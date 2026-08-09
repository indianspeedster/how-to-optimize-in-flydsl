# sgemm -- two levels of blocking, then the matrix cores

Ports `sgemm/sgemm_v1.cu` and `sgemm/sgemm_v3.cu`.

The CUDA repo's thesis is that a fast SGEMM is a *blocking* problem solved twice:
once from global memory into shared memory, and again from shared memory into
registers, with the second level being what actually raises arithmetic
intensity. The ladder here makes each level explicit and then adds the step CDNA
asks for.

## Rungs

| file | what it adds |
|---|---|
| `sgemm_v0_naive.py` | nothing -- one thread per C element, all from global |
| `sgemm_v1_lds_tile.py` | blocking level 1: a 16x16x16 LDS tile |
| `sgemm_v2_thread_tile.py` | blocking level 2: 128x128x8 tile, 8x8 per thread |
| `sgemm_v3_prefetch.py` | next tile's global loads issue before this tile's math |
| `sgemm_v4_double_buffer.py` | LDS ping-pong -- one barrier per K-tile instead of two |
| `sgemm_v5_mfma.py` | the same blocking, run on the matrix cores |

v2, v3 and v4 are deliberately near-identical files that differ only in their K
loop -- exactly as `sgemm_v1.cu` and `sgemm_v3.cu` are in the original. Diff them.

The v2/v3/v4 shapes come straight from the original and are not
arbitrary: 256 threads x 8x8 gives every thread 64 accumulators -- enough
register pressure to hide LDS latency, not enough to spill (162 VGPRs measured,
no scratch). A is transposed on the way into LDS so the inner loop reads
contiguous floats, one `ds_read_b128` per four instead of four `ds_read_b32`.

## Where this departs from the CUDA original

The original's last chapter is SASS-level register remapping with CuAssembler to
squeeze the vector FMA pipe. CDNA's answer to that problem is not a better FMA
schedule, it is a **different functional unit**: `v_mfma_f32_16x16x4_f32` retires
256 FLOP/clock/CU against the vector pipe's 128. `v5` takes it.

The operand layout for that instruction on a 64-lane wavefront (confirmed
empirically against `A @ B` before any GEMM was built on it, not recalled):

| operand | lane `l` holds |
|---|---|
| A (16x4) | `A[l % 16][l / 16]` |
| B (4x16) | `B[l / 16][l % 16]` |
| D/C (16x16), 4 VGPRs | `D[4*(l/16) + r][l % 16]`, `r in 0..3` |

## Results

> Measured on an AMD Instinct MI350X VF (gfx950, wave64), 256 CU @ 2.2 GHz,
> ROCm 7.2, FlyDSL 0.2.4. Unofficial -- see the disclaimer in the top-level README.

TFLOP/s:

| rung | 1024^3 | 2048^3 | 4096^3 |
|---|---|---|---|
| `v0_naive` | 8.6 | 8.7 | 8.1 |
| `v1_lds_tile` | 14.2 | 15.1 | 13.8 |
| `v2_thread_tile` | 10.5 | 42.1 | 68.7 |
| `v3_prefetch` | 12.0 | 47.6 | 68.8 |
| `v4_double_buffer` | 11.5 | 45.7 | 68.8 |
| `v5_mfma` | **16.9** | **66.5** | **113.8** |
| *rocBLAS* | *95.3* | *120.8* | *135.2* |

At 4096^3 the ladder spans **14x** end to end and finishes at **84% of
rocBLAS**.

Three things worth reading off that table:

- **`v2`-`v4` land within 1% of each other at 4096^3** -- about **97% of the FP32
  vector peak** (~69.5 of ~72 TFLOP/s). Prefetching and LDS double-buffering are
  not wrong; there is nothing left for them to hide. That is what makes `v5` the
  only move remaining.
- **At 1024^3 the register-tiled kernel loses to the naive LDS tile.** A 128x128
  tile gives an 8x8 = **64-block** grid on a 256-CU part, so three quarters of the
  machine idles. The V100-tuned tile shape is not wrong; it is sized for a
  machine with 80 SMs.
- **f32 is the worst datatype on this silicon in relative terms.** The matrix
  cores do 2.5 PFLOP/s of FP16 and 5 PFLOP/s of FP8 against 157 TFLOP/s of FP32.
  The ladder is held at f32 only so it stays comparable with the CUDA original;
  nothing here is a statement about what the hardware can do.
