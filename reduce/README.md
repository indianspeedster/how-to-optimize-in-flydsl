# reduce -- the classic ladder, on 64 lanes

Ports `reduce/reduce_v0_baseline.cu` .. `reduce_v7_shuffle.cu`.

Each block reduces one contiguous chunk of the input to a single float, so with
`E` elements per block the output has `N/E` entries -- exactly the original's
`d_out[blockIdx.x]`. The rungs consume different `E` (256, then 512 once a thread
adds during load, then a large chunk once it accumulates serially), so they emit
different-length outputs while reading the same `N` floats. As in the original
README, the number that compares them is **bytes read per second**.

## Rungs

| file | what it adds |
|---|---|
| `reduce_v0_baseline.py` | LDS tree, `tid % (2s) == 0` -- divergent inside every wavefront |
| `reduce_v1_no_divergence.py` | `index = 2*s*tid` -- active lanes packed |
| `reduce_v2_no_bank_conflict.py` | `s` halving with `tid < s` -- conflict-free LDS |
| `reduce_v3_add_during_load.py` | fold two globals into one LDS slot |
| `reduce_v4_unroll_last_wave.py` | the last wavefront finishes in registers, no barrier |
| `reduce_v5_full_unroll.py` | LDS levels emitted straight-line at compile time |
| `reduce_v6_multi_add.py` | serial accumulation per thread, then the tree |
| `reduce_v7_shuffle.py` | serial accumulation, then a wave shuffle |
| `reduce_v8_shuffle_vec4.py` | v7 with dwordx4 loads |
| `reduce_v9_vec4_wide_grid.py` | v8 on 32 blocks per CU |

One file per rung, as in the original -- read them in order and each is a single
idea applied to the one before it. `_common.py` holds the constants and the LDS
storage struct they share; `__init__.py` is the ladder.

## Where this departs from the CUDA original

**`v4`/`v5` -- "unroll the last warp".** The CUDA trick is that a warp is 32
lanes and executes in lockstep, so the tail of the tree needs no
`__syncthreads()`, only `volatile`. A CDNA wavefront is **64** lanes, so the
barrier-free tail starts twice as early and covers one more level. FlyDSL has no
`volatile` memref, so the tail is done in registers through cross-lane shuffles
-- strictly better than the LDS round-trip it replaces, and what a CDNA
programmer would write. The rung still isolates the same effect: the tail of the
tree stops paying for workgroup barriers.

**`v7` -- "shuffle".** `__shfl_down_sync(0xffffffff, v, d)` becomes `gpu.shuffle`
in `down` mode over 64 lanes, so the ladder is 6 steps rather than 5 and the
cross-wave LDS array holds 4 partials for a 256-thread block instead of 8.

**`v8`/`v9` have no CUDA counterpart** and neither delivers a win -- see below.

## Results

> Measured on an AMD Instinct MI350X VF (gfx950, wave64), 256 CU @ 2.2 GHz,
> ROCm 7.2, FlyDSL 0.2.4. Unofficial -- see the disclaimer in the top-level README.

| rung | N=32M | speedup | N=256M |
|---|---|---|---|
| `v0_baseline` | 1704 GB/s | 1.00x | 1521 GB/s |
| `v1_no_divergence` | 1739 GB/s | 1.02x | 1539 GB/s |
| `v2_no_bank_conflict` | 1928 GB/s | 1.13x | 1671 GB/s |
| `v3_add_during_load` | 3637 GB/s | 2.13x | 3150 GB/s |
| `v4_unroll_last_wave` | 4001 GB/s | 2.35x | 3501 GB/s |
| `v5_full_unroll` | 4021 GB/s | 2.36x | 3504 GB/s |
| `v6_multi_add` | 6861 GB/s | 4.03x | 6273 GB/s |
| `v7_shuffle` | **6940 GB/s** | **4.07x** | **6303 GB/s** |
| `v8_shuffle_vec4` | 6862 GB/s | 4.03x | 6176 GB/s |
| `v9_vec4_wide_grid` | 6825 GB/s | 4.01x | 6214 GB/s |

**6.94 TB/s is 87% of the 8 TB/s HBM peak, and 1.33x PyTorch's segmented sum.**

The decisive rung is `v6`: giving each thread enough serial work that the tree
stops mattering is worth more (1.7x over `v5`) than every tree optimization
before it combined. After that the kernel is at the memory roof, which is why
`v8` and `v9` -- a wider transaction and a CDNA-sized grid -- move nothing. Both
are kept: a ladder that shows only the steps that worked is not a ladder.
