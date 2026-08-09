# spmm -- reusing the sparse row across dense columns

Ports `spmm/spmm.cu`.

Its two kernels answer one question: given that a CSR row must be walked
serially, what does a thread block do with the fact that *every* output column of
that row walks the **same** row?

## Rungs

| rung | what it does |
|---|---|
| `v0_scalar` | one block per (row, column-chunk), one output column per thread |
| `v1_lds_row` | stage the row's `(col_index, value)` pairs in LDS first |
| `v2_vec4` | four output columns per thread, `buffer_load_dwordx4` on B |

`v0` is the CUDA `My_spmm_csr_vector_kernel_v0`. The whole block reads the same
`(col_index, value)` pair on every step, which looks wasteful and mostly is not
-- the pair is broadcast out of L1.

`v2` has no CUDA counterpart and is where the real win is: the sparse row walk is
unchanged, only the dense side gets wider, and the dense side is all of the
traffic.

## The "useless optimize" comment

`v1` is the kernel the CUDA source itself labels `// useless optimize`. The
measurement splits that verdict in two.

**On a uniform matrix the verdict holds exactly**: 0.72x, pure overhead, just as
the comment predicts.

**On a power-law matrix the same kernel is 4.6x faster than `v0`.** The reason is
not the LDS at all -- it is the `CHUNK`-sized outer loop that staging forces,
which converts one thread's unboundedly long row walk into a sequence of
barrier-synchronised passes that the whole workgroup advances through together.
"Useless" was measured on balanced matrices; it does not survive contact with a
real graph.

## Results

> Measured on an AMD Instinct MI350X VF (gfx950, wave64), 256 CU @ 2.2 GHz,
> ROCm 7.2, FlyDSL 0.2.4. Unofficial -- see the disclaimer in the top-level README.

GFLOP/s:

| rung | n=256 | n=1024 | skewed, n=256 |
|---|---|---|---|
| `v0_scalar` | 2418 | 2569 | 82.6 |
| `v1_lds_row` | 1754 | 1849 | **376.5** |
| `v2_vec4` | **4182** | **6202** | 93.6 |
| *torch CSR* | *527* | *682* | *179* |
