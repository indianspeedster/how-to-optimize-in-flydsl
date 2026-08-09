# elementwise -- how wide is one lane's transaction?

Ports `elementwise/elementwise_add.cu`.

`C = A + B` reads 2N and writes N floats and does one add per 12 bytes, so it is
pure HBM traffic and the only metric that means anything is achieved bandwidth.

The original file's whole point is a single axis: how wide is one lane's memory
transaction? CUDA expresses it as `float` / `float2` / `float4` reinterpret
casts. FlyDSL expresses it as the copy *atom* -- `BufferCopy32b` / `64b` /
`128b` -- which lowers to `buffer_load_dword` / `dwordx2` / `dwordx4`. Same axis,
named honestly.

## Rungs

| file | rung | what it adds |
|---|---|---|
| `kernels.py` | `v0_float` | one f32 per lane |
| | `v1_float2` | two f32 per lane |
| | `v2_float4` | four f32 per lane |
| | `v3_float4_x4` | four independent float4s per lane |

All four are the same builder with a different transaction width, so they share
one file: separating them would hide the fact that the *only* difference is a
constant.

## Results

> Measured on an AMD Instinct MI350X VF (gfx950, wave64), 256 CU @ 2.2 GHz,
> ROCm 7.2, FlyDSL 0.2.4. Unofficial -- see the disclaimer in the top-level README.

| rung | N=32M | N=256M | vs rocBLAS |
|---|---|---|---|
| `v0_float` | 5283 GB/s | 5127 GB/s | 0.88x |
| `v1_float2` | 5649 GB/s | 5381 GB/s | 0.92x |
| `v2_float4` | **5923 GB/s** | **5869 GB/s** | **1.00x** |
| `v3_float4_x4` | 5574 GB/s | 5597 GB/s | 0.94x |

Same conclusion as the original: wider is better, and `float4` matches the vendor
library exactly (5923 vs 5924 GB/s -- both are at the same roof).

`v3` is a **negative result kept on purpose**. It was added expecting that a
256-CU part would need more memory-level parallelism than one dwordx4 per lane
supplies. It does not: four independent float4s per lane is *slower* than one.
Once the transaction is 128 bits wide the kernel is at the roof and extra loads
in flight only cost registers.
