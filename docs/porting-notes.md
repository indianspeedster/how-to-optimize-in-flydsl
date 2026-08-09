# Porting notes: CUDA -> FlyDSL on CDNA4

Everything in this file was learned by making a kernel work (or fail) on an
MI350X against FlyDSL 0.2.4. It is split into three parts: the hardware
translation (CUDA concept -> CDNA concept), the FlyDSL sharp edges (things that
cost real debugging time), and the results that contradicted expectation.

All measurements quoted here are my own and unofficial -- not produced, reviewed,
or endorsed by AMD. See the disclaimer in `README.md`.

---

## 1. The hardware translation

| CUDA / V100 | CDNA4 / MI350X | Consequence for the port |
|---|---|---|
| warp = **32** lanes | wavefront = **64** lanes | Every "warp-per-row", "last warp", "shuffle ladder" constant doubles. `sgemv` v0 becomes the N=64 case, not N=32. The reduce tail covers one more level. |
| `__shfl_down_sync(mask, v, d)` | `gpu.shuffle` mode `down`, width 64 | No mask operand -- the segment width *is* the mask. Sub-wave widths (2...32) are legal and are how several rows share one wavefront. |
| `__syncthreads()` | `gpu.barrier()` (`s_barrier`) | Same semantics. |
| `volatile float* cache` warp-synchronous LDS | *no equivalent* | See Sec. 2.3 -- the barrier-free tail is done in registers via shuffles instead. |
| `float4` / `reinterpret_cast` | `BufferCopy128b` copy atom -> `buffer_load_dwordx4` | Same 128-bit transaction, named as an atom rather than a cast. |
| plain pointer loads | *buffer* tensors (`rocdl.make_buffer_tensor`) | Carries a resource descriptor, so accesses are hardware bounds-checked. This is why the GEMM prefetch can read one tile past the end without faulting -- **within a row**; see Sec. 2.5 for where that guarantee stops. |
| `__shared__` array | `@fx.struct` + `fx.SharedAllocator` | 160 KB per CU on gfx950 vs 96 KB on V100 -- LDS capacity is essentially never the binding constraint in this repo. |
| SASS tuning with CuAssembler | matrix cores (`v_mfma_*`) | The CUDA repo's last chapter squeezes the vector FMA schedule. On CDNA the same problem is answered by a different functional unit -- see `sgemm` v5. |
| 80 SMs (V100) | **256 CUs** | Grid sizes tuned for a V100 leave a MI350X three-quarters idle. The reduce ladder's fixed 1024-block grid and the GEMM's 128x128 tile at M=N=1024 both hit this. |

### Peak numbers this repo measures against

From AMD's *Introducing AMD CDNA 4 Architecture* whitepaper (via the `gfx950`
skill's `microarch.md`, which is not vendored in this repo), reproduced in
`common/env.py`:

- HBM3E **8 TB/s**, LDS **160 KB/CU** at 256 B/clock
- FP32 **vector** ~ 72 TFLOP/s at MI350X clocks
- FP32 **matrix** (MFMA) ~ 144 TFLOP/s -- 2x the vector pipe
- FP16/BF16 matrix 2.5 PF, FP8 5 PF, MXFP4 10 PF

That last line is the context for the whole `sgemm` chapter: f32 is the *worst*
datatype on this silicon in relative terms, and the ladder is held at f32 only so
it stays comparable with the CUDA original.

---

## 2. FlyDSL sharp edges

These cost time. They are recorded in the order a new kernel is likely to hit
them.

### 2.1 `from __future__ import annotations` breaks LDS

```
TypeError: Cannot compute layout for schema SharedStorage: ...
```

`@fx.struct` resolves its field annotations eagerly to build the LDS layout. PEP
563 turns them into strings and the layout can no longer be computed. **Any
module that declares an `@fx.struct` must not enable postponed annotations.**
Every kernel module here carries a comment saying so.

### 2.2 Host dispatch, not bandwidth, is the floor for short kernels

The single most misleading measurement in this port. A bare call to a
`@flyc.jit` launcher re-runs `inspect.Signature.bind`, protocol introspection
and DLPack resolution *every time* -- tens of microseconds of host work per
launch. Every reduce variant measured **32.0 us +/- 0.3** regardless of grid size,
transaction width, or accumulator count. It looked exactly like a hard bandwidth
roof. It was the host.

`flyc.compile(launch_fn, *args)` pre-resolves all of it and returns a callable
that only updates ctypes slots. Wrapping every launcher in
`dsl.fast_launcher` moved the reduce ladder from 4.2 TB/s to **6.9 TB/s** with
no change to any kernel.

Two details: the compiled dispatcher is positional and *not* variadic, so the
stream must be passed explicitly (`torch.cuda.current_stream()`), and
`flyc.compile` **executes the kernel once** while warming up -- fine for the
write-the-whole-output kernels here, but an atomic-accumulating epilogue would
need a scratch output for the warm-up.

### 2.3 Cross-lane shuffles do not survive an `scf.if`

Writing the CUDA idiom directly:

```python
if tid < 64:                       # -> scf.if region
    v = fx.memref_load(s_data, tid)
    v = wave_reduce_sum_down(v, 64)
```

compiles and produces **wrong results** (reduce v4/v5/v6 failed with
`max_err=1041`). Lanes outside the region contribute undefined values to the
shuffle. The fix is to predicate rather than branch:

```python
live = tid < 64
v = fx.memref_load(s_data, live.select(tid, fx.Int32(0)))
v = wave_reduce_sum_down(live.select(v, fx.Float32(0.0)), 64)
```

Every lane executes the shuffle; the out-of-range ones carry the additive
identity. Same shape of fix applies wherever a cross-lane op sits under a guard.

### 2.4 Helpers in another module may not contain data-dependent control flow

The AST rewriter only transforms the `@flyc.kernel` function's own AST. A helper
imported from elsewhere runs as plain Python during tracing, so an `if` on a
runtime value is evaluated by CPython -- silently taking one branch at compile
time. `common/dsl.py` is therefore entirely branch-free; `block_reduce_sum`
replaces CUDA's `if (lane == 0) store` with a `select` into a write-only sink
slot (racy, never read) precisely for this reason.

Compile-time `if`s inside a kernel should be wrapped in `const_expr(...)` so the
rewriter folds them instead of trying to lower them.

Relatedly: **import FlyDSL symbols at module level**, never inside the function
that builds a kernel -- the rewriter counts free variables and an inner import
breaks the rewrite.

### 2.5 Buffer bounds checking is per-descriptor, not per-row

The GEMM prefetch issues the load for tile `k+1` unconditionally, including past
the last tile. Reading past the end of a *row* of A is harmless (it lands in the
next row, still inside the tensor's descriptor), but slicing a **row index**
beyond `K` faults:

```
AcceleratorError: CUDA error: an illegal memory access
```

The fix is a branch-free clamp, `(kt < n_tiles).select(kt, n_tiles - 1)`, which
re-reads the last tile into registers that are never accumulated. Note the
clamp must accept both a Python `int` (the pre-loop call) and an SSA value (the
in-loop call) -- `fx.Int32(x)` normalises both.

### 2.6 Objects that cannot cross a runtime loop boundary

`scf.for` turns every live variable into a loop-carried argument. Layout-algebra
objects (`ThrCopy`, `TiledCopy`) cannot be reconstructed from one and the loop
build fails with `failed to construct <class ThrCopy>`. Build them **inside** the
loop body -- they are compile-time layout objects, so this costs nothing.

Accumulators have the opposite problem: a Python list of 64 SSA values would
have to be threaded through the loop. Use `fx.make_rmem_tensor` instead -- an
alloca that LLVM promotes to registers (measured: 162 VGPRs, no scratch, 0 AGPR
traffic for the 8x8 GEMM tile).

### 2.7 `a * b + c` is not an FMA

IEEE f32 forbids contracting a multiply and an add without an explicit licence,
and FlyDSL does not grant one by default. The GEMM inner loop compiled to 252
`v_pk_mul_f32` + 256 `v_pk_add_f32`, ran at **1.7 TFLOP/s**, and spilled 388
bytes to scratch. Replacing the expression with `math.fma(a, b, c)` produced 256
`v_pk_fma_f32`, no scratch, and **10.6 TFLOP/s** -- a 6x speedup from one call.

Check for this in any generated kernel by grepping the ISA dump:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dump python ...
grep -c v_pk_fma_f32 dump/kernel_0/21_final_isa.s      # want: the FMA count
grep private_seg_size dump/kernel_0/21_final_isa.s     # want: 0
```

### 2.8 The MFMA lane layout, confirmed not recalled

`v_mfma_f32_16x16x4_f32` on a 64-lane wavefront:

| operand | lane `l` holds |
|---|---|
| A (16x4) | `A[l % 16][l / 16]` |
| B (4x16) | `B[l / 16][l % 16]` |
| D/C (16x16), 4 VGPRs | `D[4*(l/16) + r][l % 16]`, `r in 0..3` |

This was validated with a 16x16x4 single-wavefront kernel against `A @ B` before
any GEMM was built on it (`max err 0.0`). Doing that first is worth the ten
minutes: a wrong layout inside a tiled GEMM is nearly undebuggable.

The alternative -- FlyDSL's `make_tiled_mma` / `make_fragment_A` layout API, as in
`examples/03-tiledMma.py` -- works on global-memory operands but crashed the MLIR
cast machinery when its operands were views over LDS. The raw intrinsic with
hand-written lane indices was both faster to get right and fully under control.

---

## 3. Results that contradicted expectation

Kept because a ladder that only shows the rungs that worked is not a ladder.

**Wider transactions stop mattering once a kernel is latency-bound.** `reduce`
v8 (dwordx4) and v9 (32 blocks/CU instead of 1024 total) were both added
expecting a win over v7. Neither moved the number: once each thread accumulates
serially, the kernel is at the memory roof and neither the transaction width nor
the grid changes that. Same story for `elementwise` v3 -- four independent
float4s per lane was *slower* than one (5.44 vs 5.86 TB/s).

**Register tiling can lose to a naive tile when the grid is too small.** At
M=N=K=1024 the 128x128 GEMM tile produces a 8x8 = **64-block** grid on a 256-CU
part, and the 16x16 tile (4096 blocks) beats it -- 14.3 vs 10.5 TFLOP/s. The
ordering inverts at 4096^3 (13.9 vs 68.8). The V100-tuned tile shape is not wrong;
it is sized for a machine with 80 SMs.

**The CUDA author's "useless optimize" is only half right.** `spmm` v1 stages
the sparse row into LDS and its CUDA source comment calls it useless. On a
uniform matrix that verdict holds exactly (0.72x -- it is pure overhead). On a
power-law matrix the same kernel is **4.6x faster** than v0. The win is not the
LDS: it is the `CHUNK`-sized outer loop that staging forces, which turns one
thread's unboundedly long row walk into barrier-synchronised passes the whole
workgroup advances through together. The original's verdict was measured on
balanced matrices.

**The vector-FMA GEMM ladder ends at the vector roof, exactly.** v2/v3/v4 all
land within 1% of each other at ~69.5 TFLOP/s at 4096^3 -- about 97% of the FP32
vector peak. Prefetching and LDS double-buffering are not *wrong*; there is
simply nothing left for them to hide. The only remaining move is the one the
hardware asks for, and `v5_mfma` takes it: **114 TFLOP/s**, 1.65x the best
vector rung and 84% of rocBLAS.
