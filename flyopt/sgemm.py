# SPDX-License-Identifier: Apache-2.0
"""SGEMM (C = A B, f32) -- the register-tiling ladder, plus the matrix cores.

Ports ``sgemm/sgemm_v1.cu`` and ``sgemm/sgemm_v3.cu``. The CUDA repo's thesis is
that a fast SGEMM is a *blocking* problem solved twice: once from global memory
into shared memory, and again from shared memory into registers, with the second
level being what actually raises arithmetic intensity. The ladder here makes each
level explicit:

    v0  one thread per C element, everything from global      (no blocking)
    v1  16x16x16 LDS tile, still one C element per thread      (blocking level 1)
    v2  128x128x8 LDS tile, 8x8 C elements per thread          (blocking level 2)
    v3  v2 + prefetch: next tile's global loads issue before   (latency hiding)
        this tile's math, so VMEM and VALU overlap
    v4  the same blocking, but the inner product runs on the
        matrix cores (MFMA) instead of the vector FMA units

``v4`` is where the port stops being a translation. The CUDA repo's last step is
SASS-level register remapping with CuAssembler to squeeze the vector FMA pipe;
CDNA's answer to that problem is not a better FMA schedule, it is a different
functional unit. ``v_mfma_f32_16x16x4_f32`` retires 256 FLOP/clock/CU against the
vector pipe's 128, so the matrix-core path starts with a 2x ceiling advantage
before any scheduling work at all.

A note on the arithmetic: f32 is the *worst* datatype on CDNA4 in relative terms.
The matrix cores do 2.5 PFLOP/s of FP16 and 5 PFLOP/s of FP8 but only 157 TFLOP/s
of FP32 -- a 32x spread. Nothing here is a statement about what the hardware can
do; it is a statement about what this particular ladder does, held at f32 so it
stays comparable to the CUDA original.
"""

# No `from __future__ import annotations` -- see flyopt/reduce.py.

import torch

from flyopt.dsl import (
    HAVE_FLYDSL,
    const_expr,
    fast_launcher,
    flyc,
    fma,
    fx,
    gpu,
    load_scalar,
    load_vec,
    mfma_f32_16x16x4_f32,
    range_constexpr,
    store_scalar,
    store_vec,
    vec_copy_atom,
    vec_divide,
)
from flyopt.registry import Op, Shape, Variant, register

# ── v0: no blocking ─────────────────────────────────────────────────────────

TS = 16   # v0/v1 use a 16x16 thread block


def _build_naive():
    """One thread per C element; every operand comes from global memory.

    K loads of A and K of B per output element, and nothing is reused, so the
    kernel runs at the L2/HBM roof and the matrix cores idle entirely.
    """

    def build(M: int, N: int, K: int):
        @flyc.kernel
        def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            tx, ty = tid % TS, tid // TS
            m = fx.block_idx.y * TS + ty
            n = fx.block_idx.x * TS + tx

            atom = vec_copy_atom(1)
            A_buf = fx.rocdl.make_buffer_tensor(A)
            B_buf = fx.rocdl.make_buffer_tensor(B)
            C_buf = fx.rocdl.make_buffer_tensor(C)
            a_row = vec_divide(fx.slice(A_buf, (m, None)), 1)
            c_row = vec_divide(fx.slice(C_buf, (m, None)), 1)

            acc = fx.Float32(0.0)
            for k in range(K):
                b_row = vec_divide(fx.slice(B_buf, (k, None)), 1)
                acc = acc + load_scalar(a_row, k, atom) * load_scalar(b_row, n, atom)
            store_scalar(acc, c_row, n, atom)

        @flyc.jit
        def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, B, C).launch(grid=(N // TS, M // TS, 1), block=(TS * TS, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build


# ── v1: one level of blocking (global -> LDS) ───────────────────────────────


def _build_lds_tile():
    """16x16x16 LDS tile, one C element per thread.

    Each A and B element loaded from global is now used 16 times instead of
    once. That is the entire first-level blocking argument, and it is worth
    roughly an order of magnitude here.
    """

    def build(M: int, N: int, K: int):
        if K % TS:
            raise ValueError(f"K={K} not a multiple of {TS}")

        @fx.struct
        class Shared:
            a: fx.Array[fx.Float32, TS * TS, 16]
            b: fx.Array[fx.Float32, TS * TS, 16]

        @flyc.kernel
        def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            tx, ty = tid % TS, tid // TS
            m = fx.block_idx.y * TS + ty
            n = fx.block_idx.x * TS + tx

            lds = fx.SharedAllocator().allocate(Shared).peek()
            As = lds.a.view(fx.make_layout((TS, TS), (TS, 1)))
            Bs = lds.b.view(fx.make_layout((TS, TS), (TS, 1)))

            atom = vec_copy_atom(1)
            A_buf = fx.rocdl.make_buffer_tensor(A)
            B_buf = fx.rocdl.make_buffer_tensor(B)
            C_buf = fx.rocdl.make_buffer_tensor(C)
            a_row = vec_divide(fx.slice(A_buf, (m, None)), 1)
            c_row = vec_divide(fx.slice(C_buf, (m, None)), 1)

            acc = fx.Float32(0.0)
            for kt in range(K // TS):
                b_row = vec_divide(fx.slice(B_buf, (kt * TS + ty, None)), 1)
                fx.memref_store(load_scalar(a_row, kt * TS + tx, atom), As, (ty, tx))
                fx.memref_store(load_scalar(b_row, n, atom), Bs, (ty, tx))
                gpu.barrier()
                for k in range_constexpr(TS):
                    acc = acc + fx.memref_load(As, (ty, k)) * fx.memref_load(Bs, (k, tx))
                gpu.barrier()
            store_scalar(acc, c_row, n, atom)

        @flyc.jit
        def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, B, C).launch(grid=(N // TS, M // TS, 1), block=(TS * TS, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build


# ── v2 / v3 / v4: two levels of blocking, TMxTN per thread ──────────────────

# The default geometry is the CUDA original's: 128x128x8 block tile, 8x8 per
# thread, 256 threads. `_TUNED` is what the sweep in docs/porting-notes.md
# picked for CDNA4 -- a deeper K tile, which buys fewer barriers per FLOP.
BM, BN, BK = 128, 128, 8
TM, TN = 8, 8
THREADS = (BM // TM) * (BN // TN)      # 256


def _build_thread_tile(prefetch: bool, *, bm=BM, bn=BN, bk=BK, tm=TM, tn=TN,
                       lds_stages: int = 1):
    """``bm x bn x bk`` block tile, ``tm x tn`` register tile per thread.

    The default shapes come straight from the CUDA original and they are not
    arbitrary:

    * 256 threads x 8x8 = the 128x128 block tile, so every thread holds 64
      accumulators -- enough register pressure to hide LDS latency, not enough
      to spill (162 VGPRs measured, 3 waves/SIMD).
    * A is transposed on the way into LDS (``As[k][m]``) so the inner loop reads
      ``tm`` *contiguous* floats per operand -- one ``ds_read_b128`` per four,
      not one ``ds_read_b32`` per element.

    Three latency-hiding levels are selectable:

    ``prefetch=False, lds_stages=1``
        global -> LDS -> barrier -> math -> barrier. Two barriers per K-tile and
        the global load latency is fully exposed.  (v2, = ``sgemm_v1.cu``)
    ``prefetch=True, lds_stages=1``
        the next tile's global loads are issued *before* this tile's math, so
        VMEM retires under the FMAs. Still two barriers. (v3)
    ``prefetch=True, lds_stages=2``
        plus an LDS ping-pong: this tile's math reads buffer p while the next
        tile's data is written to buffer 1-p, so only **one** barrier per K-tile
        is needed. This is the full double-buffering of ``sgemm_v3.cu``. (v4)
    """
    threads = (bm // tm) * (bn // tn)
    # Global -> LDS partition. Every thread moves float4s; `passes` is how many
    # each thread needs to cover the tile.
    a_thr_per_row = bk // 4            # threads spanning one A row (along K)
    b_thr_per_row = bn // 4            # threads spanning one B row (along N)
    a_rows_per_pass = threads // a_thr_per_row
    b_rows_per_pass = threads // b_thr_per_row
    if (bk % 4 or bn % 4 or bm % a_rows_per_pass or bk % b_rows_per_pass
            or tm % 4 or tn % 4 or threads % a_thr_per_row or threads % b_thr_per_row):
        raise ValueError(f"unsupported geometry {bm}x{bn}x{bk} / {tm}x{tn}")
    a_passes = bm // a_rows_per_pass
    b_passes = bk // b_rows_per_pass

    def build(M: int, N: int, K: int):
        step = bk * lds_stages
        if M % bm or N % bn or K % step:
            raise ValueError(f"({M},{N},{K}) not a multiple of ({bm},{bn},{step})")
        n_tiles = K // bk

        @fx.struct
        class Shared:
            a0: fx.Array[fx.Float32, bk * bm, 16]   # As[k][m] -- transposed
            b0: fx.Array[fx.Float32, bk * bn, 16]   # Bs[k][n]
            if lds_stages == 2:
                a1: fx.Array[fx.Float32, bk * bm, 16]
                b1: fx.Array[fx.Float32, bk * bn, 16]

        @flyc.kernel
        def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            bx, by = fx.block_idx.x, fx.block_idx.y
            tx, ty = tid % (bn // tn), tid // (bn // tn)

            lds = fx.SharedAllocator().allocate(Shared).peek()
            a_lay = fx.make_layout((bk, bm), (bm, 1))
            b_lay = fx.make_layout((bk, bn), (bn, 1))
            As = [lds.a0.view(a_lay)]
            Bs = [lds.b0.view(b_lay)]
            if const_expr(lds_stages == 2):
                As.append(lds.a1.view(a_lay))
                Bs.append(lds.b1.view(b_lay))

            atom4 = vec_copy_atom(4)
            lds4 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)

            A_buf = fx.rocdl.make_buffer_tensor(A)
            B_buf = fx.rocdl.make_buffer_tensor(B)
            C_buf = fx.rocdl.make_buffer_tensor(C)

            # This thread's slice of the global->LDS copy.
            a_row0 = tid // a_thr_per_row               # first A row it owns
            a_col = (tid % a_thr_per_row) * 4           # its column (in K)
            b_row0 = tid // b_thr_per_row               # first B row (in K)
            b_col = (tid % b_thr_per_row) * 4           # its column (in N)
            a_glb = [vec_divide(fx.slice(A_buf, (by * bm + a_row0 + p * a_rows_per_pass,
                                                 None)), 4)
                     for p in range_constexpr(a_passes)]

            acc = fx.make_rmem_tensor(tm * tn, fx.Float32)
            for i in range_constexpr(tm * tn):
                fx.memref_store(fx.Float32(0.0), acc, i)

            stage_a = fx.make_rmem_tensor(4 * a_passes, fx.Float32)
            stage_b = fx.make_rmem_tensor(4 * b_passes, fx.Float32)

            def load_tile_to_regs(kt_raw):
                # Clamp: the prefetch of the tile *after* the last one is issued
                # unconditionally (a runtime `if` around it would cost a branch
                # in the hot loop). Re-reading the final tile is harmless -- the
                # data is never accumulated -- whereas indexing past K walks off
                # the tensor and faults.
                kt_v = fx.Int32(kt_raw)   # kt_raw may be a Python int or an SSA value
                kt = (kt_v < n_tiles).select(kt_v, fx.Int32(n_tiles - 1))
                for p in range_constexpr(a_passes):
                    v = load_vec(a_glb[p], (kt * bk + a_col) // 4, atom4, 4)
                    for e in range_constexpr(4):
                        fx.memref_store(v[e], stage_a, p * 4 + e)
                for p in range_constexpr(b_passes):
                    row = fx.slice(B_buf, (kt * bk + b_row0 + p * b_rows_per_pass, None))
                    v = load_vec(vec_divide(row, 4), (bx * bn + b_col) // 4, atom4, 4)
                    for e in range_constexpr(4):
                        fx.memref_store(v[e], stage_b, p * 4 + e)

            def regs_to_lds(buf):
                # A is transposed into LDS: four scattered f32 stores, stride bm.
                for p in range_constexpr(a_passes):
                    for e in range_constexpr(4):
                        fx.memref_store(fx.memref_load(stage_a, p * 4 + e), As[buf],
                                        (a_col + e, a_row0 + p * a_rows_per_pass))
                for p in range_constexpr(b_passes):
                    for e in range_constexpr(4):
                        fx.memref_store(fx.memref_load(stage_b, p * 4 + e), Bs[buf],
                                        (b_row0 + p * b_rows_per_pass, b_col + e))

            def mma_tile(buf):
                for k in range_constexpr(bk):
                    a_k = fx.logical_divide(fx.slice(As[buf], (k, None)),
                                            fx.make_layout(4, 1))
                    b_k = fx.logical_divide(fx.slice(Bs[buf], (k, None)),
                                            fx.make_layout(4, 1))
                    fa = [load_vec(a_k, (ty * tm) // 4 + h, lds4, 4)
                          for h in range_constexpr(tm // 4)]
                    fb = [load_vec(b_k, (tx * tn) // 4 + h, lds4, 4)
                          for h in range_constexpr(tn // 4)]
                    for i in range_constexpr(tm):
                        ai = fa[i // 4][i % 4]
                        for j in range_constexpr(tn):
                            idx = i * tn + j
                            fx.memref_store(
                                fma(ai, fb[j // 4][j % 4], fx.memref_load(acc, idx)),
                                acc, idx)

            if const_expr(lds_stages == 2):
                # Ping-pong. Unrolled by two so the buffer index stays a
                # compile-time constant; the trailing prefetch of tile n_tiles
                # is read back as zeros by the buffer descriptor's bounds check
                # and never reaches the accumulators.
                load_tile_to_regs(0)
                regs_to_lds(0)
                for kt2 in range(n_tiles // 2):
                    kt = kt2 * 2
                    gpu.barrier()
                    load_tile_to_regs(kt + 1)
                    mma_tile(0)
                    regs_to_lds(1)
                    gpu.barrier()
                    load_tile_to_regs(kt + 2)
                    mma_tile(1)
                    regs_to_lds(0)
            elif const_expr(prefetch):
                load_tile_to_regs(0)
                for kt in range(n_tiles):
                    regs_to_lds(0)
                    gpu.barrier()
                    # Issue the next tile's global loads *now*: they retire in
                    # the shadow of this tile's FMAs.
                    load_tile_to_regs(kt + 1)
                    mma_tile(0)
                    gpu.barrier()
            else:
                for kt in range(n_tiles):
                    load_tile_to_regs(kt)
                    regs_to_lds(0)
                    gpu.barrier()
                    mma_tile(0)
                    gpu.barrier()

            # Epilogue: tm rows x (tn/4) float4 stores each.
            for i in range_constexpr(tm):
                c_row = vec_divide(fx.slice(C_buf, (by * bm + ty * tm + i, None)), 4)
                for h in range_constexpr(tn // 4):
                    vec = fx.make_rmem_tensor(4, fx.Float32)
                    for e in range_constexpr(4):
                        fx.memref_store(fx.memref_load(acc, i * tn + h * 4 + e), vec, e)
                    fx.copy_atom_call(
                        atom4, vec,
                        fx.slice(c_row, (None, (bx * bn + tx * tn) // 4 + h)))

        @flyc.jit
        def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, B, C).launch(grid=(N // bn, M // bm, 1), block=(threads, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build


# ── v5: the matrix cores ────────────────────────────────────────────────────

MFMA_M = MFMA_N = 16
MFMA_K = 4          # v_mfma_f32_16x16x4_f32 -- the only f32 MFMA shape worth using
WAVE_M = WAVE_N = 64   # output tile one wavefront owns


def _build_mfma(bm=128, bn=128, bk=8, lds_stages: int = 1):
    """The same blocking, with ``v_mfma_f32_16x16x4_f32`` instead of vector FMA.

    Everything above this point -- the tiling, the LDS staging, the transposed A
    layout -- is reused verbatim. The only thing that changes is the inner
    product, and that is the point: the CUDA ladder's last step was to hand-tune
    the vector FMA schedule at SASS level, and on CDNA the corresponding step is
    to stop using the vector pipe at all.

    Operand layout for ``v_mfma_f32_16x16x4_f32`` on a 64-lane wavefront
    (verified empirically, not recalled -- see docs/porting-notes.md):

        A (16x4)    lane l holds A[l % 16][l / 16]
        B (4x16)    lane l holds B[l / 16][l % 16]
        D (16x16)   lane l holds D[4*(l/16) + r][l % 16] for r in 0..3

    Each wavefront owns a 64x64 output tile = a 4x4 grid of those 16x16 tiles,
    so 16 accumulators of 4 floats -- the same 64 accumulator registers per
    thread the vector version used, reached a completely different way.
    """
    waves_m, waves_n = bm // WAVE_M, bn // WAVE_N
    threads = waves_m * waves_n * 64
    tiles_m, tiles_n = WAVE_M // MFMA_M, WAVE_N // MFMA_N
    a_thr_per_row = bk // 4
    b_thr_per_row = bn // 4
    a_rows_per_pass = threads // a_thr_per_row
    b_rows_per_pass = threads // b_thr_per_row
    if bm % a_rows_per_pass or bk % b_rows_per_pass or bk % MFMA_K:
        raise ValueError(f"unsupported MFMA geometry {bm}x{bn}x{bk}")
    a_passes = bm // a_rows_per_pass
    b_passes = bk // b_rows_per_pass

    def build(M: int, N: int, K: int):
        if M % bm or N % bn or K % bk:
            raise ValueError(f"({M},{N},{K}) not a multiple of ({bm},{bn},{bk})")
        n_tiles = K // bk

        @fx.struct
        class Shared:
            a: fx.Array[fx.Float32, bk * bm, 16]   # As[k][m] -- transposed
            b: fx.Array[fx.Float32, bk * bn, 16]   # Bs[k][n]

        @flyc.kernel
        def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            bx, by = fx.block_idx.x, fx.block_idx.y
            lane = tid % 64
            wave = tid // 64
            wm = (wave // waves_n) * WAVE_M      # this wave's output origin
            wn = (wave % waves_n) * WAVE_N
            li = lane % 16                       # lane's row/col inside a 16x16
            lk = lane // 16                      # lane's k slot (0..3)

            lds = fx.SharedAllocator().allocate(Shared).peek()
            As = lds.a.view(fx.make_layout((bk, bm), (bm, 1)))
            Bs = lds.b.view(fx.make_layout((bk, bn), (bn, 1)))

            atom4 = vec_copy_atom(4)
            atom1 = vec_copy_atom(1)
            A_buf = fx.rocdl.make_buffer_tensor(A)
            B_buf = fx.rocdl.make_buffer_tensor(B)
            C_buf = fx.rocdl.make_buffer_tensor(C)

            a_row0 = tid // a_thr_per_row
            a_col = (tid % a_thr_per_row) * 4
            b_row0 = tid // b_thr_per_row
            b_col = (tid % b_thr_per_row) * 4
            a_glb = [vec_divide(fx.slice(A_buf, (by * bm + a_row0 + p * a_rows_per_pass,
                                                 None)), 4)
                     for p in range_constexpr(a_passes)]

            # One rmem tensor per 16x16 tile: the MFMA accumulator is a
            # vector<4xf32>, and separate allocas keep them out of the runtime
            # K loop's carried values.
            accs = [fx.make_rmem_tensor(4, fx.Float32)
                    for _ in range_constexpr(tiles_m * tiles_n)]
            for t in range_constexpr(tiles_m * tiles_n):
                for e in range_constexpr(4):
                    fx.memref_store(fx.Float32(0.0), accs[t], e)

            for kt in range(n_tiles):
                for p in range_constexpr(a_passes):
                    v = load_vec(a_glb[p], (kt * bk + a_col) // 4, atom4, 4)
                    for e in range_constexpr(4):
                        fx.memref_store(v[e], As,
                                        (a_col + e, a_row0 + p * a_rows_per_pass))
                for p in range_constexpr(b_passes):
                    row = fx.slice(B_buf, (kt * bk + b_row0 + p * b_rows_per_pass, None))
                    v = load_vec(vec_divide(row, 4), (bx * bn + b_col) // 4, atom4, 4)
                    for e in range_constexpr(4):
                        fx.memref_store(v[e], Bs, (b_row0 + p * b_rows_per_pass, b_col + e))
                gpu.barrier()

                for k4 in range_constexpr(bk // MFMA_K):
                    kk = k4 * MFMA_K + lk
                    fa = [fx.memref_load(As, (kk, wm + t * MFMA_M + li))
                          for t in range_constexpr(tiles_m)]
                    fb = [fx.memref_load(Bs, (kk, wn + t * MFMA_N + li))
                          for t in range_constexpr(tiles_n)]
                    for i in range_constexpr(tiles_m):
                        for j in range_constexpr(tiles_n):
                            t = i * tiles_n + j
                            fx.memref_store_vec(
                                mfma_f32_16x16x4_f32(fa[i], fb[j],
                                                     fx.memref_load_vec(accs[t])),
                                accs[t])
                gpu.barrier()

            # Epilogue. Lane l owns rows 4*(l/16)+r of each 16x16 tile and
            # column l%16, so the 16 lanes of a quarter-wave write 16 contiguous
            # floats -- a 64 B transaction per row.
            for i in range_constexpr(tiles_m):
                for e in range_constexpr(4):
                    row = by * bm + wm + i * MFMA_M + 4 * lk + e
                    c_row = vec_divide(fx.slice(C_buf, (row, None)), 1)
                    for j in range_constexpr(tiles_n):
                        t = i * tiles_n + j
                        store_scalar(fx.memref_load(accs[t], e), c_row,
                                     bx * bn + wn + j * MFMA_N + li, atom1)

        @flyc.jit
        def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, B, C).launch(grid=(N // bn, M // bm, 1), block=(threads, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build


# ── op registration ─────────────────────────────────────────────────────────


def _na(*_a, **_k):
    raise RuntimeError("FlyDSL runtime unavailable")


def _g(fn, *a):
    return fn(*a) if HAVE_FLYDSL else (lambda *_a, **_k: _na)


def _g2(fn, *a, **kw):
    return fn(*a, **kw) if HAVE_FLYDSL else (lambda *_a, **_k: _na)


def _make_inputs(*, M, N, K):
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(M, K, generator=g, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, generator=g, device="cuda", dtype=torch.float32)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    return A, B, C


def _reference(A, B, C, *, M, N, K):
    return (A.double() @ B.double()).float()


def _metrics(t, *, M, N, K):
    return {"TFLOP/s": 2 * M * N * K / t / 1e12,
            "GB/s": (M * K + K * N + M * N) * 4 / t / 1e9}


register(
    Op(
        name="sgemm",
        doc="C = A B (f32) -- global->LDS->register blocking, then matrix cores",
        variants=[
            Variant("v0_naive", "one thread per C element, all operands from global",
                    _g(_build_naive), origin="(baseline; the CUDA repo starts at v1)",
                    baseline=True,
                    supports=lambda **s: s["M"] % TS == 0 and s["N"] % TS == 0),
            Variant("v1_lds_tile", "16x16x16 LDS tile, 1 C element per thread",
                    _g(_build_lds_tile), origin="(blocking level 1)",
                    supports=lambda **s: all(s[d] % TS == 0 for d in "MNK")),
            Variant("v2_thread_tile", "128x128x8 tile, 8x8 per thread, float4 + LDS",
                    _g(_build_thread_tile, False), origin="sgemm/sgemm_v1.cu",
                    supports=lambda **s: s["M"] % BM == 0 and s["N"] % BN == 0
                    and s["K"] % BK == 0),
            Variant("v3_prefetch", "v2 + next-tile global prefetch into registers",
                    _g(_build_thread_tile, True), origin="sgemm/sgemm_v3.cu",
                    supports=lambda **s: s["M"] % BM == 0 and s["N"] % BN == 0
                    and s["K"] % BK == 0),
            Variant("v4_double_buffer", "v3 + LDS ping-pong: one barrier per K-tile",
                    _g2(_build_thread_tile, True, lds_stages=2),
                    origin="sgemm/sgemm_v3.cu (ENABLE_DOUBLE_BUFFER)",
                    supports=lambda **s: s["M"] % BM == 0 and s["N"] % BN == 0
                    and s["K"] % (2 * BK) == 0),
            Variant("v5_mfma", "same blocking, v_mfma_f32_16x16x4_f32 matrix cores",
                    _g2(_build_mfma),
                    origin="(CDNA4 answer to the repo's SASS-tuning chapter)",
                    supports=lambda **s: s["M"] % BM == 0 and s["N"] % BN == 0
                    and s["K"] % BK == 0),
        ],
        shapes=[Shape("1024^3", {"M": 1024, "N": 1024, "K": 1024}),
                Shape("2048^3", {"M": 2048, "N": 2048, "K": 2048}),
                Shape("4096^3", {"M": 4096, "N": 4096, "K": 4096})],
        make_inputs=_make_inputs,
        reference=_reference,
        output_index=2,
        metrics=_metrics,
        torch_baseline=lambda A, B, C, *, M, N, K: torch.mm(A, B, out=C),
        tol={"rtol": 2e-3, "atol": 2e-3},
    )
)
