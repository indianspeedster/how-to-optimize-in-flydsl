# SPDX-License-Identifier: Apache-2.0
"""v6: the matrix-core rung with the production levers applied.

v5 proved the matrix cores are worth 1.65x over the best vector rung, but it is
a *first* MFMA kernel: one LDS buffer, no prefetch, no instruction scheduling,
and one fixed 128x128 tile regardless of problem size. Each of those is a known
lever, and `/gemm-optimization` (the FlyDSL production GEMM skill) names all
four. This rung applies them.

1. **A tile chosen from the problem size.** The single biggest loss was never
   kernel quality -- at 1024^3 a 128x128 tile yields an 8x8 = 64-block grid on a
   256-CU part, so three quarters of the GPU idles. `_pick_tile` sizes the tile
   so the grid keeps the machine full, which is exactly what a tuned library
   does and what v2-v5 never did.
2. **LDS ping-pong.** Two A/B buffers: this tile's MFMAs read buffer p while the
   next tile lands in buffer 1-p, so one barrier per K-tile instead of two.
3. **Global -> register prefetch.** The next tile's loads are issued before this
   tile's math, so VMEM retires under the MFMAs.
4. **Hot-loop scheduling hints.** `rocdl.sched_mfma/dsrd/vmem/dswr` interleave
   the LDS reads and global loads between MFMA groups instead of letting them
   bunch at the top of the loop.

The operand layout is v5's, unchanged and already verified on hardware:

    A (16x4)    lane l holds A[l % 16][l / 16]
    B (4x16)    lane l holds B[l / 16][l % 16]
    D (16x16)   lane l holds D[4*(l/16) + r][l % 16] for r in 0..3
"""

# No `from __future__ import annotations` -- see sgemm/sgemm_v0_naive.py.

from common.dsl import (
    fast_launcher,
    flyc,
    fx,
    gpu,
    load_vec,
    mfma_f32_16x16x4_f32,
    range_constexpr,
    rocdl,
    store_scalar,
    vec_copy_atom,
    vec_divide,
)
from common.env import arch

_SCHED = True      # hot-loop scheduling hints (swept; see docs/porting-notes.md)

MFMA_M = MFMA_N = 16
MFMA_K = 4

# (bm, bn, wave_m, wave_n, bk) -- wave_m/wave_n is the output tile ONE wavefront
# owns, so threads = (bm/wave_m) * (bn/wave_n) * 64.
# BK=16 throughout: BK=32 doubles the ping-pong LDS to 64 KB per block, which
# costs more in occupancy than the deeper tile buys back (measured: 105 vs 114
# TFLOP/s at 4096^3). BK=8 and BK=16 measure the same; 16 is kept for the
# shorter K loop.
_TILES = [
    (128, 128, 64, 64, 16),     # large: 4 waves, 64 accumulators per thread
    (128, 64, 64, 32, 16),
    (64, 64, 32, 32, 16),       # small: keeps the grid full when M, N are small
]


def _pick_tile(M, N, K):
    """Largest tile that still fills the machine.

    A tile is only worth its reuse if enough blocks exist to cover the CUs. We
    want at least two blocks per CU so the scheduler has something to overlap;
    below that, a smaller tile with more blocks wins even though it re-reads
    more data. This is the lever v2-v5 are missing, and the whole reason v5
    lost 5.5x to rocBLAS at 1024^3 while only losing 1.2x at 4096^3.
    """
    want = 2 * arch().cus
    for bm, bn, wm, wn, bk in _TILES:
        if M % bm or N % bn or K % (2 * bk):
            continue
        if (M // bm) * (N // bn) >= want:
            return bm, bn, wm, wn, bk
    # Nothing fills the machine: take the smallest tile that divides the shape.
    for t in reversed(_TILES):
        if M % t[0] == 0 and N % t[1] == 0 and K % (2 * t[4]) == 0:
            return t
    raise ValueError(f"no tile divides ({M},{N},{K})")


def build(M: int, N: int, K: int):
    BM, BN, WM, WN, BK = _pick_tile(M, N, K)
    WAVES_M, WAVES_N = BM // WM, BN // WN
    THREADS = WAVES_M * WAVES_N * 64
    TILES_M, TILES_N = WM // MFMA_M, WN // MFMA_N

    A_THR_PER_ROW = BK // 4
    B_THR_PER_ROW = BN // 4
    A_ROWS_PER_PASS = THREADS // A_THR_PER_ROW
    B_ROWS_PER_PASS = THREADS // B_THR_PER_ROW
    A_PASSES = max(1, BM // A_ROWS_PER_PASS)
    B_PASSES = max(1, BK // B_ROWS_PER_PASS)
    n_tiles = K // BK

    # One scheduler iteration per k4 micro-step: a group of MFMAs, then the
    # LDS reads and global loads that feed the next one.
    MFMA_GROUP = TILES_M * TILES_N
    K4_STEPS = BK // MFMA_K

    @fx.struct
    class Shared:
        a0: fx.Array[fx.Float32, BK * BM, 16]     # As[k][m], transposed
        b0: fx.Array[fx.Float32, BK * BN, 16]     # Bs[k][n]
        a1: fx.Array[fx.Float32, BK * BM, 16]     # the ping-pong halves
        b1: fx.Array[fx.Float32, BK * BN, 16]

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bx, by = fx.block_idx.x, fx.block_idx.y
        lane = tid % 64
        wave = tid // 64
        wm = (wave // WAVES_N) * WM
        wn = (wave % WAVES_N) * WN
        li = lane % 16
        lk = lane // 16

        lds = fx.SharedAllocator().allocate(Shared).peek()
        a_lay = fx.make_layout((BK, BM), (BM, 1))
        b_lay = fx.make_layout((BK, BN), (BN, 1))
        As = [lds.a0.view(a_lay), lds.a1.view(a_lay)]
        Bs = [lds.b0.view(b_lay), lds.b1.view(b_lay)]

        atom4 = vec_copy_atom(4)
        atom1 = vec_copy_atom(1)
        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        a_row0 = tid // A_THR_PER_ROW
        a_col = (tid % A_THR_PER_ROW) * 4
        b_row0 = tid // B_THR_PER_ROW
        b_col = (tid % B_THR_PER_ROW) * 4
        a_glb = [vec_divide(fx.slice(A_buf, (by * BM + a_row0 + p * A_ROWS_PER_PASS,
                                             None)), 4)
                 for p in range_constexpr(A_PASSES)]

        accs = [fx.make_rmem_tensor(4, fx.Float32)
                for _ in range_constexpr(TILES_M * TILES_N)]
        for t in range_constexpr(TILES_M * TILES_N):
            for e in range_constexpr(4):
                fx.memref_store(fx.Float32(0.0), accs[t], e)

        stage_a = fx.make_rmem_tensor(4 * A_PASSES, fx.Float32)
        stage_b = fx.make_rmem_tensor(4 * B_PASSES, fx.Float32)

        def load_regs(kt_raw):
            # Clamped, as in v3/v4: the trailing prefetch re-reads the last tile
            # rather than indexing past K. See docs/porting-notes.md Sec. 2.5.
            kt_v = fx.Int32(kt_raw)
            kt = (kt_v < n_tiles).select(kt_v, fx.Int32(n_tiles - 1))
            for p in range_constexpr(A_PASSES):
                v = load_vec(a_glb[p], (kt * BK + a_col) // 4, atom4, 4)
                for e in range_constexpr(4):
                    fx.memref_store(v[e], stage_a, p * 4 + e)
            for p in range_constexpr(B_PASSES):
                row = fx.slice(B_buf, (kt * BK + b_row0 + p * B_ROWS_PER_PASS, None))
                v = load_vec(vec_divide(row, 4), (bx * BN + b_col) // 4, atom4, 4)
                for e in range_constexpr(4):
                    fx.memref_store(v[e], stage_b, p * 4 + e)

        def regs_to_lds(buf):
            for p in range_constexpr(A_PASSES):
                for e in range_constexpr(4):
                    fx.memref_store(fx.memref_load(stage_a, p * 4 + e), As[buf],
                                    (a_col + e, a_row0 + p * A_ROWS_PER_PASS))
            for p in range_constexpr(B_PASSES):
                for e in range_constexpr(4):
                    fx.memref_store(fx.memref_load(stage_b, p * 4 + e), Bs[buf],
                                    (b_row0 + p * B_ROWS_PER_PASS, b_col + e))

        def schedule():
            """Interleave the MFMA groups with the loads that feed the next tile.

            The production pattern from the gemm-optimization skill. MEASURED
            EFFECT HERE: none -- within noise at every shape swept, with and
            without. Kept because it is the documented lever and because a
            different dtype or a deeper K loop may well need it, but it is not
            what makes this rung faster than v5. The tile picker is.
            """
            if not _SCHED:
                return
            for _ in range_constexpr(K4_STEPS):
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(MFMA_GROUP // 2)
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(MFMA_GROUP // 2)
            rocdl.sched_dswr(1)
            rocdl.sched_barrier(0)

        def mma_tile(buf):
            for k4 in range_constexpr(K4_STEPS):
                kk = k4 * MFMA_K + lk
                fa = [fx.memref_load(As[buf], (kk, wm + t * MFMA_M + li))
                      for t in range_constexpr(TILES_M)]
                fb = [fx.memref_load(Bs[buf], (kk, wn + t * MFMA_N + li))
                      for t in range_constexpr(TILES_N)]
                for i in range_constexpr(TILES_M):
                    for j in range_constexpr(TILES_N):
                        t = i * TILES_N + j
                        fx.memref_store_vec(
                            mfma_f32_16x16x4_f32(fa[i], fb[j],
                                                 fx.memref_load_vec(accs[t])),
                            accs[t])

        # Ping-pong, unrolled by two so the buffer index stays compile-time.
        load_regs(0)
        regs_to_lds(0)
        for kt2 in range(n_tiles // 2):
            kt = kt2 * 2
            gpu.barrier()
            load_regs(kt + 1)
            mma_tile(0)
            schedule()
            regs_to_lds(1)
            gpu.barrier()
            load_regs(kt + 2)
            mma_tile(1)
            schedule()
            regs_to_lds(0)

        for i in range_constexpr(TILES_M):
            for e in range_constexpr(4):
                row = by * BM + wm + i * MFMA_M + 4 * lk + e
                c_row = vec_divide(fx.slice(C_buf, (row, None)), 1)
                for j in range_constexpr(TILES_N):
                    t = i * TILES_N + j
                    store_scalar(fx.memref_load(accs[t], e), c_row,
                                 bx * BN + wn + j * MFMA_N + li, atom1)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        kernel(A, B, C).launch(grid=(N // BN, M // BM, 1), block=(THREADS, 1, 1),
                               stream=stream)

    return fast_launcher(launch)
