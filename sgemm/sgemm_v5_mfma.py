# SPDX-License-Identifier: Apache-2.0
"""v5: the same blocking, run on the matrix cores instead of the vector FMA pipe.

This is where the port stops being a translation. The CUDA repo's last chapter
is SASS-level register remapping with CuAssembler to squeeze the vector FMA
schedule; CDNA answers that problem with a different functional unit.
"""

# No `from __future__ import annotations` -- @fx.struct resolves its field
# annotations eagerly and PEP 563 stringification breaks the LDS layout.

from common.dsl import (
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
    vec_copy_atom,
    vec_divide,
)

MFMA_M = MFMA_N = 16
MFMA_K = 4          # v_mfma_f32_16x16x4_f32 -- the only f32 MFMA shape worth using
WAVE_M = WAVE_N = 64   # output tile one wavefront owns

BM, BN, BK = 128, 128, 8       # the same block tile as v2-v4
WAVES_M, WAVES_N = BM // WAVE_M, BN // WAVE_N
THREADS = WAVES_M * WAVES_N * 64
TILES_M, TILES_N = WAVE_M // MFMA_M, WAVE_N // MFMA_N   # 4x4 MFMA tiles per wave

# Global -> LDS partition, identical to v2-v4: every thread moves float4s.
A_THR_PER_ROW = BK // 4
B_THR_PER_ROW = BN // 4
A_ROWS_PER_PASS = THREADS // A_THR_PER_ROW
B_ROWS_PER_PASS = THREADS // B_THR_PER_ROW
A_PASSES = BM // A_ROWS_PER_PASS
B_PASSES = BK // B_ROWS_PER_PASS


def build(M: int, N: int, K: int):
    if M % BM or N % BN or K % BK:
        raise ValueError(f"({M},{N},{K}) not a multiple of ({BM},{BN},{BK})")
    n_tiles = K // BK

    @fx.struct
    class Shared:
        a: fx.Array[fx.Float32, BK * BM, 16]   # As[k][m] -- transposed
        b: fx.Array[fx.Float32, BK * BN, 16]   # Bs[k][n]

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bx, by = fx.block_idx.x, fx.block_idx.y
        lane = tid % 64
        wave = tid // 64
        wm = (wave // WAVES_N) * WAVE_M      # this wave's output origin
        wn = (wave % WAVES_N) * WAVE_N
        li = lane % 16                       # lane's row/col inside a 16x16
        lk = lane // 16                      # lane's k slot (0..3)

        lds = fx.SharedAllocator().allocate(Shared).peek()
        As = lds.a.view(fx.make_layout((BK, BM), (BM, 1)))
        Bs = lds.b.view(fx.make_layout((BK, BN), (BN, 1)))

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

        # One rmem tensor per 16x16 tile: the MFMA accumulator is a
        # vector<4xf32>, and separate allocas keep them out of the runtime
        # K loop's carried values.
        accs = [fx.make_rmem_tensor(4, fx.Float32)
                for _ in range_constexpr(TILES_M * TILES_N)]
        for t in range_constexpr(TILES_M * TILES_N):
            for e in range_constexpr(4):
                fx.memref_store(fx.Float32(0.0), accs[t], e)

        for kt in range(n_tiles):
            for p in range_constexpr(A_PASSES):
                v = load_vec(a_glb[p], (kt * BK + a_col) // 4, atom4, 4)
                for e in range_constexpr(4):
                    fx.memref_store(v[e], As,
                                    (a_col + e, a_row0 + p * A_ROWS_PER_PASS))
            for p in range_constexpr(B_PASSES):
                row = fx.slice(B_buf, (kt * BK + b_row0 + p * B_ROWS_PER_PASS, None))
                v = load_vec(vec_divide(row, 4), (bx * BN + b_col) // 4, atom4, 4)
                for e in range_constexpr(4):
                    fx.memref_store(v[e], Bs, (b_row0 + p * B_ROWS_PER_PASS, b_col + e))
            gpu.barrier()

            for k4 in range_constexpr(BK // MFMA_K):
                kk = k4 * MFMA_K + lk
                fa = [fx.memref_load(As, (kk, wm + t * MFMA_M + li))
                      for t in range_constexpr(TILES_M)]
                fb = [fx.memref_load(Bs, (kk, wn + t * MFMA_N + li))
                      for t in range_constexpr(TILES_N)]
                for i in range_constexpr(TILES_M):
                    for j in range_constexpr(TILES_N):
                        t = i * TILES_N + j
                        fx.memref_store_vec(
                            mfma_f32_16x16x4_f32(fa[i], fb[j],
                                                 fx.memref_load_vec(accs[t])),
                            accs[t])
            gpu.barrier()

        # Epilogue. Lane l owns rows 4*(l/16)+r of each 16x16 tile and
        # column l%16, so the 16 lanes of a quarter-wave write 16 contiguous
        # floats -- a 64 B transaction per row.
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
