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


def build_mfma(bm=128, bn=128, bk=8, lds_stages: int = 1):
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
