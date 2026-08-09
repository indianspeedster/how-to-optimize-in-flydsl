# SPDX-License-Identifier: Apache-2.0
"""v2 / v3 / v4: two levels of blocking, TMxTN accumulators per thread.

Ports ``sgemm_v1.cu`` and ``sgemm_v3.cu``. The second blocking level -- LDS into
registers -- is what actually raises arithmetic intensity; the two latency-hiding
switches on top of it (register prefetch, LDS ping-pong) are the ``sgemm_v3.cu``
double-buffering idea.
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

BM, BN, BK = 128, 128, 8
TM, TN = 8, 8
THREADS = (BM // TM) * (BN // TN)      # 256


def build_thread_tile(prefetch: bool, *, bm=BM, bn=BN, bk=BK, tm=TM, tn=TN,
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
