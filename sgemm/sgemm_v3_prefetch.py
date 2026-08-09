# SPDX-License-Identifier: Apache-2.0
"""Rung 3 -- overlap the global loads with the math. Ports the prefetch half of
``sgemm/sgemm_v3.cu``.

Identical to v2 except for the order of four statements in the K loop: the next
tile's global loads are issued before this tile's FMAs instead of after them.

Worth almost nothing at 4096^3 (68.8 vs 68.7 TFLOP/s) and a real 1.14x at
2048^3. The reason is the interesting part: by 4096^3 the kernel is already at
97% of the FP32 vector peak, so there is no stall left to hide.
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
    load_vec,
    range_constexpr,
    vec_copy_atom,
    vec_divide,
)

BM, BN, BK = 128, 128, 8      # block tile
TM, TN = 8, 8                 # register tile per thread
THREADS = (BM // TM) * (BN // TN)      # 256

# Global -> LDS partition. Every thread moves float4s.
A_THR_PER_ROW = BK // 4                 # threads spanning one A row (along K)
B_THR_PER_ROW = BN // 4                 # threads spanning one B row (along N)
A_ROWS_PER_PASS = THREADS // A_THR_PER_ROW
B_ROWS_PER_PASS = THREADS // B_THR_PER_ROW
A_PASSES = BM // A_ROWS_PER_PASS
B_PASSES = BK // B_ROWS_PER_PASS

def build(M: int, N: int, K: int):
    if M % BM or N % BN or K % BK:
        raise ValueError(f"({M},{N},{K}) not a multiple of ({BM},{BN},BK)")
    n_tiles = K // BK

    @fx.struct
    class Shared:
        a0: fx.Array[fx.Float32, BK * BM, 16]   # As[k][m] -- transposed
        b0: fx.Array[fx.Float32, BK * BN, 16]   # Bs[k][n]

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bx, by = fx.block_idx.x, fx.block_idx.y
        tx, ty = tid % (BN // TN), tid // (BN // TN)

        lds = fx.SharedAllocator().allocate(Shared).peek()
        a_lay = fx.make_layout((BK, BM), (BM, 1))
        b_lay = fx.make_layout((BK, BN), (BN, 1))
        As = [lds.a0.view(a_lay)]
        Bs = [lds.b0.view(b_lay)]

        atom4 = vec_copy_atom(4)
        lds4 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)

        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        # This thread's slice of the global->LDS copy.
        a_row0 = tid // A_THR_PER_ROW               # first A row it owns
        a_col = (tid % A_THR_PER_ROW) * 4           # its column (in K)
        b_row0 = tid // B_THR_PER_ROW               # first B row (in K)
        b_col = (tid % B_THR_PER_ROW) * 4           # its column (in N)
        a_glb = [vec_divide(fx.slice(A_buf, (by * BM + a_row0 + p * A_ROWS_PER_PASS,
                                             None)), 4)
                 for p in range_constexpr(A_PASSES)]

        acc = fx.make_rmem_tensor(TM * TN, fx.Float32)
        for i in range_constexpr(TM * TN):
            fx.memref_store(fx.Float32(0.0), acc, i)

        stage_a = fx.make_rmem_tensor(4 * A_PASSES, fx.Float32)
        stage_b = fx.make_rmem_tensor(4 * B_PASSES, fx.Float32)

        def load_tile_to_regs(kt_raw):
            # Clamp: the prefetch of the tile *after* the last one is issued
            # unconditionally (a runtime `if` around it would cost a branch in
            # the hot loop). Re-reading the final tile is harmless -- the data is
            # never accumulated -- whereas indexing past K walks off the tensor
            # and faults. See docs/porting-notes.md Sec. 2.5.
            kt_v = fx.Int32(kt_raw)   # may be a Python int or an SSA value
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
            # A is transposed into LDS: four scattered f32 stores, stride BM.
            # That is what lets the inner loop below read TM *contiguous* floats
            # per operand -- one ds_read_b128 per four, not four ds_read_b32.
            for p in range_constexpr(A_PASSES):
                for e in range_constexpr(4):
                    fx.memref_store(fx.memref_load(stage_a, p * 4 + e), As[buf],
                                    (a_col + e, a_row0 + p * A_ROWS_PER_PASS))
            for p in range_constexpr(B_PASSES):
                for e in range_constexpr(4):
                    fx.memref_store(fx.memref_load(stage_b, p * 4 + e), Bs[buf],
                                    (b_row0 + p * B_ROWS_PER_PASS, b_col + e))

        def mma_tile(buf):
            for k in range_constexpr(BK):
                a_k = fx.logical_divide(fx.slice(As[buf], (k, None)),
                                        fx.make_layout(4, 1))
                b_k = fx.logical_divide(fx.slice(Bs[buf], (k, None)),
                                        fx.make_layout(4, 1))
                fa = [load_vec(a_k, (ty * TM) // 4 + h, lds4, 4)
                      for h in range_constexpr(TM // 4)]
                fb = [load_vec(b_k, (tx * TN) // 4 + h, lds4, 4)
                      for h in range_constexpr(TN // 4)]
                # `fma`, not `a * b + c`: IEEE f32 forbids contracting the two
                # without an explicit licence, and the uncontracted form costs
                # 6x here. See docs/porting-notes.md Sec. 2.7.
                for i in range_constexpr(TM):
                    ai = fa[i // 4][i % 4]
                    for j in range_constexpr(TN):
                        idx = i * TN + j
                        fx.memref_store(
                            fma(ai, fb[j // 4][j % 4], fx.memref_load(acc, idx)),
                            acc, idx)

        # The next tile's global loads are issued BEFORE this tile's math, so
        # VMEM retires in the shadow of 512 FMAs instead of stalling behind a
        # barrier. Still two barriers per K-tile: LDS is single-buffered, so the
        # write of tile k+1 cannot start until every wave has finished reading
        # tile k.
        load_tile_to_regs(0)
        for kt in range(n_tiles):
            regs_to_lds(0)
            gpu.barrier()
            load_tile_to_regs(kt + 1)
            mma_tile(0)
            gpu.barrier()

        # Epilogue: TM rows x (TN/4) float4 stores each.
        for i in range_constexpr(TM):
            c_row = vec_divide(fx.slice(C_buf, (by * BM + ty * TM + i, None)), 4)
            for h in range_constexpr(TN // 4):
                vec = fx.make_rmem_tensor(4, fx.Float32)
                for e in range_constexpr(4):
                    fx.memref_store(fx.memref_load(acc, i * TN + h * 4 + e), vec, e)
                fx.copy_atom_call(
                    atom4, vec,
                    fx.slice(c_row, (None, (bx * BN + tx * TN) // 4 + h)))

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        kernel(A, B, C).launch(grid=(N // BN, M // BM, 1), block=(THREADS, 1, 1),
                               stream=stream)

    return fast_launcher(launch)
