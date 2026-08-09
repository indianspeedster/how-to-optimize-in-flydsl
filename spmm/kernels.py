# SPDX-License-Identifier: Apache-2.0
"""SpMM (C = A_csr B, dense B) -- the "reuse the sparse row" study.

Ports ``spmm/spmm.cu``. Its two kernels answer one question: given that a CSR row
must be walked serially, what does a thread block do with the fact that *every*
output column of that row walks the **same** row?

``v0`` is the CUDA ``My_spmm_csr_vector_kernel_v0``: one block row, one thread
per output column, and the whole block reads the same ``(col_index, value)``
pair on every step. That looks wasteful and mostly is not -- the pair is
broadcast out of L1 -- which is why the CUDA author labels the LDS version below
"useless optimize" in the source.

``v1`` reproduces the LDS version anyway, and the measurement splits the
original's verdict in two. On a **uniform** matrix the verdict holds: staging
costs 0.72x, exactly the wasted-effort result the comment predicts. On a
**skewed** (power-law) matrix the same kernel is **4.6x faster** than v0. The
reason is not the LDS at all -- it is the ``CHUNK``-sized outer loop the staging
forces, which converts one thread's unboundedly long row walk into a sequence of
barrier-synchronised passes that the whole block advances through together.
"Useless" was measured on balanced matrices; it does not survive contact with a
real graph. See ``docs/porting-notes.md``.

``v2`` has no CUDA counterpart and is where the real win is: give each thread
four output columns and read ``B`` with ``buffer_load_dwordx4``. The sparse row
walk is unchanged; only the dense side gets wider, and the dense side is all of
the traffic.
"""

# No `from __future__ import annotations`.

import torch

from common.dsl import (
    HAVE_FLYDSL,
    const_expr,
    fast_launcher,
    flyc,
    fma,
    fx,
    gpu,
    load_scalar,
    load_vec,
    range_constexpr,
    store_scalar,
    store_vec,
    vec_copy_atom,
    vec_divide,
)
from common.sparse import csr_to_torch, make_csr

THREADS = 256


def _build(vec_width: int = 1, stage_lds: bool = False):
    """One workgroup per (row, column-chunk); each thread owns ``vec_width`` columns.

    ``stage_lds`` first copies a chunk of the row's ``(col_index, value)`` pairs
    into LDS and reads them from there -- the CUDA v1 experiment.
    """

    def build(rows, cols, nnz_per_row, ncols, pattern):
        # With vec_width=4 a 256-thread block wants 1024 output columns; narrow
        # problems get a narrower block rather than being refused.
        threads = min(THREADS, max(64, ncols // vec_width))
        cols_per_block = threads * vec_width
        if ncols % cols_per_block:
            raise ValueError(f"ncols={ncols} not a multiple of {cols_per_block}")
        col_blocks = ncols // cols_per_block
        CHUNK = threads       # nnz staged per LDS pass

        @fx.struct
        class Shared:
            s_col: fx.Array[fx.Int32, CHUNK, 16]
            s_val: fx.Array[fx.Float32, CHUNK, 16]

        @flyc.kernel
        def kernel(row_offset: fx.Tensor, col_index: fx.Tensor, value: fx.Tensor,
                   B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            bx = fx.block_idx.x        # which chunk of output columns
            row = fx.block_idx.y

            atom_f = vec_copy_atom(1)
            atom_i = vec_copy_atom(1, fx.Int32)
            atom_v = vec_copy_atom(vec_width)

            ro = vec_divide(fx.rocdl.make_buffer_tensor(row_offset), 1)
            ci = vec_divide(fx.rocdl.make_buffer_tensor(col_index), 1)
            va = vec_divide(fx.rocdl.make_buffer_tensor(value), 1)
            B_buf = fx.rocdl.make_buffer_tensor(B)
            C_buf = fx.rocdl.make_buffer_tensor(C)

            out_col = bx * threads + tid          # in units of `vec_width` floats
            start = load_scalar(ro, row, atom_i, fx.Int32)
            end = load_scalar(ro, row + 1, atom_i, fx.Int32)

            acc = fx.make_rmem_tensor(vec_width, fx.Float32)
            for e in range_constexpr(vec_width):
                fx.memref_store(fx.Float32(0.0), acc, e)

            if const_expr(stage_lds):
                lds = fx.SharedAllocator().allocate(Shared).peek()
                s_col = lds.s_col.view(fx.make_layout(CHUNK, 1))
                s_val = lds.s_val.view(fx.make_layout(CHUNK, 1))

                for base in range(start, end, CHUNK):
                    # Cooperative load of up to CHUNK pairs. Out-of-row slots get
                    # value 0 so the multiply below is a harmless += 0 and the
                    # inner loop keeps a compile-time trip count.
                    j = base + tid
                    live = j < end
                    j_safe = live.select(j, fx.Int32(0))
                    fx.memref_store(load_scalar(ci, j_safe, atom_i, fx.Int32), s_col, tid)
                    fx.memref_store(
                        live.select(load_scalar(va, j_safe, atom_f), fx.Float32(0.0)),
                        s_val, tid)
                    gpu.barrier()
                    for t in range(CHUNK):
                        b_row = vec_divide(fx.slice(B_buf, (fx.memref_load(s_col, t),
                                                            None)), vec_width)
                        v = load_vec(b_row, out_col, atom_v, vec_width)
                        w = fx.memref_load(s_val, t)
                        for e in range_constexpr(vec_width):
                            fx.memref_store(fma(w, v[e], fx.memref_load(acc, e)), acc, e)
                    gpu.barrier()
            else:
                for i in range(start, end):
                    col = load_scalar(ci, i, atom_i, fx.Int32)
                    w = load_scalar(va, i, atom_f)
                    b_row = vec_divide(fx.slice(B_buf, (col, None)), vec_width)
                    v = load_vec(b_row, out_col, atom_v, vec_width)
                    for e in range_constexpr(vec_width):
                        fx.memref_store(fma(w, v[e], fx.memref_load(acc, e)), acc, e)

            c_row = vec_divide(fx.slice(C_buf, (row, None)), vec_width)
            store_vec(fx.memref_load_vec(acc), c_row, out_col, atom_v, vec_width)

        @flyc.jit
        def launch(row_offset: fx.Tensor, col_index: fx.Tensor, value: fx.Tensor,
                   B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            kernel(row_offset, col_index, value, B, C).launch(
                grid=(col_blocks, rows, 1), block=(threads, 1, 1), stream=stream)

        return fast_launcher(launch)

    return build
