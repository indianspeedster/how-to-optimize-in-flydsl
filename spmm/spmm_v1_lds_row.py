# SPDX-License-Identifier: Apache-2.0
"""Rung 1 -- stage the sparse row in LDS. Ports
``spmm/spmm.cu:My_spmm_csr_vector_kernel_v1``, the kernel whose CUDA source
comment reads ``// useless optimize``.

The idea: the block reads the same ``(col_index, value)`` pair 256 times, so
cooperatively load a CHUNK of them into LDS once and read them from there.

The measurement splits that verdict in two.

**On a uniform matrix the comment is exactly right**: 0.72x, pure overhead. The
pair was already being broadcast out of L1, so the LDS round-trip buys nothing
and the barriers cost.

**On a power-law matrix this kernel is 4.6x faster than v0.** The reason is not
the LDS at all -- it is the ``CHUNK``-sized outer loop that staging *forces*,
which converts one thread's unboundedly long row walk into a sequence of
barrier-synchronised passes the whole workgroup advances through together.
"Useless" was measured on balanced matrices; it does not survive contact with a
real graph.
"""

# No `from __future__ import annotations` -- @fx.struct resolves its field
# annotations eagerly and PEP 563 stringification breaks the LDS layout.

from common.dsl import (
    fast_launcher,
    flyc,
    fma,
    fx,
    gpu,
    load_scalar,
    load_vec,
    range_constexpr,
    store_vec,
    vec_copy_atom,
    vec_divide,
)

THREADS = 256

VEC = 1            # output columns each thread owns


def build(rows, cols, nnz_per_row, ncols, pattern):
    # With VEC=4 a 256-thread block wants 1024 output columns; narrow problems
    # get a narrower block rather than being refused.
    threads = min(THREADS, max(64, ncols // VEC))
    cols_per_block = threads * VEC
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
        atom_v = vec_copy_atom(VEC)

        ro = vec_divide(fx.rocdl.make_buffer_tensor(row_offset), 1)
        ci = vec_divide(fx.rocdl.make_buffer_tensor(col_index), 1)
        va = vec_divide(fx.rocdl.make_buffer_tensor(value), 1)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        out_col = bx * threads + tid          # in units of `VEC` floats
        start = load_scalar(ro, row, atom_i, fx.Int32)
        end = load_scalar(ro, row + 1, atom_i, fx.Int32)

        acc = fx.make_rmem_tensor(VEC, fx.Float32)
        for e in range_constexpr(VEC):
            fx.memref_store(fx.Float32(0.0), acc, e)

        lds = fx.SharedAllocator().allocate(Shared).peek()
        s_col = lds.s_col.view(fx.make_layout(CHUNK, 1))
        s_val = lds.s_val.view(fx.make_layout(CHUNK, 1))

        for base in range(start, end, CHUNK):
            # Cooperative load of up to CHUNK pairs. Out-of-row slots get value
            # 0 so the multiply below is a harmless += 0 and the inner loop
            # keeps a compile-time trip count.
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
                                                    None)), VEC)
                v = load_vec(b_row, out_col, atom_v, VEC)
                w = fx.memref_load(s_val, t)
                for e in range_constexpr(VEC):
                    fx.memref_store(fma(w, v[e], fx.memref_load(acc, e)), acc, e)
            gpu.barrier()

        c_row = vec_divide(fx.slice(C_buf, (row, None)), VEC)
        store_vec(fx.memref_load_vec(acc), c_row, out_col, atom_v, VEC)

    @flyc.jit
    def launch(row_offset: fx.Tensor, col_index: fx.Tensor, value: fx.Tensor,
               B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(row_offset, col_index, value, B, C).launch(
            grid=(col_blocks, rows, 1), block=(threads, 1, 1), stream=stream)

    return fast_launcher(launch)
