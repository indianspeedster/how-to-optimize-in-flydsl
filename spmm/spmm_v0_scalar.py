# SPDX-License-Identifier: Apache-2.0
"""Rung 0 -- one output column per thread. Ports
``spmm/spmm.cu:My_spmm_csr_vector_kernel_v0``.

One workgroup per (row, column-chunk). Every thread in the block walks the same
CSR row and multiplies each non-zero into its own column of B. The sparse side is
read redundantly by all 256 threads; the dense side is where all the traffic is.

The baseline for this folder.
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

        # Walk the row straight out of global memory. Every thread in the block
        # reads the same (col_index, value) pair on every step, which looks
        # wasteful and mostly is not -- it is broadcast out of L1.
        for i in range(start, end):
            col = load_scalar(ci, i, atom_i, fx.Int32)
            w = load_scalar(va, i, atom_f)
            b_row = vec_divide(fx.slice(B_buf, (col, None)), VEC)
            v = load_vec(b_row, out_col, atom_v, VEC)
            for e in range_constexpr(VEC):
                fx.memref_store(fma(w, v[e], fx.memref_load(acc, e)), acc, e)

        c_row = vec_divide(fx.slice(C_buf, (row, None)), VEC)
        store_vec(fx.memref_load_vec(acc), c_row, out_col, atom_v, VEC)

    @flyc.jit
    def launch(row_offset: fx.Tensor, col_index: fx.Tensor, value: fx.Tensor,
               B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(row_offset, col_index, value, B, C).launch(
            grid=(col_blocks, rows, 1), block=(threads, 1, 1), stream=stream)

    return fast_launcher(launch)
