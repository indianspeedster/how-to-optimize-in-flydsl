# SPDX-License-Identifier: Apache-2.0
"""Rung 4 -- a full 64-lane wavefront per row. Ports ``spmv/spmv.cu`` with
``THREADS_PER_VECTOR = 32`` on CUDA; on CDNA the widest segment is the whole
64-lane wavefront.

On a uniform matrix this is past the optimum: 64 lanes on a 32-non-zero row means
half of them do nothing, and at 8 nnz/row it is the *worst* setting in the folder
(0.53x the scalar kernel).

On a power-law matrix it is the best by a wide margin -- **13.8x the scalar
kernel** -- because a wavefront is exactly what absorbs load imbalance: the long
rows get 64 lanes instead of one. The right knob setting is a property of the
matrix, not of the hardware.
"""

# No `from __future__ import annotations` -- FlyDSL resolves annotations eagerly.

from common.dsl import (
    fast_launcher,
    flyc,
    fx,
    load_scalar,
    store_scalar,
    vec_copy_atom,
    vec_divide,
    wave_reduce_sum_down,
)
from common.env import wave_size

THREADS = 256

LANES_PER_ROW = 64


def build(rows: int, cols: int, nnz_per_row: int, pattern: str):
    W = wave_size()
    if LANES_PER_ROW > W or W % LANES_PER_ROW:
        raise ValueError(f"LANES_PER_ROW={LANES_PER_ROW} must divide {W}")
    rows_per_block = THREADS // LANES_PER_ROW
    blocks = (rows + rows_per_block - 1) // rows_per_block

    @flyc.kernel
    def kernel(row_offset: fx.Tensor, col_index: fx.Tensor, value: fx.Tensor,
               x: fx.Tensor, y: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        slot = tid % LANES_PER_ROW         # lane index inside the row group
        row = bid * rows_per_block + tid // LANES_PER_ROW

        atom_f = vec_copy_atom(1)
        atom_i = vec_copy_atom(1, fx.Int32)
        ro = vec_divide(fx.rocdl.make_buffer_tensor(row_offset), 1)
        ci = vec_divide(fx.rocdl.make_buffer_tensor(col_index), 1)
        va = vec_divide(fx.rocdl.make_buffer_tensor(value), 1)
        xd = vec_divide(fx.rocdl.make_buffer_tensor(x), 1)
        yd = vec_divide(fx.rocdl.make_buffer_tensor(y), 1)

        # Rows past the end read row 0's bounds and are simply not stored; a
        # branch here would split the wavefront before the reduction below.
        in_range = row < rows
        row_safe = in_range.select(row, fx.Int32(0))
        start = load_scalar(ro, row_safe, atom_i, fx.Int32)
        end = load_scalar(ro, row_safe + 1, atom_i, fx.Int32)

        # The cost centre: one dependent gather into x per non-zero, with no
        # locality guarantee. This is why SpMV never approaches peak bandwidth.
        acc = fx.Float32(0.0)
        for jj in range(start + slot, end, LANES_PER_ROW):
            col = load_scalar(ci, jj, atom_i, fx.Int32)
            acc = acc + load_scalar(va, jj, atom_f) * load_scalar(xd, col, atom_f)

        acc = wave_reduce_sum_down(acc, LANES_PER_ROW)
        if in_range:
            if slot == 0:
                store_scalar(acc, yd, row, atom_f)

    @flyc.jit
    def launch(row_offset: fx.Tensor, col_index: fx.Tensor, value: fx.Tensor,
               x: fx.Tensor, y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(row_offset, col_index, value, x, y).launch(
            grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
