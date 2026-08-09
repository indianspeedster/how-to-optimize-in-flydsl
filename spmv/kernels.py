# SPDX-License-Identifier: Apache-2.0
"""SpMV (y = A_csr x) -- the "how many lanes per row" study.

Ports ``spmv/spmv.cu``. That file is a single kernel with one knob,
``THREADS_PER_VECTOR``: how many lanes cooperate on one CSR row. The knob is the
whole lesson. Too few and a long row serialises; too many and most lanes sit
idle on a short row and the cross-lane reduction costs more than the row does.
The right value tracks the average non-zeros per row, and the ladder below is
that knob swept, not a sequence of different algorithms.

CDNA changes the arithmetic: the segment widths available are divisors of **64**,
so the useful settings are 1, 2, 4, ... 64 rather than CUDA's 1..32, and the
64-lane setting is a full wavefront per row rather than two warps' worth.

The gather ``x[col_index[jj]]`` is the cost centre -- one dependent load per
non-zero with no locality guarantee -- so SpMV never approaches peak bandwidth,
and the ``skewed`` shape below shows what happens when the rows stop being the
same length.
"""

# No `from __future__ import annotations`.

import torch

from common.dsl import (
    HAVE_FLYDSL,
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
from common.sparse import csr_to_torch, make_csr

THREADS = 256


def _build(lanes_per_row: int):
    """One CSR row per group of ``lanes_per_row`` consecutive lanes."""

    def build(rows: int, cols: int, nnz_per_row: int, pattern: str):
        W = wave_size()
        if lanes_per_row > W or W % lanes_per_row:
            raise ValueError(f"lanes_per_row={lanes_per_row} must divide {W}")
        rows_per_block = THREADS // lanes_per_row
        blocks = (rows + rows_per_block - 1) // rows_per_block

        @flyc.kernel
        def kernel(row_offset: fx.Tensor, col_index: fx.Tensor, value: fx.Tensor,
                   x: fx.Tensor, y: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            slot = tid % lanes_per_row          # lane index inside the row group
            row = bid * rows_per_block + tid // lanes_per_row

            atom_f = vec_copy_atom(1)
            atom_i = vec_copy_atom(1, fx.Int32)
            ro = vec_divide(fx.rocdl.make_buffer_tensor(row_offset), 1)
            ci = vec_divide(fx.rocdl.make_buffer_tensor(col_index), 1)
            va = vec_divide(fx.rocdl.make_buffer_tensor(value), 1)
            xd = vec_divide(fx.rocdl.make_buffer_tensor(x), 1)
            yd = vec_divide(fx.rocdl.make_buffer_tensor(y), 1)

            # Rows past the end read row 0's bounds and are simply not stored;
            # a branch here would split the wavefront before the reduction.
            in_range = row < rows
            row_safe = in_range.select(row, fx.Int32(0))
            start = load_scalar(ro, row_safe, atom_i, fx.Int32)
            end = load_scalar(ro, row_safe + 1, atom_i, fx.Int32)

            acc = fx.Float32(0.0)
            for jj in range(start + slot, end, lanes_per_row):
                col = load_scalar(ci, jj, atom_i, fx.Int32)
                acc = acc + load_scalar(va, jj, atom_f) * load_scalar(xd, col, atom_f)

            acc = wave_reduce_sum_down(acc, lanes_per_row)
            if in_range:
                if slot == 0:
                    store_scalar(acc, yd, row, atom_f)

        @flyc.jit
        def launch(row_offset: fx.Tensor, col_index: fx.Tensor, value: fx.Tensor,
                   x: fx.Tensor, y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            kernel(row_offset, col_index, value, x, y).launch(
                grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

        return fast_launcher(launch)

    return build
