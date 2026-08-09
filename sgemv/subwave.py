# SPDX-License-Identifier: Apache-2.0
"""v2: several rows share one wavefront, N lanes each.

Ports ``Sgemv_v2.cu`` (the ``N <= 16`` case). CUDA divides a 32-lane warp; this
divides a 64-lane wavefront, so N=16 packs four rows where CUDA packed two.
"""

# No `from __future__ import annotations` -- @fx.struct resolves its field
# annotations eagerly and PEP 563 stringification breaks the LDS layout.

from common.dsl import (
    block_reduce_sum,
    fast_launcher,
    flyc,
    fx,
    load_scalar,
    load_vec,
    range_constexpr,
    store_scalar,
    vec_copy_atom,
    vec_divide,
    wave_reduce_sum_down,
)
from common.env import wave_size

THREADS = 256


def build_subwave():
    """v2: several rows share one wavefront, N lanes each.

    For N < 64 a whole wavefront per row would leave ``64 - N`` lanes idle on
    every step. Instead the wavefront is cut into ``64/N`` segments of N lanes;
    each segment owns a row and each of its lanes owns exactly one column, so
    utilisation is 100% and the cross-lane reduction runs at segment width.
    """

    def build(M: int, N: int):
        W = wave_size()
        if W % N or N > W:
            raise ValueError(f"N={N} must divide the {W}-lane wavefront")
        rows_per_wave = W // N
        rows_per_block = (THREADS // W) * rows_per_wave
        if M % rows_per_block:
            raise ValueError(f"M={M} not a multiple of {rows_per_block}")
        blocks = M // rows_per_block

        @flyc.kernel
        def kernel(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % W
            wave = tid // W
            seg = lane // N            # which row inside this wavefront
            col = lane % N             # this lane's single column
            row = bid * rows_per_block + wave * rows_per_wave + seg

            atom_s = vec_copy_atom(1)
            A_buf = fx.rocdl.make_buffer_tensor(A)
            a_row = vec_divide(fx.slice(A_buf, (row, None)), 1)
            xd = vec_divide(fx.rocdl.make_buffer_tensor(X), 1)
            yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

            acc = load_scalar(a_row, col, atom_s) * load_scalar(xd, col, atom_s)
            # Segment-width shuffle: lanes only exchange inside their own row.
            acc = wave_reduce_sum_down(acc, N)
            if col == 0:
                store_scalar(acc, yd, row, atom_s)

        @flyc.jit
        def launch(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build
