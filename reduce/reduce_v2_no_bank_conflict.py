# SPDX-License-Identifier: Apache-2.0
"""Rung 2 -- remove the bank conflicts. Ports ``reduce_v2_no_bank_conflict.cu``.

Run the tree the other way: start at ``s = THREADS/2`` and halve. Thread ``tid``
now always reads ``s_data[tid]`` and ``s_data[tid + s]``, both unit-stride, so
consecutive lanes hit consecutive banks and the access is conflict-free -- while
keeping v1's property that the live lanes are contiguous.
"""

# No `from __future__ import annotations` -- see _common.py.

from common.dsl import (
    const_expr,
    fast_launcher,
    flyc,
    fx,
    gpu,
    load_scalar,
    load_vec,
    range_constexpr,
    store_scalar,
    vec_copy_atom,
    vec_divide,
    wave_reduce_sum_down,
)
from common.env import wave_size

from ._common import CDNA_BLOCKS, MULTI_ADD_BLOCKS, THREADS, shared_storage


def build(N: int):
    blocks = N // THREADS
    steps = 8                      # log2(THREADS)
    Storage = shared_storage(THREADS)

    @flyc.kernel
    def kernel(X: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Storage).peek()
        s_data = lds.s.view(fx.make_layout(THREADS, 1))

        atom = vec_copy_atom(1)
        xd = vec_divide(fx.rocdl.make_buffer_tensor(X), 1)
        yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

        # One element per thread, straight into LDS.
        fx.memref_store(load_scalar(xd, bid * THREADS + tid, atom), s_data, tid)
        gpu.barrier()

        # A genuine runtime loop (scf.for), matching the CUDA source. v5 is the
        # rung that turns this compile-time.
        for step in range(steps):
            stride = fx.Int32(THREADS // 2) >> step
            # Unit stride: consecutive lanes -> consecutive LDS banks.
            if tid < stride:
                acc = fx.memref_load(s_data, tid) + fx.memref_load(s_data, tid + stride)
                fx.memref_store(acc, s_data, tid)
            gpu.barrier()

        if tid == 0:
            store_scalar(fx.memref_load(s_data, 0), yd, bid, atom)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
