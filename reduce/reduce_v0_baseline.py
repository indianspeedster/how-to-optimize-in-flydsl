# SPDX-License-Identifier: Apache-2.0
"""Rung 0 -- the baseline. Ports ``reduce/reduce_v0_baseline.cu``.

An LDS tree where the active threads are chosen by ``tid % (2*s) == 0``. On
every level the live lanes are spread across the whole workgroup, so each
64-lane wavefront runs at a fraction of its width and the rest of the lanes are
masked off doing nothing. Every later rung in this folder is an attack on some
consequence of that one line.
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
            stride = fx.Int32(1) << step
            # Divergent: the live lanes are strided across the wavefront.
            if tid % (stride * 2) == 0:
                acc = fx.memref_load(s_data, tid) + fx.memref_load(s_data, tid + stride)
                fx.memref_store(acc, s_data, tid)
            gpu.barrier()

        if tid == 0:
            store_scalar(fx.memref_load(s_data, 0), yd, bid, atom)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
