# SPDX-License-Identifier: Apache-2.0
"""Rung 1 -- remove the divergence. Ports ``reduce_v1_no_divergence_branch.cu``.

Same tree, same number of adds. The only change is *which* thread does each add:
instead of the strided ``tid % (2*s) == 0``, thread ``tid`` works on element
``2*s*tid``, so the live lanes are the low ones and whole wavefronts retire
early instead of every wavefront running half-masked.

The remaining problem is the LDS access pattern: a stride of ``2*s`` makes the
live lanes land on the same banks.
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
            # Active lanes packed at the bottom -- but stride 2*s means the
            # LDS addresses they touch collide on banks.
            index = stride * 2 * tid
            if index < THREADS:
                acc = fx.memref_load(s_data, index) + fx.memref_load(s_data, index + stride)
                fx.memref_store(acc, s_data, index)
            gpu.barrier()

        if tid == 0:
            store_scalar(fx.memref_load(s_data, 0), yd, bid, atom)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
