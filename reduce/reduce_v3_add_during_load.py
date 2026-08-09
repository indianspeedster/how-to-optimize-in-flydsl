# SPDX-License-Identifier: Apache-2.0
"""Rung 3 -- add during load. Ports ``reduce_v3_add_during_load.cu``.

v0-v2 spend a whole thread on one element, so half the threads go idle after the
first level. Give each thread two elements and add them on the way into LDS: the
same N elements now need half the blocks and one level less of tree. This is the
largest single win in the first half of the ladder (1.9x over v2) and it costs
one line.

The tree itself is still v2's: all the way down in LDS, one barrier per level.
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
    W = wave_size()
    per_block = THREADS * 2            # each thread folds two globals
    blocks = N // per_block
    Storage = shared_storage(THREADS)
    # The tree runs all the way down in LDS: strides THREADS/2 ... 1.
    lds_strides = []
    st = THREADS // 2
    while st >= 1:
        lds_strides.append(st)
        st //= 2

    @flyc.kernel
    def kernel(X: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds_mem = fx.SharedAllocator().allocate(Storage).peek()
        s_data = lds_mem.s.view(fx.make_layout(THREADS, 1))

        atom = vec_copy_atom(1)
        xd = vec_divide(fx.rocdl.make_buffer_tensor(X), 1)
        yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

        # "Add during load": fold two global elements into one LDS slot. That
        # halves the tree AND halves the LDS traffic, for one extra global read
        # that was going to happen anyway.
        base = bid * per_block + tid
        a0 = load_scalar(xd, base, atom)
        a1 = load_scalar(xd, base + THREADS, atom)
        fx.memref_store(a0 + a1, s_data, tid)
        gpu.barrier()

        for step in range(len(lds_strides)):
            s = fx.Int32(lds_strides[0]) >> step
            if tid < s:
                acc = fx.memref_load(s_data, tid) + fx.memref_load(s_data, tid + s)
                fx.memref_store(acc, s_data, tid)
            gpu.barrier()

        if tid == 0:
            store_scalar(fx.memref_load(s_data, 0), yd, bid, atom)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
