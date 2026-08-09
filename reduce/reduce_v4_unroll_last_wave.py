# SPDX-License-Identifier: Apache-2.0
"""Rung 4 -- stop paying for barriers in the tail. Ports
``reduce_v4_unroll_last_warp.cu``.

Once the live set fits inside a single wavefront the workgroup barrier between
levels buys nothing: those lanes are in lockstep already. CUDA exploits that with
``volatile`` LDS over the last 32 lanes.

Two things change on CDNA. The wavefront is **64** lanes, so the barrier-free
tail starts twice as early and covers one more level. And FlyDSL has no
``volatile`` memref, so rather than emulating warp-synchronous LDS the tail is
done in registers with cross-lane shuffles -- strictly better than the LDS
round-trip it replaces, and what a CDNA programmer would write.
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
    # The LDS phase stops at stride W: that is what leaves exactly one live
    # partial per lane for the wavefront tail below.
    lds_strides = []
    st = THREADS // 2
    while st >= W:
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

        # The last W partials live one per lane of wavefront 0. Finish them in
        # registers: no barrier, no LDS round-trip, W-lane shift-down butterfly.
        #
        # The lane guard is *predicated*, not branched. A cross-lane shuffle
        # placed inside an scf.if region does not survive lowering -- lanes
        # outside the region feed undefined values into the shuffle -- so every
        # thread executes it and the out-of-range lanes carry zero instead.
        # See docs/porting-notes.md Sec. 2.3.
        live = tid < W
        v = fx.memref_load(s_data, live.select(tid, fx.Int32(0)))
        v = live.select(v, fx.Float32(0.0))
        v = wave_reduce_sum_down(v, W)
        if tid == 0:
            store_scalar(v, yd, bid, atom)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
