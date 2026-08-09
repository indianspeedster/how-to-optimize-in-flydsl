# SPDX-License-Identifier: Apache-2.0
"""Rung 5 -- unroll the tree completely. Ports
``reduce_v5_completely_unroll.cu``.

Identical to v4 except that the LDS levels are emitted straight-line at compile
time (``range_constexpr``) instead of as a runtime ``scf.for``.

The measured gain over v4 is inside the noise (4021 vs 4001 GB/s). The rung is
kept because it is in the original ladder and because the reason it does nothing
is the interesting part: by here the kernel is bound by LDS traffic and barriers,
not by loop overhead. The next rung attacks the thing that is actually binding.
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

        # Straight-line, not an scf.for: every stride is a compile-time
        # constant, so the address arithmetic folds and the loop bookkeeping
        # disappears.
        for level in range_constexpr(len(lds_strides)):
            s = lds_strides[level]
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
