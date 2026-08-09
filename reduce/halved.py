# SPDX-License-Identifier: Apache-2.0
"""Rungs 3-5: add during load, then shorten and unroll the tree.

Ports ``reduce_v3_add_during_load.cu``, ``reduce_v4_unroll_last_warp.cu`` and
``reduce_v5_completely_unroll.cu``. The CUDA "last warp" is 32 lanes; here it is
a 64-lane wavefront, and the barrier-free tail runs in registers through
shuffles because FlyDSL has no ``volatile`` memref.
"""

# No `from __future__ import annotations` -- see reduce/_common.py.

from common.dsl import (
    HAVE_FLYDSL,
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

from ._common import THREADS, MULTI_ADD_BLOCKS, shared_storage


def build_halved(tail: str, unroll: bool):
    """v3/v4/v5.

    ``tail='lds'``     the whole tree lives in LDS with a barrier per level (v3)
    ``tail='wave'``    levels down to one wavefront use LDS + barriers, the last
                       64 lanes finish in registers via shuffles -- no barriers,
                       no LDS round-trip (v4, v5)
    ``unroll=True``    the LDS levels are emitted straight-line at compile time
                       instead of as an scf.for (v5)
    """
    W = None  # resolved at build time (wave size is a hardware fact)

    def build(N: int):
        nonlocal W
        W = wave_size()
        per_block = THREADS * 2
        blocks = N // per_block
        Storage = shared_storage(THREADS)
        # LDS levels run from THREADS/2 down to `floor`. The wave tail takes
        # over holding ONE partial per lane, so the LDS phase must run down to
        # and including stride W -- that is what leaves exactly W live slots.
        floor = W if tail == "wave" else 1
        lds_strides = []
        st = THREADS // 2
        while st >= floor:
            lds_strides.append(st)
            st //= 2

        @flyc.kernel
        def kernel(X: fx.Tensor, Y: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lds = fx.SharedAllocator().allocate(Storage).peek()
            s_data = lds.s.view(fx.make_layout(THREADS, 1))

            atom = vec_copy_atom(1)
            xd = vec_divide(fx.rocdl.make_buffer_tensor(X), 1)
            yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

            # "Add during load": halve the tree by folding two global elements
            # into one LDS slot, which also halves the LDS traffic.
            base = bid * per_block + tid
            a0 = load_scalar(xd, base, atom)
            a1 = load_scalar(xd, base + THREADS, atom)
            fx.memref_store(a0 + a1, s_data, tid)
            gpu.barrier()

            if const_expr(unroll):
                for stride in range_constexpr(len(lds_strides)):
                    s = lds_strides[stride]
                    if tid < s:
                        acc = fx.memref_load(s_data, tid) + fx.memref_load(s_data, tid + s)
                        fx.memref_store(acc, s_data, tid)
                    gpu.barrier()
            else:
                for step in range(len(lds_strides)):
                    s = fx.Int32(lds_strides[0]) >> step
                    if tid < s:
                        acc = fx.memref_load(s_data, tid) + fx.memref_load(s_data, tid + s)
                        fx.memref_store(acc, s_data, tid)
                    gpu.barrier()

            if const_expr(tail == "wave"):
                # One wavefront owns the remaining W partials. Finish in
                # registers: no barrier, no LDS round-trip, W-lane shift-down.
                #
                # The lane guard is *predicated*, not branched: a cross-lane
                # shuffle placed inside an scf.if region does not survive
                # lowering (lanes outside the region read undefined values), so
                # every thread executes the shuffle and the out-of-range lanes
                # are zeroed instead. See docs/porting-notes.md.
                live = tid < W
                v = fx.memref_load(s_data, live.select(tid, fx.Int32(0)))
                v = live.select(v, fx.Float32(0.0))
                v = wave_reduce_sum_down(v, W)
                if tid == 0:
                    store_scalar(v, yd, bid, atom)
            else:
                if tid == 0:
                    store_scalar(fx.memref_load(s_data, 0), yd, bid, atom)

        @flyc.jit
        def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            kernel(X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

        return fast_launcher(launch)

    return build
