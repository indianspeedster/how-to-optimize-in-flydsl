# SPDX-License-Identifier: Apache-2.0
"""Rungs 0-2: one element per thread, an LDS tree, three indexing schemes.

Ports ``reduce_v0_baseline.cu``, ``reduce_v1_no_divergence_branch.cu`` and
``reduce_v2_no_bank_conflict.cu``. All three move the same bytes and do the same
adds; only the choice of which threads stay active changes.
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


def build_tree(scheme: str):
    """v0/v1/v2: the same LDS tree, differing only in which threads stay active.

    ``interleaved``  tid % (2s) == 0        -- divergent inside every wavefront
    ``contiguous``   index = 2*s*tid        -- active lanes packed, but the LDS
                                               stride 2s hits the same bank
    ``sequential``   tid < s, s halving     -- active lanes packed *and* the LDS
                                               access stays conflict-free
    """

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

            fx.memref_store(load_scalar(xd, bid * THREADS + tid, atom), s_data, tid)
            gpu.barrier()

            # A genuine runtime loop (scf.for), matching the CUDA source: the
            # "complete unroll" rung below is what turns it compile-time.
            for step in range(steps):
                if const_expr(scheme == "sequential"):
                    stride = fx.Int32(THREADS // 2) >> step
                else:
                    stride = fx.Int32(1) << step

                if const_expr(scheme == "interleaved"):
                    if tid % (stride * 2) == 0:
                        acc = fx.memref_load(s_data, tid) + fx.memref_load(s_data, tid + stride)
                        fx.memref_store(acc, s_data, tid)
                elif const_expr(scheme == "contiguous"):
                    index = stride * 2 * tid
                    if index < THREADS:
                        acc = fx.memref_load(s_data, index) + fx.memref_load(s_data, index + stride)
                        fx.memref_store(acc, s_data, index)
                else:
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

    return build
