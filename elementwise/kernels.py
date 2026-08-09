# SPDX-License-Identifier: Apache-2.0
"""Elementwise add -- the vectorized-access study.

Ports ``elementwise/elementwise_add.cu``. The whole point of that file is a
single axis: how wide is one lane's memory transaction? CUDA expresses it as
``float`` / ``float2`` / ``float4`` reinterpret casts; FlyDSL expresses it as the
copy *atom* -- ``BufferCopy32b`` / ``64b`` / ``128b`` -- which lowers to
``buffer_load_dword`` / ``dwordx2`` / ``dwordx4``. Same axis, named honestly.

C = A + B reads 2N and writes N floats, so it is purely HBM-bound: the metric is
achieved bandwidth against the 8 TB/s peak, exactly as in the original README.

The fourth rung has no CUDA counterpart. On CDNA4 a 256-CU part needs far more
memory-level parallelism in flight than a V100 did, and one dwordx4 per lane does
not supply it; ``v3_float4_x4`` gives each lane four independent float4s so the
loads issue back-to-back without waiting on each other.
"""

from __future__ import annotations

import torch

from common.dsl import (
    HAVE_FLYDSL,
    fast_launcher,
    fx,
    flyc,
    load_vec,
    range_constexpr,
    store_vec,
    vec_copy_atom,
    vec_divide,
)

THREADS = 256


def _build(vec_width: int, per_thread: int = 1):
    """Build ``C = A + B`` for a lane transaction of ``vec_width`` f32 elements.

    ``per_thread`` independent transactions per lane raise memory-level
    parallelism without changing the transaction width.
    """

    def build(N: int):
        elems_per_block = THREADS * vec_width * per_thread
        if N % elems_per_block:
            raise ValueError(f"N={N} not divisible by {elems_per_block}")
        num_blocks = N // elems_per_block

        @flyc.kernel
        def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x

            # Buffer tensors carry an AMD resource descriptor, so every access
            # below is a buffer_load/store with hardware bounds checking.
            a = vec_divide(fx.rocdl.make_buffer_tensor(A), vec_width)
            b = vec_divide(fx.rocdl.make_buffer_tensor(B), vec_width)
            c = vec_divide(fx.rocdl.make_buffer_tensor(C), vec_width)
            atom = vec_copy_atom(vec_width)

            # Each of the `per_thread` slices is strided by the block's thread
            # count, which keeps every lane's access coalesced within a slice.
            base = bid * (THREADS * per_thread) + tid
            for i in range_constexpr(per_thread):
                idx = base + i * THREADS
                va = load_vec(a, idx, atom, vec_width)
                vb = load_vec(b, idx, atom, vec_width)
                store_vec(va + vb, c, idx, atom, vec_width)

        @flyc.jit
        def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, B, C).launch(grid=(num_blocks, 1, 1), block=(THREADS, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build
