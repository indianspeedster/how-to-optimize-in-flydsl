# SPDX-License-Identifier: Apache-2.0
"""Rung 1 -- two f32 per lane. Ports the ``vec2_add`` kernel.

CUDA reaches this with a ``float2`` reinterpret cast. FlyDSL reaches it by
choosing a wider copy *atom*: ``BufferCopy64b``, which lowers to
``buffer_load_dwordx2``. Same axis, named honestly.

Half the instructions for the same bytes, and 1.07x the bandwidth.
"""

# No `from __future__ import annotations` -- FlyDSL resolves annotations eagerly.

from common.dsl import (
    fast_launcher,
    flyc,
    fx,
    load_vec,
    range_constexpr,
    store_vec,
    vec_copy_atom,
    vec_divide,
)

THREADS = 256

VEC = 2           # f32 moved by one lane transaction
PER_THREAD = 1    # independent transactions each lane issues


def build(N: int):
    elems_per_block = THREADS * VEC * PER_THREAD
    if N % elems_per_block:
        raise ValueError(f"N={N} not divisible by {elems_per_block}")
    num_blocks = N // elems_per_block

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        # Buffer tensors carry an AMD resource descriptor, so every access below
        # is a buffer_load/store with hardware bounds checking.
        a = vec_divide(fx.rocdl.make_buffer_tensor(A), VEC)
        b = vec_divide(fx.rocdl.make_buffer_tensor(B), VEC)
        c = vec_divide(fx.rocdl.make_buffer_tensor(C), VEC)
        atom = vec_copy_atom(VEC)

        base = bid * (THREADS * PER_THREAD) + tid
        for i in range_constexpr(PER_THREAD):
            idx = base + i * THREADS
            va = load_vec(a, idx, atom, VEC)
            vb = load_vec(b, idx, atom, VEC)
            store_vec(va + vb, c, idx, atom, VEC)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        kernel(A, B, C).launch(grid=(num_blocks, 1, 1), block=(THREADS, 1, 1),
                               stream=stream)

    return fast_launcher(launch)
