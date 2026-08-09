# SPDX-License-Identifier: Apache-2.0
"""Rung 2 -- four f32 per lane. Ports the ``vec4_add`` kernel.

``BufferCopy128b`` -> ``buffer_load_dwordx4``: the widest transaction a single
lane can issue, and 128 bytes per 64-lane wavefront request, which is exactly
CDNA4's L1 line size.

The fastest rung, and it matches the vendor library to within a GB/s (5923 vs
5924) -- both are at the same roof.
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

VEC = 4           # f32 moved by one lane transaction
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
