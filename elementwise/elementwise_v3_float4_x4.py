# SPDX-License-Identifier: Apache-2.0
"""Rung 3 -- four independent float4s per lane. **No CUDA counterpart, and no
gain.**

The hypothesis: a 256-CU part needs far more memory-level parallelism in flight
than a V100 did, and one dwordx4 per lane may not supply it. So issue four
independent float4 loads per lane, which the compiler can put in flight together.

It is *slower* than v2 (5574 vs 5923 GB/s). Once the transaction is already 128
bits wide the kernel is at the memory roof; extra loads in flight only cost
registers and lengthen the tail.

Kept because a ladder that shows only the steps that worked is not a ladder.
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
PER_THREAD = 4    # independent transactions each lane issues


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
