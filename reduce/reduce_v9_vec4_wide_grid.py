# SPDX-License-Identifier: Apache-2.0
"""Rung 9 -- v8 on a CDNA-sized grid. **No CUDA counterpart, and no gain.**

The original fixes the grid at 1024 blocks, which was sized for a V100's 80 SMs.
On a 256-CU part that is four blocks per CU -- four wavefront-quads, which
should not be enough to cover HBM latency. So: 32 blocks per CU, an 8192-block
grid.

Also no gain (6825 vs 6940 GB/s). The serial accumulation inside each thread
already supplies all the memory-level parallelism the kernel can use, so more
blocks only add tail effects.

Two rungs, two disproved hypotheses, one conclusion: v7 is the roof.
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

VEC = 4            # f32 per lane transaction: 1 -> dword, 4 -> dwordx4
BLOCKS = CDNA_BLOCKS


def build(N: int):
    W = wave_size()
    per_block = N // BLOCKS
    if per_block % (THREADS * VEC):
        raise ValueError(f"N={N}: {per_block} elems/block not divisible by "
                         f"{THREADS * VEC}")
    per_thread = per_block // (THREADS * VEC)
    waves = THREADS // W
    # LDS tree levels, if this rung uses one: down to stride W so the wavefront
    # tail starts with exactly one live partial per lane.
    lds_levels = []
    _st = THREADS // 2
    while _st >= W:
        lds_levels.append(_st)
        _st //= 2
    # +1: the write-only sink slot that replaces CUDA's `if (lane == 0)` store.
    Storage = shared_storage(waves + 1)

    @flyc.kernel
    def kernel(X: fx.Tensor, Y: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lds_mem = fx.SharedAllocator().allocate(Storage).peek()

        atom_s = vec_copy_atom(1)
        atom_v = vec_copy_atom(VEC)
        xd = vec_divide(fx.rocdl.make_buffer_tensor(X), VEC)
        yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

        # Serial accumulation -- the decisive change in this ladder. Consecutive
        # lanes touch consecutive transactions and each step advances by a whole
        # block, so every wavefront issues one fully coalesced request per step.
        # (Splitting `acc` into several independent accumulators to break the
        # f32 add chain was tried and changed nothing: this is memory bound.)
        base = bid * (THREADS * per_thread) + tid
        acc = fx.Float32(0.0)
        for i in range_constexpr(per_thread):
            v = load_vec(xd, base + i * THREADS, atom_v, VEC)
            for l in range_constexpr(VEC):
                acc = acc + v[l]

        # Finish with one shuffle per wavefront and a 4-slot LDS array. No tree
        # at all: the tree was only ever needed because each thread held one
        # element, and now it holds `per_thread` of them.
        s_data = lds_mem.s.view(fx.make_layout(waves + 1, 1))
        lane = tid % W
        wave = tid // W
        acc = wave_reduce_sum_down(acc, W)
        if lane == 0:
            fx.memref_store(acc, s_data, wave)
        gpu.barrier()
        # `waves` (4) partials: one thread folds them. Cheaper than a second
        # wave reduction -- it is three adds.
        if tid == 0:
            total = fx.memref_load(s_data, 0)
            for w in range_constexpr(1, waves):
                total = total + fx.memref_load(s_data, w)
            store_scalar(total, yd, bid, atom_s)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(X, Y).launch(grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
