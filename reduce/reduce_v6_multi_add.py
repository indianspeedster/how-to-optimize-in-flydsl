# SPDX-License-Identifier: Apache-2.0
"""Rung 6 -- give each thread real work. Ports ``reduce_v6_multi_add.cu``.

Every rung so far optimised the *tree*. This one makes the tree almost
irrelevant: fix the grid at 1024 blocks and have each thread serially accumulate
``N/(1024*256)`` elements into a register before any tree runs at all. The reads
are pure streaming and perfectly coalesced.

Worth 1.7x over v5 -- more than every tree optimization before it combined. After
this rung the kernel is at the memory roof, and the two that follow prove it.
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

VEC = 1            # f32 per lane transaction: 1 -> dword, 4 -> dwordx4
BLOCKS = MULTI_ADD_BLOCKS


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
    Storage = shared_storage(THREADS)

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
            acc = acc + v[0]

        # Finish with the v5 tree: LDS levels down to stride W, then the
        # barrier-free wavefront tail.
        s_data = lds_mem.s.view(fx.make_layout(THREADS, 1))
        fx.memref_store(acc, s_data, tid)
        gpu.barrier()
        for level in range_constexpr(len(lds_levels)):
            st = lds_levels[level]
            if tid < st:
                v2 = fx.memref_load(s_data, tid) + fx.memref_load(s_data, tid + st)
                fx.memref_store(v2, s_data, tid)
            gpu.barrier()
        # Predicated, not branched -- see docs/porting-notes.md Sec. 2.3.
        live = tid < W
        v3 = fx.memref_load(s_data, live.select(tid, fx.Int32(0)))
        v3 = wave_reduce_sum_down(live.select(v3, fx.Float32(0.0)), W)
        if tid == 0:
            store_scalar(v3, yd, bid, atom_s)

    @flyc.jit
    def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(X, Y).launch(grid=(BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
