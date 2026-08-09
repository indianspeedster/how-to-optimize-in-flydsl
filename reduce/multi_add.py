# SPDX-License-Identifier: Apache-2.0
"""Rungs 6-9: serial accumulation first, then a cheap cross-lane finish.

Ports ``reduce_v6_multi_add.cu`` and ``reduce_v7_shuffle.cu``; v8 (dwordx4) and
v9 (a CDNA-sized grid) are additions with no CUDA counterpart, kept because
neither delivered the win it was added to test.
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


def build_multi_add(finish: str, vec_width: int = 1, blocks: int = MULTI_ADD_BLOCKS,
                     n_acc: int = 1):
    """v6/v7/v8.

    The decisive change: give each block enough work that the tree stops
    mattering. Each thread serially accumulates ``N/(1024*256)`` elements into a
    register -- pure streaming, perfectly coalesced -- and only then reduces.

    ``finish='lds'``    the v5 tree (LDS + wave tail)              -> v6
    ``finish='wave'``   wave shuffle, then one LDS slot per wave   -> v7, v8
    ``vec_width``       f32 per lane transaction: 1 (dword) or 4 (dwordx4)
    """

    def build(N: int):
        W = wave_size()
        per_block = N // blocks
        if per_block % (THREADS * vec_width):
            raise ValueError(f"N={N}: {per_block} elems/block not divisible by "
                             f"{THREADS * vec_width}")
        per_thread = per_block // (THREADS * vec_width)
        waves = THREADS // W
        # As in _build_halved: run the LDS tree down to stride W so the wave
        # tail starts with exactly one live partial per lane.
        lds_levels = []
        _st = THREADS // 2
        while _st >= W:
            lds_levels.append(_st)
            _st //= 2
        # +1: the write-only sink slot that replaces CUDA's `if (lane == 0)`.
        Storage = shared_storage(THREADS if finish == "lds" else waves + 1)

        @flyc.kernel
        def kernel(X: fx.Tensor, Y: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lds = fx.SharedAllocator().allocate(Storage).peek()

            atom_s = vec_copy_atom(1)
            atom_v = vec_copy_atom(vec_width)
            xd = vec_divide(fx.rocdl.make_buffer_tensor(X), vec_width)
            yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

            # Serial accumulation. Consecutive lanes touch consecutive
            # transactions, and each iteration advances by a whole block, so
            # every wavefront issues one fully coalesced request per step.
            base = bid * (THREADS * per_thread) + tid
            # `n_acc` independent accumulators break the serial f32-add
            # dependency chain: one chain of `per_thread` adds at ~4 cycles each
            # can be longer than the memory it is meant to hide.
            accs = [fx.Float32(0.0) for _ in range_constexpr(n_acc)]
            for i in range_constexpr(per_thread):
                v = load_vec(xd, base + i * THREADS, atom_v, vec_width)
                if const_expr(vec_width == 1):
                    accs[i % n_acc] = accs[i % n_acc] + v[0]
                else:
                    for l in range_constexpr(vec_width):
                        accs[l % n_acc] = accs[l % n_acc] + v[l]
            acc = accs[0]
            for a in range_constexpr(1, n_acc):
                acc = acc + accs[a]

            if const_expr(finish == "lds"):
                s_data = lds.s.view(fx.make_layout(THREADS, 1))
                fx.memref_store(acc, s_data, tid)
                gpu.barrier()
                for level in range_constexpr(len(lds_levels)):
                    st = lds_levels[level]
                    if tid < st:
                        v2 = fx.memref_load(s_data, tid) + fx.memref_load(s_data, tid + st)
                        fx.memref_store(v2, s_data, tid)
                    gpu.barrier()
                # Predicated, not branched -- see the note in _build_halved.
                live = tid < W
                v3 = fx.memref_load(s_data, live.select(tid, fx.Int32(0)))
                v3 = wave_reduce_sum_down(live.select(v3, fx.Float32(0.0)), W)
                if tid == 0:
                    store_scalar(v3, yd, bid, atom_s)
            else:
                s_data = lds.s.view(fx.make_layout(waves + 1, 1))
                lane = tid % W
                wave = tid // W
                acc = wave_reduce_sum_down(acc, W)
                if lane == 0:
                    fx.memref_store(acc, s_data, wave)
                gpu.barrier()
                # `waves` (4) partials: one thread folds them. Cheaper than a
                # second wave reduction, and it is 3 adds.
                if tid == 0:
                    total = fx.memref_load(s_data, 0)
                    for w in range_constexpr(1, waves):
                        total = total + fx.memref_load(s_data, w)
                    store_scalar(total, yd, bid, atom_s)

        @flyc.jit
        def launch(X: fx.Tensor, Y: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            kernel(X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

        return fast_launcher(launch)

    return build
