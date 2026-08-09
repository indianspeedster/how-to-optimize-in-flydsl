# SPDX-License-Identifier: Apache-2.0
"""Block-wise sum reduction -- the eight-rung optimization ladder.

Ports ``reduce/reduce_v0_baseline.cu`` .. ``reduce_v7_shuffle.cu``. Each block
reduces one contiguous chunk of the input to a single float, so with ``E``
elements per block the output has ``N/E`` entries -- exactly the CUDA original's
``d_out[blockIdx.x]``. The rungs consume different ``E`` (256, then 512 once a
thread adds during load, then a large chunk once it accumulates serially), so
they emit different-length outputs while reading the same ``N`` floats. As in the
original README, the number that compares them is **bytes read per second**.

Two rungs deviate from a literal transcription, both because CDNA is not CUDA:

``v4``/``v5`` -- "unroll the last warp".
    The CUDA trick is that a warp is 32 lanes and executes in lockstep, so the
    tail of the tree needs no ``__syncthreads()``, only ``volatile``. A CDNA
    wavefront is **64** lanes, so the barrier-free tail starts twice as early
    (s <= 64, not s <= 32) and covers one more level. FlyDSL has no ``volatile``
    memref; instead the tail is done in registers through cross-lane shuffles,
    which is strictly better than the LDS round-trip it replaces and is what a
    CDNA programmer would write. The rung still isolates the same effect: the
    tail of the tree stops paying for workgroup barriers.

``v7`` -- "shuffle".
    ``__shfl_down_sync(0xffffffff, v, d)`` becomes ``gpu.shuffle`` in ``down``
    mode over a 64-lane wavefront, so the ladder is 6 steps (32,16,8,4,2,1)
    rather than 5, and the LDS cross-wave array holds 4 partials for a
    256-thread block instead of 8.

``v8`` and ``v9`` have no CUDA counterpart: ``v8`` is ``v7`` with ``float4``
loads, ``v9`` additionally widens the grid from the original's fixed 1024 blocks
to 32 per CU. Both were added expecting a win and **neither delivers one** --
once a thread accumulates serially, the kernel is already at the memory roof and
neither the transaction width nor the grid moves it. They are kept because a
ladder that only shows the steps that worked is not an honest ladder.
"""

# NOTE: deliberately no `from __future__ import annotations`. `@fx.struct`
# resolves its field annotations at class-creation time; PEP 563 turns them into
# strings and the LDS layout can no longer be computed ("Cannot compute layout
# for schema SharedStorage").

import torch

from flyopt.dsl import (
    HAVE_FLYDSL,
    fast_launcher,
    flyc,
    fx,
    gpu,
    const_expr,
    load_scalar,
    load_vec,
    range_constexpr,
    store_scalar,
    vec_copy_atom,
    vec_divide,
    wave_reduce_sum_down,
)
from flyopt.env import wave_size
from flyopt.registry import Op, Shape, Variant, register

THREADS = 256          # 4 wavefronts on CDNA -- the original's THREAD_PER_BLOCK
MULTI_ADD_BLOCKS = 1024  # the original's fixed grid for reduce6 / reduce7
# A 256-CU part wants far more blocks in flight than the V100 the original was
# tuned on: 1024 blocks is 4 per CU, i.e. 4 wavefront-quads, which cannot cover
# HBM latency. 32 blocks per CU is the CDNA4-sized grid the last rung uses.
CDNA_BLOCKS = 32 * 256

# Elements one block consumes, per rung. This is the *only* thing that makes the
# rungs produce different-shaped outputs, and it is the honest port: the CUDA
# files change their grid the same way as the optimization progresses.
_ELEMS_PER_BLOCK = {
    "v0_baseline": THREADS,
    "v1_no_divergence": THREADS,
    "v2_no_bank_conflict": THREADS,
    "v3_add_during_load": THREADS * 2,
    "v4_unroll_last_wave": THREADS * 2,
    "v5_full_unroll": THREADS * 2,
}
# Rungs whose grid is fixed and whose chunk therefore scales with N.
_GRID_BLOCKS = {
    "v6_multi_add": MULTI_ADD_BLOCKS,
    "v7_shuffle": MULTI_ADD_BLOCKS,
    "v8_shuffle_vec4": MULTI_ADD_BLOCKS,
    "v9_vec4_wide_grid": CDNA_BLOCKS,
}


def elems_per_block(variant: str, N: int) -> int:
    if variant in _ELEMS_PER_BLOCK:
        return _ELEMS_PER_BLOCK[variant]
    return N // _GRID_BLOCKS[variant]


def _shared_storage(slots: int):
    """LDS storage struct for ``slots`` f32, 16-byte aligned."""

    @fx.struct
    class SharedStorage:
        s: fx.Array[fx.Float32, slots, 16]

    return SharedStorage


# -- rungs 0-2: one element per thread, LDS tree, three indexing schemes ------


def _build_tree(scheme: str):
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
        Storage = _shared_storage(THREADS)

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


# -- rungs 3-5: add during load, then shorten / unroll the tree ---------------


def _build_halved(tail: str, unroll: bool):
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
        Storage = _shared_storage(THREADS)
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


# -- rungs 6-8: serial accumulation first, then a cheap cross-lane finish -----


def _build_multi_add(finish: str, vec_width: int = 1, blocks: int = MULTI_ADD_BLOCKS,
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
        Storage = _shared_storage(THREADS if finish == "lds" else waves + 1)

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


# -- op registration ---------------------------------------------------------


def _na(*_a, **_k):
    raise RuntimeError("FlyDSL runtime unavailable")


def _g(fn, *a, **k):
    return fn(*a, **k) if HAVE_FLYDSL else (lambda *_a, **_k: _na)


def _make_inputs(*, N: int, variant: str):
    # Small non-negative integers: every partial sum is an exactly representable
    # f32 integer (max 8 * 32768 = 262144 << 2^24), so the check is bit-exact and
    # independent of summation order. That is what lets tol be zero and makes a
    # numerically-different-but-correct rung indistinguishable from a right one.
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(0, 8, (N,), generator=g, device="cuda", dtype=torch.int32).float()
    y = torch.zeros(N // elems_per_block(variant, N), device="cuda", dtype=torch.float32)
    return x, y


def _reference(x, y, *, N: int, variant: str):
    return x.view(-1, elems_per_block(variant, N)).sum(dim=1)


def _metrics(t, *, N: int, variant: str = ""):
    return {"GB/s": N * 4 / t / 1e9}


register(
    Op(
        name="reduce",
        doc="block-wise sum -- the classic 8-rung reduction ladder, on wave64",
        variants=[
            Variant("v0_baseline", "LDS tree, tid % (2s) == 0 -- divergent",
                    _g(_build_tree, "interleaved"),
                    origin="reduce/reduce_v0_baseline.cu", baseline=True),
            Variant("v1_no_divergence", "index = 2*s*tid -- active lanes contiguous",
                    _g(_build_tree, "contiguous"), origin="reduce/reduce_v1_no_divergence_branch.cu"),
            Variant("v2_no_bank_conflict", "s halving, tid < s -- conflict-free LDS",
                    _g(_build_tree, "sequential"), origin="reduce/reduce_v2_no_bank_conflict.cu"),
            Variant("v3_add_during_load", "fold 2 globals per thread before the tree",
                    _g(_build_halved, "lds", False), origin="reduce/reduce_v3_add_during_load.cu"),
            Variant("v4_unroll_last_wave", "last 64 lanes finish in registers, no barrier",
                    _g(_build_halved, "wave", False), origin="reduce/reduce_v4_unroll_last_warp.cu"),
            Variant("v5_full_unroll", "LDS levels emitted straight-line (constexpr)",
                    _g(_build_halved, "wave", True), origin="reduce/reduce_v5_completely_unroll.cu"),
            Variant("v6_multi_add", "serial accumulate N/262144 per thread, then the tree",
                    _g(_build_multi_add, "lds"), origin="reduce/reduce_v6_multi_add.cu"),
            Variant("v7_shuffle", "serial accumulate, then wave shuffle + 4 LDS slots",
                    _g(_build_multi_add, "wave"), origin="reduce/reduce_v7_shuffle.cu"),
            Variant("v8_shuffle_vec4", "v7 with dwordx4 loads",
                    _g(_build_multi_add, "wave", 4),
                    origin="(CDNA4 addition, no CUDA counterpart)"),
            Variant("v9_vec4_wide_grid", f"v8 on a {CDNA_BLOCKS}-block grid (32/CU)",
                    _g(_build_multi_add, "wave", 4, CDNA_BLOCKS),
                    origin="(CDNA4 addition, no CUDA counterpart)"),
        ],
        shapes=[Shape("N=32M", {"N": 32 * 1024 * 1024}),
                Shape("N=256M", {"N": 256 * 1024 * 1024})],
        make_inputs=_make_inputs,
        reference=_reference,
        output_index=1,
        metrics=_metrics,
        torch_baseline=lambda x, y, *, N, variant: torch.sum(
            x.view(-1, elems_per_block(variant, N)), dim=1, out=y),
        tol={"rtol": 0.0, "atol": 0.0},
        per_variant=True,
    )
)
