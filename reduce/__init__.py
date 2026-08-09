# SPDX-License-Identifier: Apache-2.0
"""Block-wise sum reduction -- the ten-rung ladder. See README.md in this folder.

The rung builders live one family per file, mirroring the CUDA original's
one-file-per-step layout:

    tree.py       v0-v2   one element per thread, LDS tree
    halved.py     v3-v5   add during load, then shorten and unroll the tree
    multi_add.py  v6-v9   serial accumulation, then a cheap cross-lane finish

This file is the ladder itself: the problem, the reference, and the rungs in
order.
"""

# No `from __future__ import annotations` -- see reduce/_common.py.

import torch

from common.dsl import HAVE_FLYDSL
from common.registry import Op, Shape, Variant, register

from ._common import CDNA_BLOCKS, elems_per_block
from .halved import build_halved
from .multi_add import build_multi_add
from .tree import build_tree


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
        doc="block-wise sum -- the classic reduction ladder, on wave64",
        variants=[
            Variant("v0_baseline", "LDS tree, tid % (2s) == 0 -- divergent",
                    _g(build_tree, "interleaved"),
                    origin="reduce/reduce_v0_baseline.cu", baseline=True),
            Variant("v1_no_divergence", "index = 2*s*tid -- active lanes contiguous",
                    _g(build_tree, "contiguous"),
                    origin="reduce/reduce_v1_no_divergence_branch.cu"),
            Variant("v2_no_bank_conflict", "s halving, tid < s -- conflict-free LDS",
                    _g(build_tree, "sequential"),
                    origin="reduce/reduce_v2_no_bank_conflict.cu"),
            Variant("v3_add_during_load", "fold 2 globals per thread before the tree",
                    _g(build_halved, "lds", False),
                    origin="reduce/reduce_v3_add_during_load.cu"),
            Variant("v4_unroll_last_wave", "last 64 lanes finish in registers, no barrier",
                    _g(build_halved, "wave", False),
                    origin="reduce/reduce_v4_unroll_last_warp.cu"),
            Variant("v5_full_unroll", "LDS levels emitted straight-line (constexpr)",
                    _g(build_halved, "wave", True),
                    origin="reduce/reduce_v5_completely_unroll.cu"),
            Variant("v6_multi_add", "serial accumulate N/262144 per thread, then the tree",
                    _g(build_multi_add, "lds"), origin="reduce/reduce_v6_multi_add.cu"),
            Variant("v7_shuffle", "serial accumulate, then wave shuffle + 4 LDS slots",
                    _g(build_multi_add, "wave"), origin="reduce/reduce_v7_shuffle.cu"),
            Variant("v8_shuffle_vec4", "v7 with dwordx4 loads",
                    _g(build_multi_add, "wave", 4),
                    origin="(CDNA4 addition, no CUDA counterpart)"),
            Variant("v9_vec4_wide_grid", f"v8 on a {CDNA_BLOCKS}-block grid (32/CU)",
                    _g(build_multi_add, "wave", 4, CDNA_BLOCKS),
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
