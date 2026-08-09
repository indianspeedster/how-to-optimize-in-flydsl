# SPDX-License-Identifier: Apache-2.0
"""Block-wise sum reduction -- the ten-rung ladder. See README.md in this folder.

One file per rung, as in the CUDA original -- read them in order and each is a
single idea applied to the one before it. This file is the ladder itself: the
problem, the reference, and the rungs in order.
"""

# No `from __future__ import annotations` -- see reduce/_common.py.

import torch

from common.dsl import HAVE_FLYDSL
from common.registry import Op, Shape, Variant, register

from ._common import CDNA_BLOCKS, elems_per_block
from .reduce_v0_baseline import build as build_v0
from .reduce_v1_no_divergence import build as build_v1
from .reduce_v2_no_bank_conflict import build as build_v2
from .reduce_v3_add_during_load import build as build_v3
from .reduce_v4_unroll_last_wave import build as build_v4
from .reduce_v5_full_unroll import build as build_v5
from .reduce_v6_multi_add import build as build_v6
from .reduce_v7_shuffle import build as build_v7
from .reduce_v8_shuffle_vec4 import build as build_v8
from .reduce_v9_vec4_wide_grid import build as build_v9


def _na(*_a, **_k):
    raise RuntimeError("FlyDSL runtime unavailable")


def _g(build):
    """A rung's builder, or a stub that explains the missing runtime."""
    return build if HAVE_FLYDSL else _na


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
                    _g(build_v0),
                    origin="reduce/reduce_v0_baseline.cu", baseline=True),
            Variant("v1_no_divergence", "index = 2*s*tid -- active lanes contiguous",
                    _g(build_v1),
                    origin="reduce/reduce_v1_no_divergence_branch.cu"),
            Variant("v2_no_bank_conflict", "s halving, tid < s -- conflict-free LDS",
                    _g(build_v2),
                    origin="reduce/reduce_v2_no_bank_conflict.cu"),
            Variant("v3_add_during_load", "fold 2 globals per thread before the tree",
                    _g(build_v3),
                    origin="reduce/reduce_v3_add_during_load.cu"),
            Variant("v4_unroll_last_wave", "last 64 lanes finish in registers, no barrier",
                    _g(build_v4),
                    origin="reduce/reduce_v4_unroll_last_warp.cu"),
            Variant("v5_full_unroll", "LDS levels emitted straight-line (constexpr)",
                    _g(build_v5),
                    origin="reduce/reduce_v5_completely_unroll.cu"),
            Variant("v6_multi_add", "serial accumulate N/262144 per thread, then the tree",
                    _g(build_v6), origin="reduce/reduce_v6_multi_add.cu"),
            Variant("v7_shuffle", "serial accumulate, then wave shuffle + 4 LDS slots",
                    _g(build_v7), origin="reduce/reduce_v7_shuffle.cu"),
            Variant("v8_shuffle_vec4", "v7 with dwordx4 loads",
                    _g(build_v8),
                    origin="(CDNA4 addition, no CUDA counterpart)"),
            Variant("v9_vec4_wide_grid", f"v8 on a {CDNA_BLOCKS}-block grid (32/CU)",
                    _g(build_v9),
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
