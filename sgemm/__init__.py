# SPDX-License-Identifier: Apache-2.0
"""SGEMM (C = A B, f32) -- the blocking ladder. See README.md in this folder.

One file per rung, as in the CUDA original -- read them in order and each is a
single idea applied to the one before it. This file is the ladder itself.
"""

# No `from __future__ import annotations` -- see sgemm/naive.py.

import torch

from common.dsl import HAVE_FLYDSL
from common.registry import Op, Shape, Variant, register

from .sgemm_v0_naive import TS, build as build_v0
from .sgemm_v1_lds_tile import build as build_v1
from .sgemm_v2_thread_tile import BM, BK, BN, build as build_v2
from .sgemm_v3_prefetch import build as build_v3
from .sgemm_v4_double_buffer import build as build_v4
from .sgemm_v5_mfma import build as build_v5
from .sgemm_v6_tuned import build as build_v6


def _na(*_a, **_k):
    raise RuntimeError("FlyDSL runtime unavailable")


def _g(build):
    """A rung's builder, or a stub that explains the missing runtime."""
    return build if HAVE_FLYDSL else _na


def _make_inputs(*, M, N, K):
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(M, K, generator=g, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, generator=g, device="cuda", dtype=torch.float32)
    C = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    return A, B, C


def _reference(A, B, C, *, M, N, K):
    return (A.double() @ B.double()).float()


def _metrics(t, *, M, N, K):
    return {"TFLOP/s": 2 * M * N * K / t / 1e12,
            "GB/s": (M * K + K * N + M * N) * 4 / t / 1e9}


register(
    Op(
        name="sgemm",
        doc="C = A B (f32) -- global->LDS->register blocking, then matrix cores",
        variants=[
            Variant("v0_naive", "one thread per C element, all operands from global",
                    _g(build_v0), origin="(baseline; the CUDA repo starts at v1)",
                    baseline=True,
                    supports=lambda **s: s["M"] % TS == 0 and s["N"] % TS == 0),
            Variant("v1_lds_tile", "16x16x16 LDS tile, 1 C element per thread",
                    _g(build_v1), origin="(blocking level 1)",
                    supports=lambda **s: all(s[d] % TS == 0 for d in "MNK")),
            Variant("v2_thread_tile", "128x128x8 tile, 8x8 per thread, float4 + LDS",
                    _g(build_v2), origin="sgemm/sgemm_v1.cu",
                    supports=lambda **s: s["M"] % BM == 0 and s["N"] % BN == 0
                    and s["K"] % BK == 0),
            Variant("v3_prefetch", "v2 + next-tile global prefetch into registers",
                    _g(build_v3), origin="sgemm/sgemm_v3.cu",
                    supports=lambda **s: s["M"] % BM == 0 and s["N"] % BN == 0
                    and s["K"] % BK == 0),
            Variant("v4_double_buffer", "v3 + LDS ping-pong: one barrier per K-tile",
                    _g(build_v4),
                    origin="sgemm/sgemm_v3.cu (ENABLE_DOUBLE_BUFFER)",
                    supports=lambda **s: s["M"] % BM == 0 and s["N"] % BN == 0
                    and s["K"] % (2 * BK) == 0),
            Variant("v5_mfma", "same blocking, v_mfma_f32_16x16x4_f32 matrix cores",
                    _g(build_v5),
                    origin="(CDNA4 answer to the repo's SASS-tuning chapter)",
                    supports=lambda **s: s["M"] % BM == 0 and s["N"] % BN == 0
                    and s["K"] % BK == 0),
            Variant("v6_tuned", "v5 + tile picked from the shape, LDS ping-pong, sched hints",
                    _g(build_v6),
                    origin="(CDNA4 addition, from the /gemm-optimization skill)",
                    supports=lambda **s: s["M"] % 64 == 0 and s["N"] % 64 == 0
                    and s["K"] % 32 == 0),
        ],
        shapes=[Shape("1024^3", {"M": 1024, "N": 1024, "K": 1024}),
                Shape("2048^3", {"M": 2048, "N": 2048, "K": 2048}),
                Shape("4096^3", {"M": 4096, "N": 4096, "K": 4096})],
        make_inputs=_make_inputs,
        reference=_reference,
        output_index=2,
        metrics=_metrics,
        torch_baseline=lambda A, B, C, *, M, N, K: torch.mm(A, B, out=C),
        tol={"rtol": 2e-3, "atol": 2e-3},
    )
)
