# SPDX-License-Identifier: Apache-2.0
"""SGEMV (y = A x) -- the ladder. See README.md in this folder.

One file per rung, as in the CUDA original -- read them in order and each is a
single idea applied to the one before it. This file is the ladder itself.
"""

# No `from __future__ import annotations` -- see sgemv/block_per_row.py.

import torch

from common.dsl import HAVE_FLYDSL
from common.env import wave_size
from common.registry import Op, Shape, Variant, register

from .sgemv_v0_wave_per_row import THREADS, build as build_v0
from .sgemv_v1_wave_per_row_vec4 import build as build_v1
from .sgemv_v2_subwave_per_row import build as build_v2
from .sgemv_v3_block_per_row import build as build_v3


def _na(*_a, **_k):
    raise RuntimeError("FlyDSL runtime unavailable")


def _g(build):
    """A rung's builder, or a stub that explains the missing runtime."""
    return build if HAVE_FLYDSL else _na


def _make_inputs(*, M: int, N: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(M, N, generator=g, device="cuda", dtype=torch.float32)
    x = torch.randn(N, generator=g, device="cuda", dtype=torch.float32)
    y = torch.zeros(M, device="cuda", dtype=torch.float32)
    return A, x, y


def _reference(A, x, y, *, M, N):
    return (A.double() @ x.double()).float()


def _metrics(t, *, M, N):
    return {"GB/s": (M * N + N + M) * 4 / t / 1e9, "GFLOP/s": 2 * M * N / t / 1e9}


_W = wave_size() if HAVE_FLYDSL else 64


def _sup_wave(*, M, N, vec=1):
    return N % (_W * vec) == 0 and M % (THREADS // _W) == 0


register(
    Op(
        name="sgemv",
        doc="y = A x -- mapping rows onto 64-lane wavefronts",
        variants=[
            Variant("v0_wave_per_row", "1 wavefront per row, 1 column per lane",
                    _g(build_v0), origin="sgemv/Sgemv_v0.cu",
                    baseline=True, supports=lambda **s: _sup_wave(**s, vec=1)),
            Variant("v1_wave_per_row_vec4", "1 wavefront per row, float4 per lane",
                    _g(build_v1), origin="sgemv/Sgemv_v1.cu",
                    supports=lambda **s: _sup_wave(**s, vec=4)),
            Variant("v2_subwave_per_row", "64/N rows per wavefront, N lanes each",
                    _g(build_v2), origin="sgemv/Sgemv_v2.cu",
                    supports=lambda **s: _W % s["N"] == 0 and s["N"] <= _W
                    and s["M"] % ((THREADS // _W) * (_W // s["N"])) == 0),
            Variant("v3_block_per_row", "1 workgroup per row + LDS block reduce",
                    _g(build_v3),
                    origin="(CDNA4 addition, no CUDA counterpart)",
                    supports=lambda **s: s["N"] % (THREADS * 4) == 0),
        ],
        shapes=[
            Shape("M=16384,N=16", {"M": 16384, "N": 16}),      # the N<=16 case
            Shape("M=16384,N=64", {"M": 16384, "N": 64}),      # one wave per row
            Shape("M=16384,N=256", {"M": 16384, "N": 256}),    # the N>=128 case
            Shape("M=16384,N=4096", {"M": 16384, "N": 4096}),  # long rows
            Shape("M=1024,N=16384", {"M": 1024, "N": 16384}),  # few, very long rows
        ],
        make_inputs=_make_inputs,
        reference=_reference,
        output_index=2,
        metrics=_metrics,
        torch_baseline=lambda A, x, y, *, M, N: torch.mv(A, x, out=y),
        tol={"rtol": 1e-3, "atol": 1e-3},
    )
)
