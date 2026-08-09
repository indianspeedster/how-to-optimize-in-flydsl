# SPDX-License-Identifier: Apache-2.0
"""elementwise -- the ladder: the problem, the reference, and the rungs in order.

The rung builders live in ``kernels.py``; the chapter is ``README.md``.
"""

# No `from __future__ import annotations` -- see kernels.py.

import torch

from common.dsl import HAVE_FLYDSL
from common.registry import Op, Shape, Variant, register
from .kernels import _build


def _unavailable(*_a, **_k):
    raise RuntimeError("FlyDSL runtime unavailable")


_b = _build if HAVE_FLYDSL else (lambda *a, **k: _unavailable)


def _make_inputs(N: int):
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c = torch.zeros(N, device="cuda", dtype=torch.float32)
    return a, b, c


def _reference(a, b, c, *, N):
    return a + b


def _metrics(t, *, N):
    # 2 reads + 1 write of f32.
    return {"GB/s": 3 * N * 4 / t / 1e9}


register(
    Op(
        name="elementwise",
        doc="C = A + B  -- vectorized global access (float / float2 / float4)",
        variants=[
            Variant("v0_float", "one f32 per lane (buffer_load_dword)",
                    _b(1), origin="elementwise/elementwise_add.cu:add", baseline=True),
            Variant("v1_float2", "two f32 per lane (buffer_load_dwordx2)",
                    _b(2), origin="elementwise/elementwise_add.cu:vec2_add"),
            Variant("v2_float4", "four f32 per lane (buffer_load_dwordx4)",
                    _b(4), origin="elementwise/elementwise_add.cu:vec4_add"),
            Variant("v3_float4_x4", "4x float4 per lane -- more loads in flight",
                    _b(4, per_thread=4), origin="(CDNA4 addition, no CUDA counterpart)"),
        ],
        shapes=[Shape("N=32M", {"N": 32 * 1024 * 1024}),
                Shape("N=256M", {"N": 256 * 1024 * 1024})],
        make_inputs=_make_inputs,
        reference=_reference,
        output_index=2,
        metrics=_metrics,
        torch_baseline=lambda a, b, c, *, N: torch.add(a, b, out=c),
        tol={"rtol": 0.0, "atol": 0.0},   # bit-exact: it is a single add
    )
)
