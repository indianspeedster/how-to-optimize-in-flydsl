# SPDX-License-Identifier: Apache-2.0
"""spmv -- the ladder: the problem, the reference, and the rungs in order.

The rung builders live in ``kernels.py``; the chapter is ``README.md``.
"""

# No `from __future__ import annotations` -- see kernels.py.

import torch

from common.dsl import HAVE_FLYDSL
from common.registry import Op, Shape, Variant, register
from common.sparse import csr_to_torch, make_csr
from .kernels import _build, THREADS


# -- op registration ---------------------------------------------------------


def _na(*_a, **_k):
    raise RuntimeError("FlyDSL runtime unavailable")


def _g(fn, *a):
    return fn(*a) if HAVE_FLYDSL else (lambda *_a, **_k: _na)


def _make_inputs(*, rows, cols, nnz_per_row, pattern):
    ro, ci, va = make_csr(rows, cols, nnz_per_row, pattern)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(cols, generator=g, device="cuda", dtype=torch.float32)
    y = torch.zeros(rows, device="cuda", dtype=torch.float32)
    return ro, ci, va, x, y


def _reference(ro, ci, va, x, y, *, rows, cols, nnz_per_row, pattern):
    A = csr_to_torch(ro, ci, va, rows, cols)
    return (A @ x).float()


def _nnz(ro):
    return int(ro[-1].item())


def _metrics(t, *, rows, cols, nnz_per_row, pattern):
    nnz = rows * nnz_per_row      # exact for uniform, the mean for skewed
    # value + col_index per non-zero, plus the (unpredictable) gather into x.
    return {"GFLOP/s": 2 * nnz / t / 1e9, "GB/s": nnz * 12 / t / 1e9}


register(
    Op(
        name="spmv",
        doc="y = A_csr x  -- sweeping the lanes-per-row knob on a 64-lane wave",
        variants=[
            Variant("v0_thread_per_row", "1 lane per row (the classic scalar CSR kernel)",
                    _g(_build, 1), origin="spmv/spmv.cu (THREADS_PER_VECTOR=1)",
                    baseline=True),
            Variant("v1_4_lanes", "4 lanes per row", _g(_build, 4),
                    origin="spmv/spmv.cu (THREADS_PER_VECTOR=4)"),
            Variant("v2_8_lanes", "8 lanes per row", _g(_build, 8),
                    origin="spmv/spmv.cu (THREADS_PER_VECTOR=8)"),
            Variant("v3_16_lanes", "16 lanes per row", _g(_build, 16),
                    origin="spmv/spmv.cu (THREADS_PER_VECTOR=16)"),
            Variant("v4_wave_per_row", "a full 64-lane wavefront per row",
                    _g(_build, 64), origin="spmv/spmv.cu (THREADS_PER_VECTOR=32 on CUDA)"),
        ],
        shapes=[
            Shape("1M rows, 32 nnz/row", {"rows": 1 << 20, "cols": 1 << 20,
                                          "nnz_per_row": 32, "pattern": "uniform"}),
            Shape("1M rows, 8 nnz/row", {"rows": 1 << 20, "cols": 1 << 20,
                                         "nnz_per_row": 8, "pattern": "uniform"}),
            Shape("1M rows, skewed", {"rows": 1 << 20, "cols": 1 << 20,
                                      "nnz_per_row": 32, "pattern": "skewed"}),
        ],
        make_inputs=_make_inputs,
        reference=_reference,
        output_index=4,
        metrics=_metrics,
        torch_baseline=lambda ro, ci, va, x, y, *, rows, cols, nnz_per_row, pattern:
            csr_to_torch(ro, ci, va, rows, cols) @ x,
        tol={"rtol": 1e-4, "atol": 1e-4},
    )
)
