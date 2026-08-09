# SPDX-License-Identifier: Apache-2.0
"""spmm -- the ladder: the problem, the reference, and the rungs in order.

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


def _g(fn, *a, **kw):
    return fn(*a, **kw) if HAVE_FLYDSL else (lambda *_a, **_k: _na)


def _make_inputs(*, rows, cols, nnz_per_row, ncols, pattern):
    ro, ci, va = make_csr(rows, cols, nnz_per_row, pattern)
    g = torch.Generator(device="cuda").manual_seed(1)
    B = torch.randn(cols, ncols, generator=g, device="cuda", dtype=torch.float32)
    C = torch.zeros(rows, ncols, device="cuda", dtype=torch.float32)
    return ro, ci, va, B, C


def _reference(ro, ci, va, B, C, *, rows, cols, nnz_per_row, ncols, pattern):
    return (csr_to_torch(ro, ci, va, rows, cols) @ B).float()


def _metrics(t, *, rows, cols, nnz_per_row, ncols, pattern):
    nnz = rows * nnz_per_row
    return {"GFLOP/s": 2 * nnz * ncols / t / 1e9,
            "GB/s": (nnz * ncols * 4 + rows * ncols * 4) / t / 1e9}


register(
    Op(
        name="spmm",
        doc="C = A_csr B (dense B) -- one workgroup per sparse row",
        variants=[
            Variant("v0_scalar", "1 output column per thread, scalar B loads",
                    _g(_build, 1), origin="spmm/spmm.cu:My_spmm_csr_vector_kernel_v0",
                    baseline=True),
            Variant("v1_lds_row", "stage the row's (col,val) pairs in LDS first",
                    _g(_build, 1, stage_lds=True),
                    origin="spmm/spmm.cu:My_spmm_csr_vector_kernel_v1"),
            Variant("v2_vec4", "4 output columns per thread, float4 B loads",
                    _g(_build, 4), origin="(CDNA4 addition, no CUDA counterpart)"),
        ],
        shapes=[
            Shape("4096x4096, 32nnz, n=256", {"rows": 4096, "cols": 4096,
                                              "nnz_per_row": 32, "ncols": 256,
                                              "pattern": "uniform"}),
            Shape("4096x4096, 32nnz, n=1024", {"rows": 4096, "cols": 4096,
                                               "nnz_per_row": 32, "ncols": 1024,
                                               "pattern": "uniform"}),
            Shape("4096x4096, skewed, n=256", {"rows": 4096, "cols": 4096,
                                               "nnz_per_row": 32, "ncols": 256,
                                               "pattern": "skewed"}),
        ],
        make_inputs=_make_inputs,
        reference=_reference,
        output_index=4,
        metrics=_metrics,
        torch_baseline=lambda ro, ci, va, B, C, *, rows, cols, nnz_per_row, ncols,
        pattern: csr_to_torch(ro, ci, va, rows, cols) @ B,
        tol={"rtol": 1e-4, "atol": 1e-4},
    )
)
