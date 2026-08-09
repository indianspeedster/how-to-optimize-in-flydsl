# SPDX-License-Identifier: Apache-2.0
"""Shared CSR problem generation for the sparse ops.

The CUDA repo reads SuiteSparse ``.mtx`` / ``.smtx`` files from a ``matrix/``
directory that is not in the repository, so the sparse benchmarks there are not
reproducible as shipped. This module generates the matrices instead: same
structure (CSR, f32 values), fixed seed, no downloads. Two patterns, because
they stress completely different things:

``uniform``
    a fixed number of non-zeros per row at random columns. Every row costs the
    same, so load balance is perfect and the kernel is limited purely by the
    gather into ``x``/``B``.

``skewed``
    a power-law row-length distribution (a few very long rows, many short ones),
    which is what real graphs look like. This is where a one-thread-per-row
    kernel falls apart and a lane-group-per-row kernel does not.
"""

from __future__ import annotations

import torch


def make_csr(rows: int, cols: int, nnz_per_row: int, pattern: str = "uniform",
             seed: int = 0):
    """Return ``(row_offset int32[rows+1], col_index int32[nnz], value f32[nnz])``.

    Column indices inside a row are sorted, which is what a real CSR carries and
    what gives the ``x`` gather whatever locality it can get.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if pattern == "uniform":
        counts = torch.full((rows,), nnz_per_row, dtype=torch.int64)
    elif pattern == "skewed":
        # Pareto-ish: mean is held at nnz_per_row so the two patterns move the
        # same number of bytes and stay comparable.
        u = torch.rand(rows, generator=g, dtype=torch.float64).clamp_min(1e-9)
        raw = u.pow(-1.0 / 1.6)
        counts = (raw * (nnz_per_row / raw.mean())).round().clamp(1, cols).to(torch.int64)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    nnz = int(counts.sum())
    row_offset = torch.zeros(rows + 1, dtype=torch.int32)
    row_offset[1:] = counts.cumsum(0).to(torch.int32)

    col = torch.randint(0, cols, (nnz,), generator=g, dtype=torch.int32)
    # Sort each row's columns. Adding the row id * cols makes one global sort
    # equivalent to a per-row sort, which is far cheaper than a Python loop.
    row_id = torch.repeat_interleave(torch.arange(rows, dtype=torch.int64), counts)
    key = row_id * cols + col.to(torch.int64)
    col = (key.sort().values % cols).to(torch.int32)

    value = torch.randn(nnz, generator=g, dtype=torch.float32)
    return (row_offset.cuda(), col.cuda(), value.cuda())


def csr_to_torch(row_offset, col_index, value, rows, cols):
    """A ``torch.sparse_csr`` view of the same matrix, for the vendor baseline."""
    return torch.sparse_csr_tensor(row_offset.to(torch.int32), col_index.to(torch.int32),
                                   value, size=(rows, cols))
