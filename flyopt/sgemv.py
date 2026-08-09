# SPDX-License-Identifier: Apache-2.0
"""SGEMV (y = A x) -- the "shape it to the wavefront" study.

Ports ``sgemv/Sgemv_v0.cu`` (N == 32), ``Sgemv_v1.cu`` (N >= 128) and
``Sgemv_v2.cu`` (N <= 16). The original's thesis is stated in its README: the
whole game is mapping rows onto warps so no lane sits idle. That thesis survives
the port unchanged; every *number* in it does not, because a CDNA wavefront is 64
lanes, not 32:

* ``v0`` -- one wavefront per row, one column per lane -- is the ``N == 64`` case
  here, not ``N == 32``.
* ``v2`` -- pack several rows into one wavefront -- divides 64 by N, so N = 16
  puts **4** rows in a wavefront where CUDA put 2.

SGEMV reads M*N floats of A and does 2*M*N flops, i.e. one FMA per 4 bytes: it is
hopelessly memory bound at every size, so the headline metric is bandwidth, and
"as fast as rocBLAS" means "both are at the memory roof".

``v3`` has no CUDA counterpart. When N is large the row is long enough that one
wavefront per row leaves the machine underfilled at small M; giving the whole
256-thread workgroup one row and finishing with an LDS block reduction keeps
every CU busy.
"""

# No `from __future__ import annotations`: @fx.struct resolves its field
# annotations eagerly and PEP 563 stringification breaks the LDS layout.

import torch

from flyopt.dsl import (
    HAVE_FLYDSL,
    block_reduce_sum,
    const_expr,
    fast_launcher,
    flyc,
    fx,
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

THREADS = 256


def _build_wave_per_row(vec_width: int):
    """v0 / v1: one wavefront owns one row; lanes stride across the columns.

    ``vec_width=1`` is the scalar port of Sgemv_v0; ``vec_width=4`` is Sgemv_v1's
    ``float4`` reinterpret cast, which on CDNA is a ``buffer_load_dwordx4``.
    """

    def build(M: int, N: int):
        W = wave_size()
        waves = THREADS // W
        cols_per_step = W * vec_width
        if N % cols_per_step:
            raise ValueError(f"N={N} not a multiple of {cols_per_step}")
        if M % waves:
            raise ValueError(f"M={M} not a multiple of {waves}")
        steps = N // cols_per_step
        blocks = M // waves

        @flyc.kernel
        def kernel(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % W
            wave = tid // W
            row = bid * waves + wave

            atom_v = vec_copy_atom(vec_width)
            atom_s = vec_copy_atom(1)
            A_buf = fx.rocdl.make_buffer_tensor(A)
            a_row = vec_divide(fx.slice(A_buf, (row, None)), vec_width)
            xd = vec_divide(fx.rocdl.make_buffer_tensor(X), vec_width)
            yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

            acc = fx.Float32(0.0)
            for s in range_constexpr(steps):
                idx = lane + s * W
                va = load_vec(a_row, idx, atom_v, vec_width)
                vx = load_vec(xd, idx, atom_v, vec_width)
                prod = va * vx
                for l in range_constexpr(vec_width):
                    acc = acc + prod[l]

            acc = wave_reduce_sum_down(acc, W)
            if lane == 0:
                store_scalar(acc, yd, row, atom_s)

        @flyc.jit
        def launch(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build


def _build_subwave():
    """v2: several rows share one wavefront, N lanes each.

    For N < 64 a whole wavefront per row would leave ``64 - N`` lanes idle on
    every step. Instead the wavefront is cut into ``64/N`` segments of N lanes;
    each segment owns a row and each of its lanes owns exactly one column, so
    utilisation is 100% and the cross-lane reduction runs at segment width.
    """

    def build(M: int, N: int):
        W = wave_size()
        if W % N or N > W:
            raise ValueError(f"N={N} must divide the {W}-lane wavefront")
        rows_per_wave = W // N
        rows_per_block = (THREADS // W) * rows_per_wave
        if M % rows_per_block:
            raise ValueError(f"M={M} not a multiple of {rows_per_block}")
        blocks = M // rows_per_block

        @flyc.kernel
        def kernel(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % W
            wave = tid // W
            seg = lane // N            # which row inside this wavefront
            col = lane % N             # this lane's single column
            row = bid * rows_per_block + wave * rows_per_wave + seg

            atom_s = vec_copy_atom(1)
            A_buf = fx.rocdl.make_buffer_tensor(A)
            a_row = vec_divide(fx.slice(A_buf, (row, None)), 1)
            xd = vec_divide(fx.rocdl.make_buffer_tensor(X), 1)
            yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

            acc = load_scalar(a_row, col, atom_s) * load_scalar(xd, col, atom_s)
            # Segment-width shuffle: lanes only exchange inside their own row.
            acc = wave_reduce_sum_down(acc, N)
            if col == 0:
                store_scalar(acc, yd, row, atom_s)

        @flyc.jit
        def launch(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, X, Y).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build


def _build_block_per_row(vec_width: int = 4):
    """v3: one workgroup per row, LDS block reduction. For long rows / small M."""

    def build(M: int, N: int):
        W = wave_size()
        waves = THREADS // W
        cols_per_step = THREADS * vec_width
        if N % cols_per_step:
            raise ValueError(f"N={N} not a multiple of {cols_per_step}")
        steps = N // cols_per_step

        @fx.struct
        class SharedStorage:
            s: fx.Array[fx.Float32, waves + 1, 16]

        @flyc.kernel
        def kernel(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor):
            tid = fx.thread_idx.x
            row = fx.block_idx.x
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            s_red = lds.s.view(fx.make_layout(waves + 1, 1))

            atom_v = vec_copy_atom(vec_width)
            atom_s = vec_copy_atom(1)
            A_buf = fx.rocdl.make_buffer_tensor(A)
            a_row = vec_divide(fx.slice(A_buf, (row, None)), vec_width)
            xd = vec_divide(fx.rocdl.make_buffer_tensor(X), vec_width)
            yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

            acc = fx.Float32(0.0)
            for s in range_constexpr(steps):
                idx = tid + s * THREADS
                prod = load_vec(a_row, idx, atom_v, vec_width) * \
                    load_vec(xd, idx, atom_v, vec_width)
                for l in range_constexpr(vec_width):
                    acc = acc + prod[l]

            total = block_reduce_sum(acc, s_red, waves, tid, W)
            if tid == 0:
                store_scalar(total, yd, row, atom_s)

        @flyc.jit
        def launch(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, X, Y).launch(grid=(M, 1, 1), block=(THREADS, 1, 1), stream=stream)

        return fast_launcher(launch)

    return build


# ── op registration ─────────────────────────────────────────────────────────


def _na(*_a, **_k):
    raise RuntimeError("FlyDSL runtime unavailable")


def _g(fn, *a):
    return fn(*a) if HAVE_FLYDSL else (lambda *_a, **_k: _na)


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
        doc="y = A x  -- mapping rows onto 64-lane wavefronts",
        variants=[
            Variant("v0_wave_per_row", "1 wavefront per row, 1 column per lane",
                    _g(_build_wave_per_row, 1), origin="sgemv/Sgemv_v0.cu",
                    baseline=True, supports=lambda **s: _sup_wave(**s, vec=1)),
            Variant("v1_wave_per_row_vec4", "1 wavefront per row, float4 per lane",
                    _g(_build_wave_per_row, 4), origin="sgemv/Sgemv_v1.cu",
                    supports=lambda **s: _sup_wave(**s, vec=4)),
            Variant("v2_subwave_per_row", "64/N rows per wavefront, N lanes each",
                    _g(_build_subwave), origin="sgemv/Sgemv_v2.cu",
                    supports=lambda **s: _W % s["N"] == 0 and s["N"] <= _W
                    and s["M"] % ((THREADS // _W) * (_W // s["N"])) == 0),
            Variant("v3_block_per_row", "1 workgroup per row + LDS block reduce",
                    _g(_build_block_per_row, 4),
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
