# SPDX-License-Identifier: Apache-2.0
"""v0 / v1: one wavefront owns one row, lanes stride across the columns.

Ports ``Sgemv_v0.cu`` (scalar, the CUDA ``N == 32`` case -- ``N == 64`` here) and
``Sgemv_v1.cu`` (``float4`` per lane, the ``N >= 128`` case).
"""

# No `from __future__ import annotations` -- @fx.struct resolves its field
# annotations eagerly and PEP 563 stringification breaks the LDS layout.

from common.dsl import (
    block_reduce_sum,
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
from common.env import wave_size

THREADS = 256


def build_wave_per_row(vec_width: int):
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
