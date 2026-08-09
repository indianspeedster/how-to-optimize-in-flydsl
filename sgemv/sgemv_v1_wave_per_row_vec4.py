# SPDX-License-Identifier: Apache-2.0
"""Rung 1 -- one wavefront per row, float4 per lane. Ports ``sgemv/Sgemv_v1.cu``
(the ``N >= 128`` case).

Identical to v0 except that each lane moves a ``float4`` -- CUDA's
``reinterpret_cast<float4*>``, here a ``BufferCopy128b`` atom lowering to
``buffer_load_dwordx4``.

It does **not** beat v0 (2288 vs 2298 GB/s at N=256). One wavefront per row
already issues enough parallel loads to saturate the path, so the wider
transaction has nothing left to buy -- the same result the elementwise ladder's
last rung shows. Kept because the original has it and because the negative
result is the point.
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
VEC = 4          # f32 moved by one lane transaction




def build(M: int, N: int):
    W = wave_size()
    waves = THREADS // W
    cols_per_step = W * VEC
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

        atom_v = vec_copy_atom(VEC)
        atom_s = vec_copy_atom(1)
        A_buf = fx.rocdl.make_buffer_tensor(A)
        a_row = vec_divide(fx.slice(A_buf, (row, None)), VEC)
        xd = vec_divide(fx.rocdl.make_buffer_tensor(X), VEC)
        yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

        acc = fx.Float32(0.0)
        for s in range_constexpr(steps):
            idx = lane + s * W
            va = load_vec(a_row, idx, atom_v, VEC)
            vx = load_vec(xd, idx, atom_v, VEC)
            prod = va * vx
            for l in range_constexpr(VEC):
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
