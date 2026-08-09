# SPDX-License-Identifier: Apache-2.0
"""v3: one workgroup per row, finished with an LDS block reduction.

No CUDA counterpart. When N is large a single wavefront per row leaves a 256-CU
machine underfilled at small M; a whole workgroup per row does not.
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
    cols_per_step = THREADS * VEC
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

        atom_v = vec_copy_atom(VEC)
        atom_s = vec_copy_atom(1)
        A_buf = fx.rocdl.make_buffer_tensor(A)
        a_row = vec_divide(fx.slice(A_buf, (row, None)), VEC)
        xd = vec_divide(fx.rocdl.make_buffer_tensor(X), VEC)
        yd = vec_divide(fx.rocdl.make_buffer_tensor(Y), 1)

        acc = fx.Float32(0.0)
        for s in range_constexpr(steps):
            idx = tid + s * THREADS
            prod = load_vec(a_row, idx, atom_v, VEC) * \
                load_vec(xd, idx, atom_v, VEC)
            for l in range_constexpr(VEC):
                acc = acc + prod[l]

        total = block_reduce_sum(acc, s_red, waves, tid, W)
        if tid == 0:
            store_scalar(total, yd, row, atom_s)

    @flyc.jit
    def launch(A: fx.Tensor, X: fx.Tensor, Y: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        kernel(A, X, Y).launch(grid=(M, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return fast_launcher(launch)
