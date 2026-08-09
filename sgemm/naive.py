# SPDX-License-Identifier: Apache-2.0
"""v0: no blocking at all -- one thread per C element, every operand from global.

The rung the rest of the ladder is measured against. K loads of A and K of B per
output element with no reuse, so the kernel sits at the L2/HBM roof and the
matrix cores idle entirely.
"""

# No `from __future__ import annotations` -- @fx.struct resolves its field
# annotations eagerly and PEP 563 stringification breaks the LDS layout.

from common.dsl import (
    const_expr,
    fast_launcher,
    flyc,
    fma,
    fx,
    gpu,
    load_scalar,
    load_vec,
    mfma_f32_16x16x4_f32,
    range_constexpr,
    store_scalar,
    vec_copy_atom,
    vec_divide,
)

TS = 16   # v0/v1 use a 16x16 thread block

def build_naive():
    """One thread per C element; every operand comes from global memory.

    K loads of A and K of B per output element, and nothing is reused, so the
    kernel runs at the L2/HBM roof and the matrix cores idle entirely.
    """

    def build(M: int, N: int, K: int):
        @flyc.kernel
        def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            tx, ty = tid % TS, tid // TS
            m = fx.block_idx.y * TS + ty
            n = fx.block_idx.x * TS + tx

            atom = vec_copy_atom(1)
            A_buf = fx.rocdl.make_buffer_tensor(A)
            B_buf = fx.rocdl.make_buffer_tensor(B)
            C_buf = fx.rocdl.make_buffer_tensor(C)
            a_row = vec_divide(fx.slice(A_buf, (m, None)), 1)
            c_row = vec_divide(fx.slice(C_buf, (m, None)), 1)

            acc = fx.Float32(0.0)
            for k in range(K):
                b_row = vec_divide(fx.slice(B_buf, (k, None)), 1)
                acc = acc + load_scalar(a_row, k, atom) * load_scalar(b_row, n, atom)
            store_scalar(acc, c_row, n, atom)

        @flyc.jit
        def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, B, C).launch(grid=(N // TS, M // TS, 1), block=(TS * TS, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build
