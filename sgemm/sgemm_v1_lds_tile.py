# SPDX-License-Identifier: Apache-2.0
"""v1: one level of blocking, global -> LDS.

A 16x16x16 tile means each A and B element loaded from global is used 16 times
instead of once. That is the whole first-level blocking argument.
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

TS = 16

def build(M: int, N: int, K: int):
    if K % TS:
        raise ValueError(f"K={K} not a multiple of {TS}")

    @fx.struct
    class Shared:
        a: fx.Array[fx.Float32, TS * TS, 16]
        b: fx.Array[fx.Float32, TS * TS, 16]

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        tx, ty = tid % TS, tid // TS
        m = fx.block_idx.y * TS + ty
        n = fx.block_idx.x * TS + tx

        lds = fx.SharedAllocator().allocate(Shared).peek()
        As = lds.a.view(fx.make_layout((TS, TS), (TS, 1)))
        Bs = lds.b.view(fx.make_layout((TS, TS), (TS, 1)))

        atom = vec_copy_atom(1)
        A_buf = fx.rocdl.make_buffer_tensor(A)
        B_buf = fx.rocdl.make_buffer_tensor(B)
        C_buf = fx.rocdl.make_buffer_tensor(C)
        a_row = vec_divide(fx.slice(A_buf, (m, None)), 1)
        c_row = vec_divide(fx.slice(C_buf, (m, None)), 1)

        acc = fx.Float32(0.0)
        for kt in range(K // TS):
            b_row = vec_divide(fx.slice(B_buf, (kt * TS + ty, None)), 1)
            fx.memref_store(load_scalar(a_row, kt * TS + tx, atom), As, (ty, tx))
            fx.memref_store(load_scalar(b_row, n, atom), Bs, (ty, tx))
            gpu.barrier()
            for k in range_constexpr(TS):
                acc = acc + fx.memref_load(As, (ty, k)) * fx.memref_load(Bs, (k, tx))
            gpu.barrier()
        store_scalar(acc, c_row, n, atom)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        kernel(A, B, C).launch(grid=(N // TS, M // TS, 1), block=(TS * TS, 1, 1),
                               stream=stream)

    return fast_launcher(launch)
