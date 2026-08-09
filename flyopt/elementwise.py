# SPDX-License-Identifier: Apache-2.0
"""Elementwise add -- the vectorized-access study.

Ports ``elementwise/elementwise_add.cu``. The whole point of that file is a
single axis: how wide is one lane's memory transaction? CUDA expresses it as
``float`` / ``float2`` / ``float4`` reinterpret casts; FlyDSL expresses it as the
copy *atom* -- ``BufferCopy32b`` / ``64b`` / ``128b`` -- which lowers to
``buffer_load_dword`` / ``dwordx2`` / ``dwordx4``. Same axis, named honestly.

C = A + B reads 2N and writes N floats, so it is purely HBM-bound: the metric is
achieved bandwidth against the 8 TB/s peak, exactly as in the original README.

The fourth rung has no CUDA counterpart. On CDNA4 a 256-CU part needs far more
memory-level parallelism in flight than a V100 did, and one dwordx4 per lane does
not supply it; ``v3_float4_x4`` gives each lane four independent float4s so the
loads issue back-to-back without waiting on each other.
"""

from __future__ import annotations

import torch

from flyopt.dsl import (
    HAVE_FLYDSL,
    fast_launcher,
    fx,
    flyc,
    load_vec,
    range_constexpr,
    store_vec,
    vec_copy_atom,
    vec_divide,
)
from flyopt.registry import Op, Shape, Variant, register

THREADS = 256


def _build(vec_width: int, per_thread: int = 1):
    """Build ``C = A + B`` for a lane transaction of ``vec_width`` f32 elements.

    ``per_thread`` independent transactions per lane raise memory-level
    parallelism without changing the transaction width.
    """

    def build(N: int):
        elems_per_block = THREADS * vec_width * per_thread
        if N % elems_per_block:
            raise ValueError(f"N={N} not divisible by {elems_per_block}")
        num_blocks = N // elems_per_block

        @flyc.kernel
        def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x

            # Buffer tensors carry an AMD resource descriptor, so every access
            # below is a buffer_load/store with hardware bounds checking.
            a = vec_divide(fx.rocdl.make_buffer_tensor(A), vec_width)
            b = vec_divide(fx.rocdl.make_buffer_tensor(B), vec_width)
            c = vec_divide(fx.rocdl.make_buffer_tensor(C), vec_width)
            atom = vec_copy_atom(vec_width)

            # Each of the `per_thread` slices is strided by the block's thread
            # count, which keeps every lane's access coalesced within a slice.
            base = bid * (THREADS * per_thread) + tid
            for i in range_constexpr(per_thread):
                idx = base + i * THREADS
                va = load_vec(a, idx, atom, vec_width)
                vb = load_vec(b, idx, atom, vec_width)
                store_vec(va + vb, c, idx, atom, vec_width)

        @flyc.jit
        def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                   stream: fx.Stream = fx.Stream(None)):
            kernel(A, B, C).launch(grid=(num_blocks, 1, 1), block=(THREADS, 1, 1),
                                   stream=stream)

        return fast_launcher(launch)

    return build


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
