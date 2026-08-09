# SPDX-License-Identifier: Apache-2.0
"""Shared FlyDSL device-side helpers.

Two rules govern what may live in this file, both consequences of how FlyDSL
compiles:

1. **Module-level imports only.** FlyDSL's AST rewriter counts a kernel's
   free variables; importing FlyDSL symbols *inside* the function that builds a
   kernel breaks the rewrite. Every FlyDSL name this repo uses is bound here, at
   module scope, behind the availability guard.

2. **No data-dependent control flow in a helper.** The rewriter only transforms
   the AST of the ``@flyc.kernel`` function itself. A helper called from a kernel
   body is executed as ordinary Python during tracing, so an ``if`` on a runtime
   value would be evaluated by CPython (wrong) instead of lowered to ``scf.if``.
   The helpers below are therefore branch-free: they use ``.select()`` and a
   scratch LDS slot where a CUDA kernel would write ``if (lane == 0)``.

That second rule is why :func:`block_reduce_sum` looks different from the CUDA
original it ports -- see ``docs/porting-notes.md``.
"""

from __future__ import annotations

from flyopt.env import flydsl_available

HAVE_FLYDSL = flydsl_available()

if HAVE_FLYDSL:
    import flydsl.compiler as flyc  # noqa: F401
    import flydsl.expr as fx
    from flydsl._mlir.dialects import rocdl as _rocdl_dial
    from flydsl._mlir.dialects.gpu import ShuffleOp
    from flydsl.expr.typing import Vector as Vec
    from flydsl.expr import arith, const_expr, gpu, range_constexpr  # noqa: F401
    from flydsl.expr import math as fmath  # noqa: F401
    from flydsl.expr.vector import ReductionOp, full  # noqa: F401

    FASTMATH = arith.FastMathFlags.fast

    def fma(a, b, c):
        """``a * b + c`` as one instruction.

        Written as ``a * b + c`` the compiler emits a separate ``v_pk_mul_f32``
        and ``v_pk_add_f32``: IEEE f32 forbids contracting them without an
        explicit licence, so a GEMM inner loop silently costs twice the
        instructions it should. ``math.fma`` states the intent directly.
        """
        return fmath.fma(a, b, c)

    # ── launch ──────────────────────────────────────────────────────────────

    _NO_CF = object()

    def fast_launcher(launch_fn):
        """Wrap a ``@flyc.jit`` launcher so repeat calls skip host dispatch.

        A bare ``launch(*args)`` re-does ``inspect.Signature.bind``, protocol
        introspection and DLPack resolution on *every* call -- tens of
        microseconds of host work per launch. That is invisible for a 70 us
        kernel and completely dominant for a 25 us one: it silently pins every
        short kernel in this repo to the same wall-clock floor, which looks
        exactly like a bandwidth roof and is not one.

        ``flyc.compile`` pre-resolves all of it once and returns a callable that
        only updates ctypes slots. Tensors and runtime scalars are slots, so one
        compiled function serves every later call with the same signature.

        Note: ``flyc.compile`` *executes* the kernel once while warming up. Every
        kernel in this repo writes its whole output, so the extra run is
        idempotent -- an accumulating (atomic) epilogue would need a scratch
        output for the warm-up instead.
        """
        state = {"cf": _NO_CF}

        def call(*args):
            import torch

            # The compiled dispatcher is positional and *not* variadic: every
            # parameter of the launcher, the stream included, must be supplied.
            full = (*args, torch.cuda.current_stream())
            if state["cf"] is _NO_CF:
                try:
                    state["cf"] = flyc.compile(launch_fn, *full)
                except Exception:
                    state["cf"] = None      # version skew: fall back, stay correct
            cf = state["cf"]
            return cf(*full) if cf is not None else launch_fn(*args)

        call.launch_fn = launch_fn
        return call

    # ── vectorized global access ────────────────────────────────────────────
    #
    # A "vec_width" of V f32 elements is one buffer_load_dwordxV per lane. The
    # copy atom must match the transaction width exactly: 1 -> 32b, 2 -> 64b,
    # 4 -> 128b. This is the CUDA float / float2 / float4 axis, one to one.
    _COPY_BY_WIDTH = {1: "BufferCopy32b", 2: "BufferCopy64b", 4: "BufferCopy128b"}

    def vec_copy_atom(vec_width: int, dtype=None):
        """Copy atom moving ``vec_width`` f32 elements (32/64/128 bit) per lane."""
        if vec_width not in _COPY_BY_WIDTH:
            raise ValueError(f"vec_width must be 1, 2 or 4 (got {vec_width})")
        dtype = fx.Float32 if dtype is None else dtype
        return fx.make_copy_atom(getattr(fx.rocdl, _COPY_BY_WIDTH[vec_width])(), dtype)

    def vec_divide(buf_tensor, vec_width: int):
        """View a 1-D buffer tensor as (vec_width, n/vec_width) for atom slicing."""
        return fx.logical_divide(buf_tensor, fx.make_layout(vec_width, 1))

    def load_vec(divided, index, copy_atom, vec_width: int, dtype=None):
        """One vectorized load: returns an SSA vector of ``vec_width`` elements."""
        dtype = fx.Float32 if dtype is None else dtype
        r = fx.make_rmem_tensor(vec_width, dtype)
        fx.copy_atom_call(copy_atom, fx.slice(divided, (None, index)), r)
        return fx.memref_load_vec(r)

    def store_vec(value, divided, index, copy_atom, vec_width: int, dtype=None):
        """One vectorized store of an SSA vector of ``vec_width`` elements."""
        dtype = fx.Float32 if dtype is None else dtype
        r = fx.make_rmem_tensor(vec_width, dtype)
        fx.memref_store_vec(value, r)
        fx.copy_atom_call(copy_atom, r, fx.slice(divided, (None, index)))

    def load_scalar(divided, index, copy_atom, dtype=None):
        """One scalar load out of a ``vec_divide(..., 1)`` view."""
        dtype = fx.Float32 if dtype is None else dtype
        r = fx.make_rmem_tensor(1, dtype)
        fx.copy_atom_call(copy_atom, fx.slice(divided, (None, index)), r)
        return fx.memref_load_vec(r)[0]

    def store_scalar(value, divided, index, copy_atom, dtype=None):
        """One scalar store into a ``vec_divide(..., 1)`` view."""
        dtype = fx.Float32 if dtype is None else dtype
        r = fx.make_rmem_tensor(1, dtype)
        fx.memref_store_vec(full(1, dtype(value), dtype), r)
        fx.copy_atom_call(copy_atom, r, fx.slice(divided, (None, index)))

    # ── matrix cores ────────────────────────────────────────────────────────

    def mfma_f32_16x16x4_f32(a, b, c):
        """``v_mfma_f32_16x16x4_f32``: D(16x16) = A(16x4) B(4x16) + C, one wave.

        ``a`` and ``b`` are one f32 per lane, ``c``/result a ``vector<4xf32>``.
        The lane->element mapping is fixed by the hardware and is documented (and
        empirically confirmed) at the call site in flyopt/sgemm.py.
        """
        op = _rocdl_dial.mfma_f32_16x16x4f32(
            Vec.make_type(4, fx.Float32),
            arith._to_raw(a), arith._to_raw(b), arith._to_raw(c), 0, 0, 0)
        return Vec(op.result if hasattr(op, "result") else op, 4, fx.Float32)

    # ── cross-lane reduction ────────────────────────────────────────────────

    def shuffle_down(value, delta: int, width: int):
        """``__shfl_down_sync`` equivalent (gpu.shuffle, mode="down").

        Kept alongside the XOR butterfly because the CUDA originals use
        shfl_down: only lane 0 ends up holding the reduced value, which is
        exactly what those kernels rely on.
        """
        raw = arith._to_raw(value)
        off = arith._to_raw(fx.Int32(delta))
        wid = arith._to_raw(fx.Int32(width))
        return fx.Float32(ShuffleOp(raw, off, wid, mode="down").shuffleResult)

    def wave_reduce_sum(value, width: int | None = None):
        """Butterfly (XOR) sum across ``width`` lanes; *every* lane gets the total.

        ``width`` defaults to the native wavefront (64 on CDNA). Partial-wave
        widths (2, 4, ... 32) are legal and are how the sgemv "many short rows"
        variant packs several rows into one wavefront.
        """
        from flyopt.env import wave_size

        width = wave_size() if width is None else width
        w = value
        off = width // 2
        while off >= 1:
            w = w.addf(w.shuffle_xor(off, width), fastmath=FASTMATH)
            off //= 2
        return w

    def wave_reduce_sum_down(value, width: int | None = None):
        """Shift-down sum across ``width`` lanes; only lane 0's result is valid.

        The literal port of ``warpReduceSum`` from the CUDA repo. Same
        instruction count as the butterfly, but the other lanes hold garbage --
        use it only where a ``lane == 0`` guard follows.
        """
        from flyopt.env import wave_size

        width = wave_size() if width is None else width
        w = value
        off = width // 2
        while off >= 1:
            w = w.addf(shuffle_down(w, off, width), fastmath=FASTMATH)
            off //= 2
        return w

    def block_reduce_sum(value, s_red, red_slots: int, tid, wave_width: int | None = None):
        """Sum across the whole workgroup via LDS; every thread gets the total.

        ``s_red`` must be a view of at least ``red_slots + 1`` f32 slots: the
        extra slot is a write-only sink. Instead of the CUDA ``if (lane == 0)``
        guard -- which this file may not contain (see the module docstring) --
        every lane stores, and lanes other than lane 0 are steered into the sink.
        Concurrent writes to the sink race, but nothing ever reads it.

        The final combine is a register-level unrolled sum over ``red_slots``
        (<= 8 for a 512-thread block), so there is no second barrier and no
        second wave reduction.
        """
        from flyopt.env import wave_size

        wave_width = wave_size() if wave_width is None else wave_width
        w = wave_reduce_sum(value, wave_width)
        lane = tid % wave_width
        wave = tid // wave_width
        slot = (lane == 0).select(wave, fx.Int32(red_slots))
        fx.memref_store(w, s_red, slot)
        gpu.barrier()
        total = fx.memref_load(s_red, 0)
        for i in range_constexpr(1, red_slots):
            total = total.addf(fx.memref_load(s_red, i), fastmath=FASTMATH)
        return total
