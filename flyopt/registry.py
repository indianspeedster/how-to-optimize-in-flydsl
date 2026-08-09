# SPDX-License-Identifier: Apache-2.0
"""The op / variant registry the bench and the tests both drive.

The original CUDA repo is a *ladder*: each file is one optimization step over the
previous one, and the point is the delta between rungs. So the unit here is a
``Variant`` -- one rung -- and an ``Op`` owns the ladder plus everything needed to
run it: how to build inputs, what the right answer is, and how to turn a time
into the number that matters for that op (GB/s for memory-bound, TFLOP/s for
compute-bound).

A variant is a *builder*, not a kernel: FlyDSL specialises on compile-time shape,
so ``build(**shape)`` returns the callable for that shape and the bench caches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class Variant:
    """One rung of an optimization ladder."""

    name: str
    summary: str                     # the optimization this rung adds
    build: Callable[..., Callable]   # build(**shape) -> run(*tensors)
    origin: str = ""                 # the CUDA file this ports
    baseline: bool = False           # the rung speedups are measured against
    # Shapes this variant refuses (e.g. a wave-per-row kernel needs N % 64 == 0).
    supports: Callable[..., bool] = lambda **shape: True


@dataclass(frozen=True)
class Shape:
    """A named problem size. ``params`` is splatted into build/inputs/reference."""

    label: str
    params: dict[str, Any]

    def __str__(self) -> str:
        return self.label


@dataclass
class Op:
    name: str
    doc: str
    variants: list[Variant]
    shapes: list[Shape]
    make_inputs: Callable[..., tuple]      # (**shape) -> tuple of torch tensors
    reference: Callable[..., Any]          # (*inputs, **shape) -> expected output
    # Which positional input the kernel writes into (checked against reference).
    output_index: int
    # (time_s, **shape) -> {"metric name": value}; the first entry is the headline.
    metrics: Callable[..., dict[str, float]]
    # Optional vendor-library comparison, e.g. rocBLAS via torch. (*inputs, **shape)
    torch_baseline: Callable[..., Any] | None = None
    tol: dict[str, float] = field(default_factory=lambda: {"rtol": 1e-4, "atol": 1e-4})
    # When True, make_inputs / reference / torch_baseline / metrics additionally
    # receive ``variant=<name>``. The reduce ladder needs this: the rungs differ
    # in how many elements one block consumes, so they produce different-length
    # partial-sum vectors from the same input -- exactly as in the CUDA original,
    # where the comparison is bytes-read per second, not output shape.
    per_variant: bool = False

    def variant(self, name: str) -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"{self.name}: no variant {name!r} (have {[v.name for v in self.variants]})")

    def baseline_variant(self) -> Variant | None:
        return next((v for v in self.variants if v.baseline), None)


_OPS: dict[str, Op] = {}


def register(op: Op) -> Op:
    if op.name in _OPS:
        raise ValueError(f"duplicate op {op.name!r}")
    _OPS[op.name] = op
    return op


def get(name: str) -> Op:
    _load_all()
    if name not in _OPS:
        raise KeyError(f"unknown op {name!r} (have {sorted(_OPS)})")
    return _OPS[name]


def all_ops() -> Iterable[Op]:
    _load_all()
    # Ladder order matches the original repo's README, not alphabetical.
    order = ["elementwise", "reduce", "sgemv", "sgemm", "spmv", "spmm"]
    return [_OPS[n] for n in order if n in _OPS] + [
        o for n, o in sorted(_OPS.items()) if n not in order
    ]


_loaded = False


def _load_all() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    from flyopt import elementwise, reduce, sgemm, sgemv, spmm, spmv  # noqa: F401
