# SPDX-License-Identifier: Apache-2.0
"""Correctness for every registered variant, at one small shape per op.

The bench also checks correctness before it times anything, but the bench is
slow (it runs the full shape ladder) and its failure mode is a table row. This
suite is the fast gate: it runs the smallest shape of every op, so a broken
kernel shows up as a named failing test in seconds.

Variants that do not support a shape are skipped, not failed -- a wave-per-row
kernel legitimately refuses N=16.
"""

import pytest
import torch

from common import env, registry

pytestmark = pytest.mark.skipif(not env.flydsl_available(),
                                reason="needs the FlyDSL wheel and a ROCm GPU")


def _small_shape(op):
    """The cheapest shape an op declares -- the one this suite runs."""
    return op.shapes[0]


def _cases():
    for op in registry.all_ops():
        shape = _small_shape(op)
        for variant in op.variants:
            yield pytest.param(op.name, variant.name, id=f"{op.name}-{variant.name}")


@pytest.mark.parametrize("op_name,variant_name", list(_cases()))
def test_variant_matches_reference(op_name, variant_name):
    op = registry.get(op_name)
    variant = op.variant(variant_name)
    shape = _small_shape(op)

    if not variant.supports(**shape.params):
        pytest.skip(f"{variant_name} does not support {shape.label}")

    kw = dict(shape.params)
    if op.per_variant:
        kw["variant"] = variant_name

    inputs = op.make_inputs(**kw)
    want = op.reference(*inputs, **kw)
    out = inputs[op.output_index]
    out.zero_()

    run = variant.build(**shape.params)
    run(*inputs)
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), want.float(),
                               rtol=max(op.tol["rtol"], 1e-7),
                               atol=max(op.tol["atol"], 1e-7))


def test_every_op_has_a_baseline():
    """Speedups are reported against a baseline rung; every ladder needs one."""
    for op in registry.all_ops():
        assert op.baseline_variant() is not None, f"{op.name} has no baseline variant"


def test_variant_origins_are_recorded():
    """Every rung must say which CUDA file it came from, or that it is new."""
    for op in registry.all_ops():
        for v in op.variants:
            assert v.origin, f"{op.name}/{v.name} has no origin"
