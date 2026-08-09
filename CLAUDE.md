# CLAUDE.md

FlyDSL ports of the `How_to_optimize_in_GPU` kernel ladders, targeting AMD
CDNA4 (gfx950 / MI350X). Read `README.md` for what each ladder shows and
`docs/porting-notes.md` before writing any new kernel.

## Environment

There is no installable environment here. Everything runs through the venv that
carries the FlyDSL wheel and the ROCm build of PyTorch:

```bash
export PYTHONPATH=$PWD
PY=/root/flydsl-wgrad-ragged/.venv/bin/python
$PY -m bench            # verify + time every rung
$PY -m pytest tests -q  # correctness only
```

`make bench` / `make test` / `make list` wrap the same commands. The FlyDSL
wheel is **0.2.4** and its IR is version-unstable: ground any API you are unsure
of by grepping the installed package
(`/root/flydsl-wgrad/.venv/lib/python3.12/site-packages/flydsl/`), not by
recollection. `/root/FlyDSL` is a source checkout with `docs/` and `examples/`
that is *close to* but not identical with the wheel -- useful for reading, not
authoritative for API.

## Skills

This port was built with the `flydsl`, `gfx950` and `optimize-kernel` skills.
They are **not** vendored here -- they live outside this repo (on the machine
where it was written, under `flydsl-wgrad-ragged/.claude/skills/`). If you have
them, load `/flydsl` before writing a kernel; it auto-loads `/gfx950`, whose
`isa-index.md` and `isa.xml` are the authority on instruction shapes. Without
them, `common/env.py` carries the handful of silicon facts this repo actually
budgets against.

## Structure

One folder per kernel, mirroring the CUDA original: `elementwise/`, `reduce/`,
`sgemv/`, `sgemm/`, `spmv/`, `spmm/`. Inside each, `README.md` is the chapter,
`__init__.py` is the ladder (problem, reference, metrics, rungs in order), and
every rung gets its own `<op>_vN_<name>.py` exporting a single `build(...)`.
Shared machinery lives in `common/`.

Adding a rung means writing `<op>_vN_<name>.py` and adding a `Variant` to the
`Op` in `<op>/__init__.py`. Rungs are deliberately self-contained even where
that duplicates code: the repo teaches by diff, so a rung that can only be
understood by tracing a shared parameter is a rung that failed to teach.
A `Variant` is a *builder* -- `build(**shape) -> run(*tensors)` -- because FlyDSL
specialises on compile-time shape. The bench and the tests both drive the
registry, so a registered rung is automatically verified, timed, and tested.

Conventions that are not optional:

- **Never `from __future__ import annotations` in a module that declares an
  `@fx.struct`** -- it breaks the LDS layout. Every kernel module says so.
- **Wrap every `@flyc.jit` launcher in `dsl.fast_launcher`.** Without it, host
  dispatch dominates any kernel under ~50 us and looks exactly like a hardware
  roof.
- **`common/dsl.py` stays branch-free.** Helpers there run as plain Python during
  tracing, so a data-dependent `if` would be resolved at compile time. Use
  `.select()`.
- **Use `dsl.fma`, not `a * b + c`,** in any inner loop. The latter does not
  contract and costs 6x in the GEMM.
- **Update the folder's `README.md`** when you add or retune a rung; the table
  there is the chapter, and the top-level README only carries the headline.
- **Every `Variant` needs an `origin`** naming the CUDA file it ports, or saying
  it is a CDNA addition. A test enforces this.

## Reporting results

Rungs that were added and did not help are kept and labelled, not deleted -- see
`reduce` v8/v9, `elementwise` v3, `spmm` v1. If a change does not move the bench
verdict, say so in the docstring and in the README rather than quietly dropping
it. Correctness is checked before timing, always; a `FAIL` row never gets a time.
