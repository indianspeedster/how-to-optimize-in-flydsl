# SPDX-License-Identifier: Apache-2.0
"""``python -m bench [op ...]`` -- run the ladders and print the verdict table.

Every rung is checked for correctness *before* it is timed. A wrong kernel has no
speed: it is reported ``FAIL`` and its timing is suppressed, so a broken
optimization can never look like a win.

    python -m bench                     # every op, every shape
    python -m bench reduce sgemm        # named ops
    python -m bench sgemm --shape 4096  # one shape (substring match on its label)
    python -m bench reduce --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

import torch

from bench.timing import autoscale_iters, time_callable
from common import env, registry

# The functools.lru_cache-free build cache: FlyDSL specialises on shape, so one
# compiled callable per (variant, shape).
_built: dict[tuple, object] = {}


def _build(variant, shape):
    key = (id(variant), tuple(sorted(shape.params.items())))
    if key not in _built:
        _built[key] = variant.build(**shape.params)
    return _built[key]


def _check(got, want, tol) -> tuple[bool, float]:
    got32, want32 = got.float(), want.float()
    diff = (got32 - want32).abs()
    denom = want32.abs().clamp_min(1e-30)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    ok = bool(torch.allclose(got32, want32, rtol=tol["rtol"], atol=tol["atol"]))
    return ok, max_rel if tol["rtol"] > 0 else max_abs


def run_op(op, shape_filter=None, variant_filter=None, iters=None, quiet=False):
    results = []
    shapes = [s for s in op.shapes if not shape_filter or shape_filter in s.label]
    if not shapes:
        raise SystemExit(f"{op.name}: no shape matching {shape_filter!r} "
                         f"(have {[s.label for s in op.shapes]})")

    for shape in shapes:
        base_time = None
        cached = {}   # only rebuilt per variant when op.per_variant

        for variant in op.variants:
            row = {"op": op.name, "shape": shape.label, "params": shape.params,
                   "variant": variant.name, "summary": variant.summary,
                   "origin": variant.origin}

            if not variant.supports(**shape.params):
                row["status"] = "SKIP"
                row["note"] = "shape unsupported by this variant"
                results.append(row)
                continue

            kw = dict(shape.params)
            if op.per_variant:
                kw["variant"] = variant.name
            ck = tuple(sorted(kw.items()))
            if ck not in cached:
                # Inputs are seeded, so every variant sees identical data.
                inputs = op.make_inputs(**kw)
                want = op.reference(*inputs, **kw)
                # The vendor library on the same problem -- the "cublas column"
                # of the original README, here rocBLAS/PyTorch.
                tt = None
                if op.torch_baseline is not None:
                    try:
                        fn = lambda: op.torch_baseline(*inputs, **kw)  # noqa: E731
                        tt = time_callable(fn, iters=iters or autoscale_iters(fn)).seconds
                    except Exception:
                        tt = None
                cached[ck] = (inputs, want, tt)
            inputs, want, torch_time = cached[ck]
            out = inputs[op.output_index]

            try:
                run = _build(variant, shape)
                out.zero_()
                run(*inputs)
                torch.cuda.synchronize()
                ok, err = _check(out, want, op.tol)
                row["max_err"] = err
                if not ok:
                    row["status"] = "FAIL"
                    results.append(row)
                    if not quiet:
                        _print_row(row, base_time, torch_time)
                    continue

                fn = lambda: run(*inputs)  # noqa: E731
                n = iters or autoscale_iters(fn)
                t = time_callable(fn, iters=n)
                row.update(status="OK", seconds=t.seconds, best_seconds=t.best_seconds,
                           spread=t.spread, iters=t.iters)
                row["metrics"] = op.metrics(t.seconds, **kw)
                if variant.baseline:
                    base_time = t.seconds
                if base_time:
                    row["speedup_vs_baseline"] = base_time / t.seconds
                if torch_time:
                    row["vs_torch"] = torch_time / t.seconds
                    row["torch_seconds"] = torch_time
            except Exception as exc:
                row["status"] = "ERROR"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()

            results.append(row)
            if not quiet:
                _print_row(row, base_time, torch_time)

        if not quiet:
            for ck, (_i, _w, tt) in cached.items():
                if tt is None:
                    continue
                m = op.metrics(tt, **dict(ck))
                head = next(iter(m))
                label = "torch/rocBLAS"
                if op.per_variant:
                    label = f"torch [{dict(ck)['variant']}]"
                print(f"  {label:<22} {tt * 1e6:>10.1f} us  {m[head]:>10.1f} {head}")
    return results


_HEADER_DONE = set()


def _print_row(row, base_time, torch_time):
    key = (row["op"], row["shape"])
    if key not in _HEADER_DONE:
        _HEADER_DONE.add(key)
        print(f"\n{row['op']}  [{row['shape']}]")
        print(f"  {'variant':<22} {'time':>10}  {'metric':>21}  "
              f"{'vs v0':>7} {'vs torch':>8}  {'':<4}")
        print("  " + "-" * 78)
    if row["status"] != "OK":
        note = row.get("error") or row.get("note") or f"max_err={row.get('max_err')}"
        print(f"  {row['variant']:<22} {row['status']:>10}  {note[:60]}")
        return
    m = row["metrics"]
    head = next(iter(m))
    su = row.get("speedup_vs_baseline")
    vt = row.get("vs_torch")
    print(f"  {row['variant']:<22} {row['seconds'] * 1e6:>10.1f} us  "
          f"{m[head]:>10.1f} {head:<10} "
          f"{(f'{su:.2f}x' if su else '-'):>7} {(f'{vt:.2f}x' if vt else '-'):>8}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench")
    p.add_argument("ops", nargs="*", help="ops to run (default: all)")
    p.add_argument("--shape", help="substring filter on the shape label")
    p.add_argument("--variant", help="substring filter on the variant name")
    p.add_argument("--iters", type=int, help="fixed iteration count (default: autoscale)")
    p.add_argument("--json", help="write full results here")
    p.add_argument("--list", action="store_true", help="list ops and variants, then exit")
    args = p.parse_args(argv)

    if args.list:
        for op in registry.all_ops():
            print(f"\n{op.name}: {op.doc}")
            for v in op.variants:
                print(f"  {v.name:<22} {v.summary}")
                if v.origin:
                    print(f"  {'':22} <- {v.origin}")
        return 0

    if not env.flydsl_available():
        print("FlyDSL runtime unavailable (need the wheel + a ROCm GPU).", file=sys.stderr)
        return 2
    print(env.describe())

    ops = [registry.get(n) for n in args.ops] if args.ops else list(registry.all_ops())
    all_rows = []
    for op in ops:
        variants = op.variants
        if args.variant:
            op.variants = [v for v in variants if args.variant in v.name]
        try:
            all_rows += run_op(op, args.shape, args.variant, args.iters)
        finally:
            op.variants = variants

    print()
    n_fail = sum(1 for r in all_rows if r["status"] in ("FAIL", "ERROR"))
    print(f"{len(all_rows)} rows, {n_fail} failed/errored")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": env.describe(), "rows": all_rows}, f, indent=2)
        print(f"wrote {args.json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
