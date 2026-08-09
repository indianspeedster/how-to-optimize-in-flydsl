# SPDX-License-Identifier: Apache-2.0
"""Kernel timing.

Measurement policy, and why:

* **CUDA/HIP events, not wall clock.** Launch overhead and host-side JIT
  bookkeeping are not what we are measuring.
* **Median of several rounds, each an average over ``iters`` launches.** A single
  launch on a 256-CU part is dominated by ramp-up; the average absorbs that, and
  the median across rounds absorbs a clock/DVFS excursion. Reporting the *min*
  would flatter every kernel equally but hides variance, so we keep both.
* **Warm up until the JIT has compiled and the clocks have settled** before any
  timed round -- FlyDSL compiles on first call, which would otherwise land
  entirely inside round 0.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch


@dataclass
class Timing:
    seconds: float          # median round
    best_seconds: float     # fastest round
    spread: float           # (max - min) / median, a variance smell test
    iters: int
    rounds: int


def time_callable(fn, *, iters: int = 50, rounds: int = 5, warmup: int = 10) -> Timing:
    """Time ``fn()`` (already bound to its arguments) with HIP events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_round = []
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        per_round.append(start.elapsed_time(end) / 1e3 / iters)

    med = statistics.median(per_round)
    return Timing(
        seconds=med,
        best_seconds=min(per_round),
        spread=(max(per_round) - min(per_round)) / med if med > 0 else 0.0,
        iters=iters,
        rounds=rounds,
    )


def autoscale_iters(fn, target_seconds: float = 0.05, cap: int = 2000) -> int:
    """Pick an iteration count so one timed round runs for ~``target_seconds``.

    A 1 GB elementwise kernel and a 64x64 GEMM differ by four orders of magnitude
    in duration; a fixed iteration count either takes forever on one or measures
    only launch overhead on the other.
    """
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(3):
        fn()
    end.record()
    torch.cuda.synchronize()
    one = start.elapsed_time(end) / 1e3 / 3
    if one <= 0:
        return cap
    return max(3, min(cap, int(target_seconds / one)))
