# SPDX-License-Identifier: Apache-2.0
"""Runtime/hardware detection and the silicon constants the kernels budget against.

Everything that is a *fact about the machine* lives here, so the kernels and the
bench never hard-code a number twice. The peak figures come from the gfx950
skill's ``microarch.md`` (AMD CDNA4 whitepaper PID#2258402-C), not from
measurement -- they are the denominators the bench reports efficiency against.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    """The subset of silicon facts a kernel in this repo actually budgets against."""

    name: str            # gcnArchName, e.g. "gfx950"
    wave_size: int       # lanes per wavefront (64 on CDNA, 32 on RDNA/CUDA warp)
    lds_bytes: int       # LDS per CU -- the hard tile cliff
    cus: int             # compute units
    clock_ghz: float     # peak engine clock
    hbm_gbps: float      # peak HBM bandwidth, GB/s
    fp32_vector_tflops: float   # peak vector (non-matrix) FP32
    fp32_matrix_tflops: float   # peak matrix-core FP32 (MFMA f32 path)


# gfx950 == MI350X / MI355X. Same ISA and CU design; they differ only in clock.
# [microarch.md]: 256 CU, 160 KB LDS/CU, 8 TB/s HBM3E, FP32 matrix 256 FLOP/clk/CU.
# Vector FP32 is 128 FLOP/clk/CU (64 lanes x 2 for FMA).
_GFX950 = dict(wave_size=64, lds_bytes=160 * 1024, cus=256, hbm_gbps=8000.0)

ARCHS = {
    "gfx950-MI355X": Arch("gfx950", clock_ghz=2.4, fp32_vector_tflops=78.6,
                          fp32_matrix_tflops=157.3, **_GFX950),
    "gfx950-MI350X": Arch("gfx950", clock_ghz=2.2, fp32_vector_tflops=72.1,
                          fp32_matrix_tflops=144.2, **_GFX950),
    # gfx942 (MI300X) fallback so the repo is at least *runnable* off CDNA4.
    "gfx942": Arch("gfx942", wave_size=64, lds_bytes=64 * 1024, cus=304,
                   clock_ghz=2.1, hbm_gbps=5300.0,
                   fp32_vector_tflops=163.4, fp32_matrix_tflops=163.4),
}


@functools.lru_cache(maxsize=1)
def device_name() -> str:
    import torch

    return torch.cuda.get_device_name(0)


@functools.lru_cache(maxsize=1)
def gcn_arch() -> str:
    import torch

    # gcnArchName carries the feature suffix ("gfx950:sramecc+:xnack-"); strip it.
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


@functools.lru_cache(maxsize=1)
def arch() -> Arch:
    """The Arch record for the attached GPU.

    MI350X and MI355X are both gfx950 and are distinguished only by the marketing
    name, so we match on that; an unknown gfx950 part is assumed to be the lower
    clocked MI350X (conservative -- it makes reported efficiency an upper bound).
    """
    g = gcn_arch()
    if g == "gfx950":
        return ARCHS["gfx950-MI355X" if "MI355" in device_name() else "gfx950-MI350X"]
    if g in ARCHS:
        return ARCHS[g]
    raise RuntimeError(
        f"No hardware record for {g!r} ({device_name()}). Add one to flyopt/env.py "
        f"-- do not benchmark against a guessed peak."
    )


def wave_size() -> int:
    return arch().wave_size


def flydsl_available() -> bool:
    """True when the FlyDSL wheel imports *and* a GPU is attached.

    Every kernel module guards its FlyDSL imports on this so the package stays
    importable (for docs/tests collection) on a machine with no ROCm.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import flydsl  # noqa: F401

        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def flydsl_version() -> str:
    import flydsl

    return getattr(flydsl, "__version__", "unknown")


def describe() -> str:
    a = arch()
    return (
        f"{device_name()} ({a.name}, wave{a.wave_size}) | "
        f"{a.cus} CU @ {a.clock_ghz} GHz | LDS {a.lds_bytes // 1024} KB/CU | "
        f"HBM {a.hbm_gbps / 1000:.1f} TB/s | FlyDSL {flydsl_version()}"
    )
