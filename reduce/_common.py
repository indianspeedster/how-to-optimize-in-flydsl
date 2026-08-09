# SPDX-License-Identifier: Apache-2.0
"""Constants and LDS storage shared by the three reduce families.

Each family lives in its own file -- ``tree.py`` (rungs 0-2), ``halved.py``
(3-5), ``multi_add.py`` (6-9) -- mirroring the way the CUDA original gives each
step its own source file. What they share lives here.
"""

# No `from __future__ import annotations` -- @fx.struct resolves its field
# annotations eagerly and PEP 563 stringification breaks the LDS layout.

from common.dsl import fx

THREADS = 256          # 4 wavefronts on CDNA -- the original's THREAD_PER_BLOCK
MULTI_ADD_BLOCKS = 1024  # the original's fixed grid for reduce6 / reduce7
# A 256-CU part wants far more blocks in flight than the V100 the original was
# tuned on: 1024 blocks is 4 per CU, i.e. 4 wavefront-quads, which cannot cover
# HBM latency. 32 blocks per CU is the CDNA4-sized grid the last rung uses.
CDNA_BLOCKS = 32 * 256

# Elements one block consumes, per rung. This is the *only* thing that makes the
# rungs produce different-shaped outputs, and it is the honest port: the CUDA
# files change their grid the same way as the optimization progresses.
_ELEMS_PER_BLOCK = {
    "v0_baseline": THREADS,
    "v1_no_divergence": THREADS,
    "v2_no_bank_conflict": THREADS,
    "v3_add_during_load": THREADS * 2,
    "v4_unroll_last_wave": THREADS * 2,
    "v5_full_unroll": THREADS * 2,
}
# Rungs whose grid is fixed and whose chunk therefore scales with N.
_GRID_BLOCKS = {
    "v6_multi_add": MULTI_ADD_BLOCKS,
    "v7_shuffle": MULTI_ADD_BLOCKS,
    "v8_shuffle_vec4": MULTI_ADD_BLOCKS,
    "v9_vec4_wide_grid": CDNA_BLOCKS,
}


def elems_per_block(variant: str, N: int) -> int:
    if variant in _ELEMS_PER_BLOCK:
        return _ELEMS_PER_BLOCK[variant]
    return N // _GRID_BLOCKS[variant]


def shared_storage(slots: int):
    """LDS storage struct for ``slots`` f32, 16-byte aligned."""

    @fx.struct
    class SharedStorage:
        s: fx.Array[fx.Float32, slots, 16]

    return SharedStorage
