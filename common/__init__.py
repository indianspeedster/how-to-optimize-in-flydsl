# SPDX-License-Identifier: Apache-2.0
"""Machinery shared by every kernel folder.

``env``      hardware detection and the peak figures results are measured against
``dsl``      device-side FlyDSL helpers (copy atoms, shuffles, MFMA, fast launch)
``registry`` the Op / Variant model the bench and the tests both drive
``sparse``   reproducible CSR generation for the spmv / spmm folders
"""

from . import dsl, env, registry, sparse  # noqa: F401
