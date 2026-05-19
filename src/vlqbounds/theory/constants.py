"""
vlq_bounds.foundation.constants
════════════════════════════════
Standard Model physical constants, derived coupling prefactors, and the
canonical Pandas DataFrame schema used to record analysis results.

All values follow PDG 2022 conventions unless noted otherwise.
The module is intentionally free of imports from the rest of the package
so it can be safely imported anywhere in the dependency graph.
"""

from __future__ import annotations

import math
from typing import Final, List

import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
#  SM masses  (GeV)
# ══════════════════════════════════════════════════════════════════════════════

#: Top-quark pole mass (PDG 2022)
Mt: Final[float] = 172.52

#: W-boson mass (PDG 2022)
MW: Final[float] = 80.43

#: Z-boson mass (PDG 2022)
MZ: Final[float] = 91.1876

#: Bottom-quark MS-bar mass at m_b (PDG 2022)
Mb: Final[float] = 4.18

#: Higgs-boson mass (PDG 2022)
Mh: Final[float] = 125.09

# ══════════════════════════════════════════════════════════════════════════════
#  EW couplings and mixing
# ══════════════════════════════════════════════════════════════════════════════

#: SU(2) gauge coupling  g = G_F × √2 × 2 M_W²  ≈ 0.653
G: Final[float] = 0.653

#: Mathematical constant π
PI: Final[float] = math.pi

#: Cosine of the weak-mixing angle  cos θ_W = M_W / M_Z
C_W: Final[float] = MW / MZ

#: Higgs vacuum expectation value  v = 2 M_W / g  ≈ 246.22 GeV
v: Final[float] = 2.0 * MW / G

# ══════════════════════════════════════════════════════════════════════════════
#  Decay-width prefactors
# ══════════════════════════════════════════════════════════════════════════════
#
#  These prefactors appear in the standard VLQ partial-width formulæ.
#  They are stored here to avoid recomputing them on every call.
#
#  Convention:
#    Γ(Q → W q′) uses Cst1
#    Γ(Q → Z q)  uses Cst2
#    Γ(Q → H q)  uses Cst3

#: Prefactor for Q → W + q′  :  g² / (64π)
Cst1: Final[float] = G**2 / (64.0 * PI)

#: Prefactor for Q → Z + q   :  g² / (128π cos²θ_W)
Cst2: Final[float] = G**2 / (128.0 * PI * C_W**2)

#: Prefactor for Q → H + q   :  g² / (128π)
Cst3: Final[float] = G**2 / (128.0 * PI)

# ══════════════════════════════════════════════════════════════════════════════
#  Numerical tolerances
# ══════════════════════════════════════════════════════════════════════════════

#: Absolute tolerance used when comparing a coupling value to a fixed target
#: (e.g. deciding whether Γ/M "equals" 0.01).
Threshold: Final[float] = 0.01

#: Minimum physically sensible Γ/M ratio (avoids log-interpolation singularity)
MIN_WIDTH_RATIO: Final[float] = 1.0e-6

#: Maximum physically sensible Γ/M ratio
MAX_WIDTH_RATIO: Final[float] = 1.0

# ══════════════════════════════════════════════════════════════════════════════
#  DataFrame schema
# ══════════════════════════════════════════════════════════════════════════════

#: Columns describing the input parameter point
INPUT_COLUMNS: Final[List[str]] = ["mass", "mixing", "coupling", "model", "which_doublet"]

#: Columns for the three cross-section values (theory, observed limit, expected limit)
CROSS_SECTION_COLUMNS: Final[List[str]] = ["predicted_xs", "observed_xs", "expected_xs"]

#: Columns for dimensionless ratios
RATIO_COLUMNS: Final[List[str]] = ["width_ratio", "obs_ratio", "exp_ratio"]

#: Columns for the exclusion verdict
RESULT_COLUMNS: Final[List[str]] = ["result", "channel", "process"]

#: Columns describing the experimental dataset
EXPERIMENT_COLUMNS: Final[List[str]] = ["experiment", "luminosity", "energy", "label"]

#: Complete ordered column list for the results DataFrame
ALL_COLUMNS: Final[List[str]] = (
    INPUT_COLUMNS
    + CROSS_SECTION_COLUMNS
    + RATIO_COLUMNS
    + RESULT_COLUMNS
    + EXPERIMENT_COLUMNS
)

#: Canonical empty DataFrame with the correct schema and dtype=object for all columns.
#: Clone this with ``df.copy()`` rather than modifying the module-level instance.
df: pd.DataFrame = pd.DataFrame(
    {col: pd.Series(dtype="object") for col in ALL_COLUMNS}
)


# ══════════════════════════════════════════════════════════════════════════════
#  Derived / convenience constants
# ══════════════════════════════════════════════════════════════════════════════

#: Branching-ratio sum for the SM-like scenario (all three channels):
#: BR(W) : BR(Z) : BR(H)  →  2 : 1 : 1  (VLQ mass ≫ thresholds)
SM_BR_W: Final[float] = 0.50
SM_BR_Z: Final[float] = 0.25
SM_BR_H: Final[float] = 0.25


def summary() -> str:
    """Return a human-readable summary of all constants defined in this module."""
    lines = [
        "VLQBounds physical constants",
        "─" * 44,
        f"  Mt  = {Mt:.4f} GeV    (top-quark pole mass)",
        f"  MW  = {MW:.4f} GeV    (W-boson mass)",
        f"  MZ  = {MZ:.6f} GeV  (Z-boson mass)",
        f"  Mb  = {Mb:.4f} GeV    (b-quark MS mass)",
        f"  Mh  = {Mh:.4f} GeV   (Higgs mass)",
        f"  G   = {G:.4f}        (SU(2) coupling)",
        f"  v   = {v:.4f} GeV   (Higgs VEV)",
        f"  C_W = {C_W:.6f}     (cos θ_W)",
        f"  Cst1= {Cst1:.6e}   (W prefactor)",
        f"  Cst2= {Cst2:.6e}   (Z prefactor)",
        f"  Cst3= {Cst3:.6e}   (H prefactor)",
    ]
    return "\n".join(lines)
