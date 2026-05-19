"""
exotic_partner_models.py
════════════════════════════════════════════════════════════════════════════
Vector-like exotic-charge quark partner model classes.

Classes
───────
  XPartnerDoublet  — SU(2) doublet Q = +5/3 exotic partner  (X, T) family
  YPartnerDoublet  — SU(2) doublet Q = −4/3 exotic partner  (B, Y) family
  YPartnerTriplet  — SU(2) triplet Q = −4/3 exotic partner  (T, B, Y) family

All three species decay predominantly via the charged W boson, and their
partial widths share the same kinematic structure as the top/bottom partners.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .model_base import VectorLikeModelBase, c


# ══════════════════════════════════════════════════════════════════════════════
#  XPartnerDoublet  (Q = +5/3, (X, T) representation)
# ══════════════════════════════════════════════════════════════════════════════

class XPartnerDoublet(VectorLikeModelBase):
    """
    SU(2) doublet vector-like quark with electric charge Q = +5/3.

    The only open decay channel at leading order is  X → W t.
    Single-production proceeds via  pp → Xtq  with right-handed mixing.

    Parameter input
    ────────────────
      mX  : VLQ mass in GeV  (required)
      k_X : Universal coupling κ
      w_m : Γ/M ratio
      s_r : sin θ_R  (right-handed mixing)
    """

    species_code  = "X"
    scenario_name = "Doublet"

    def __init__(self) -> None:
        super().__init__(family="XT")

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_mX(self, mx: float) -> None:
        """Set the X mass in GeV."""
        self._set_mass(mx)

    def set_sin_r(self, s_r: float) -> None:
        """Set the right-handed mixing sine sin θ_R."""
        self._set_primary_mixing(s_r)

    def set_coupling_strength(self, k: float) -> None:
        """Set the universal coupling strength κ."""
        self._set_kappa(k)

    def set_width_mass_ratio(self, wr: float) -> None:
        """Set the Γ/M ratio."""
        self._set_width_ratio(wr)

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_mX(self) -> float:
        """Return the X mass in GeV."""
        return self._get_mass()

    def get_sin_right(self) -> float:
        """
        Return sin θ_R.

        Derivation priority:
        1. Directly set value.
        2. κ (returned as-is for single-channel models).
        3. Inferred from Γ/M.
        """
        if self._safe_primary_mixing() is not None:
            return abs(self._safe_primary_mixing())
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        return self._infer_kappa_from_width_ratio(model="Doublet")

    def get_coupling_strength(self) -> float:
        """
        Return κ.

        Derivation priority: Γ/M inversion → direct κ → sin θ_R.
        """
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Doublet")
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        return self._safe_primary_mixing()

    def get_width_mass_ratio(self) -> float:
        """Return Γ/M, computed from Γ(X → Wt) when not set directly."""
        if self._safe_width_ratio() is not None:
            return self._safe_width_ratio()
        return self.x_decay_to_wt() / self.get_mX()

    # ── Cross sections ────────────────────────────────────────────────────────

    def get_xs_pp_Xtq_twtq(self, i: int) -> Optional[float]:
        """σ(pp → Xtq → Wttq) for channel 14 (CMS 1809.08597 γ/M=1 scenario)."""
        if i == 14:
            return self._interpolate_curve_file(
                "1809.08597_Fig8_upper_right_pp_X_Wt_gamma_M1.json",
                species="X",
            )
        return None

    def get_xs_pp_QQ(self) -> float:
        """Return σ(pp → XX̄) at NLO+NLL in pb."""
        return self._get_pair_production_cross_section()

    # ── Partial width ─────────────────────────────────────────────────────────

    def x_decay_to_wt(self) -> float:
        """Γ(X → Wt) in GeV.  The only open decay channel for this species."""
        return self._width(
            c.MW, c.Mt, c.Cst1,
            coupling=[self.get_sin_right(), 0.0],
            model="Doublet",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  YPartnerDoublet  (Q = −4/3, (B, Y) doublet)
# ══════════════════════════════════════════════════════════════════════════════

class YPartnerDoublet(VectorLikeModelBase):
    """
    SU(2) doublet vector-like quark with electric charge Q = −4/3.

    The dominant decay channel is  Y → W b  (charged-current decay).
    Single-production proceeds via  pp → Ybq  with right-handed mixing.

    Parameter input
    ────────────────
      mY  : VLQ mass in GeV  (required)
      k_Y : Universal coupling κ
      w_m : Γ/M ratio
      s_r : sin θ_R  (right-handed mixing)
    """

    species_code  = "Y"
    scenario_name = "Doublet"

    def __init__(self) -> None:
        super().__init__(family="BY")

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_mY(self, my: float) -> None:
        """Set the Y mass in GeV."""
        self._set_mass(my)

    def set_sin_r(self, s_r: float) -> None:
        """Set the right-handed mixing sine sin θ_R."""
        self._set_primary_mixing(s_r)

    def set_coupling_strength(self, k: float) -> None:
        self._set_kappa(k)

    def set_width_mass_ratio(self, wr: float) -> None:
        self._set_width_ratio(wr)

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_mY(self) -> float:
        return self._get_mass()

    def get_sin_right(self) -> float:
        """Return sin θ_R, derived from κ or Γ/M when not set directly."""
        if self._safe_primary_mixing() is not None:
            return abs(self._safe_primary_mixing())
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        return self._infer_kappa_from_width_ratio(model="Doublet")

    def get_coupling_strength(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Doublet")
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        return self._safe_primary_mixing()

    def get_width_mass_ratio(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._safe_width_ratio()
        return self.y_decay_to_wb() / self.get_mY()

    # ── Cross sections ────────────────────────────────────────────────────────

    def get_xs_pp_Ybq_Wbbq(self, i: int) -> Optional[float]:
        """σ(pp → Ybq → Wbbq) for various doublet-Y channels."""
        if i in (6, 7):
            return self._interpolate_scan_grid(
                grid_key="2409.20273",
                model_label="doublet",
                species="Y",
                coupling_strength=self.get_coupling_strength(),
            )
        if i == 0:
            return self._interpolate_curve_file(
                "1701.08328_CMS_Fig5_pp_Ybq_bWbq_c05_theo.json", species="Y"
            )
        if i == 4:
            return self._interpolate_curve_file(
                "1602.05606_ATLAS_Fig6_pp_Ybj_Wb_doublet_theo.json", species="Y"
            )
        return None

    def get_xs_pp_QQ(self) -> float:
        """Return σ(pp → YȲ) at NLO+NLL in pb."""
        return self._get_pair_production_cross_section()

    # ── Partial width ─────────────────────────────────────────────────────────

    def y_decay_to_wb(self) -> float:
        """Γ(Y → Wb) in GeV."""
        return self._width(
            c.MW, c.Mb, c.Cst1,
            coupling=[self.get_sin_right(), 0.0],
            model="Doublet",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  YPartnerTriplet  (Q = −4/3, (T, B, Y) triplet)
# ══════════════════════════════════════════════════════════════════════════════

class YPartnerTriplet(VectorLikeModelBase):
    """
    SU(2) triplet vector-like quark with electric charge Q = −4/3.

    Belongs to the (T, B, Y) triplet representation.
    The dominant decay channel is  Y → W b.
    Single-production proceeds via  pp → Ybq  with left-handed down mixing.

    Parameter input
    ────────────────
      mY    : VLQ mass in GeV  (required)
      k_Y   : Universal coupling κ
      w_m   : Γ/M ratio
      s_d_l : sin θ_dL  (left-handed down-type mixing angle)
    """

    species_code  = "Y"
    scenario_name = "Triplet"

    def __init__(self) -> None:
        super().__init__(family="TBY")

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_mY(self, my: float) -> None:
        """Set the Y mass in GeV."""
        self._set_mass(my)

    def set_sin_down_left(self, s_d_l: float) -> None:
        """Set the left-handed down-type mixing sine sin θ_dL."""
        self._set_primary_mixing(s_d_l)

    def set_coupling_strength(self, k: float) -> None:
        self._set_kappa(k)

    def set_width_mass_ratio(self, wr: float) -> None:
        self._set_width_ratio(wr)

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_mY(self) -> float:
        return self._get_mass()

    def get_sin_down_left(self) -> float:
        """Return sin θ_dL, derived from κ or Γ/M when not set directly."""
        if self._safe_primary_mixing() is not None:
            return abs(self._safe_primary_mixing())
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        return self._infer_kappa_from_width_ratio(model="Triplet")

    def get_coupling_strength(self) -> float:
        """
        Return κ.

        In the triplet representation  κ = √2 × sin θ_dL.
        """
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Triplet")
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        return math.sqrt(2.0) * self.get_sin_down_left()

    def get_width_mass_ratio(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._safe_width_ratio()
        return self.y_decay_to_wb() / self.get_mY()

    # ── Cross sections ────────────────────────────────────────────────────────

    def get_xs_pp_Ybq_Wbbq(self, i: int) -> Optional[float]:
        """σ(pp → Ybq → Wbbq) for triplet-Y channel 0."""
        if i == 0:
            return self._interpolate_curve_file(
                "1701.08328_CMS_Fig5_pp_Ybq_bWbq_c05_theo.json", species="Y"
            )
        return None

    def get_xs_pp_QQ(self) -> float:
        """Return σ(pp → YȲ) at NLO+NLL in pb."""
        return self._get_pair_production_cross_section()

    # ── Partial width ─────────────────────────────────────────────────────────

    def y_decay_to_wb(self) -> float:
        """Γ(Y → Wb) in GeV  (triplet coupling convention)."""
        return self._width(
            c.MW, c.Mb, c.Cst1,
            coupling=[self.get_sin_down_left(), 0.0],
            model="Triplet",
        )
