"""
bottom_partner_models.py
════════════════════════════════════════════════════════════════════════════
Vector-like bottom-quark partner model classes.

Classes
───────
  BottomPartnerSinglet — SU(2) singlet B partner (mixes with bL)
  BottomPartnerDoublet — SU(2) doublet B partner in the (B,Y) or (T,B) representation
  BottomPartnerPure    — 100%-branching-ratio B partner for limit interpretation
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .model_base import VectorLikeModelBase, c


# ══════════════════════════════════════════════════════════════════════════════
#  SU(2) Singlet B
# ══════════════════════════════════════════════════════════════════════════════

class BottomPartnerSinglet(VectorLikeModelBase):
    """
    SU(2) singlet bottom-quark partner.

    Decay channels
    ──────────────
      B → W t   (dominant for large sin θ_L)
      B → Z b   (subleading)
      B → H b   (subleading)

    Branching ratios approach 2:1:1 in the heavy-mass decoupling limit.

    Parameter input
    ────────────────
      mB  : VLQ mass in GeV  (required)
      k_B : Universal coupling κ
      w_m : Γ/M ratio
      s_l : sin θ_L
    """

    species_code  = "B"
    scenario_name = "Singlet"

    def __init__(self) -> None:
        super().__init__()

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_mB(self, mB: float) -> None:
        self._set_mass(mB)

    def set_sin_l(self, s_l: float) -> None:
        self._set_primary_mixing(s_l)

    def set_coupling_strength(self, k: float) -> None:
        self._set_kappa(k)

    def set_width_mass_ratio(self, wr: float) -> None:
        self._set_width_ratio(wr)

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_mB(self) -> float:
        return self._get_mass()

    def get_sin_left(self) -> float:
        """Return sin θ_L, derived from κ or Γ/M when not set directly."""
        if self._safe_primary_mixing() is not None:
            return self._safe_primary_mixing()
        if self._safe_kappa() is not None:
            return self._safe_kappa() / math.sqrt(2.0)
        return self._infer_kappa_from_width_ratio(model="Singlet") / math.sqrt(2.0)

    def get_coupling_strength(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Singlet")
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        return math.sqrt(2.0) * self.get_sin_left()

    def get_coupling_to_zt_or_ht(self) -> float:
        """Return sin θ_L cos θ_L for the Z-b and H-b vertex."""
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Singlet") / math.sqrt(2.0)
        if self._safe_kappa() is not None:
            return self._safe_kappa() / math.sqrt(2.0)
        s = self.get_sin_left()
        return s * math.sqrt(max(0.0, 1.0 - s**2))

    def get_width_mass_ratio(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._safe_width_ratio()
        return (
            self.B_decay_to_wt()
            + self.B_decay_to_zb()
            + self.B_decay_to_hb()
        ) / self.get_mB()

    # ── Pair production ───────────────────────────────────────────────────────

    def get_xs_pp_QQ(self) -> float:
        return self._get_pair_production_cross_section()

    # ── Single-production theory cross sections ───────────────────────────────

    def get_xs_pp_Bbq_tWbq(self, i: int) -> Optional[float]:
        if i == 20:
            return self._interpolate_curve_file(
                "2111.10216_Fig4_left_pp_B_Wt_singlet_theo.json", species="B"
            )
        return None

    def get_xs_pp_Btq_tWtq(self, i: int) -> Optional[float]:
        if i == 21:
            return self._interpolate_curve_file(
                "2111.10216_Fig4_right_pp_Bt_Wt_singlet_theo.json", species="B"
            )
        return None

    def get_xs_pp_Bj_bHj_ts_channels(self, i: int) -> Optional[float]:
        if i in (15, 16, 17):
            return self._interpolate_scan_grid(
                grid_key="2308.02595",
                model_label="singlet",
                species="B",
                coupling_strength=self.get_coupling_strength(),
            )
        return None

    def get_xs_pp_Bbq_bHbq(self, i: int) -> Optional[float]:
        if i in (4, 5, 6, 7):
            return self._interpolate_scan_grid(
                grid_key="1802.01486",
                model_label="singlet",
                species="B",
                width_ratio=self.get_width_mass_ratio(),
            )
        return None

    def get_xs_pp_Bbq_bZbq(self, i: int) -> Optional[float]:
        if i == 19:
            return self._interpolate_curve_file(
                "1701.07409_Fig5_right_pp_Bt_bZ_cbZ05_singlet_theo.json", species="B"
            )
        return None

    def get_xs_pp_Btq_bZtq(self, i: int) -> Optional[float]:
        if i == 18:
            return self._interpolate_curve_file(
                "1701.07409_Fig5_left_pp_Bt_bZ_cWt05_singlet_theo.json", species="B"
            )
        return None

    # ── Partial widths ────────────────────────────────────────────────────────

    def B_decay_to_wt(self) -> float:
        """Γ(B → Wt) in GeV."""
        return self._width(
            c.MW, c.Mt, c.Cst1,
            coupling=[self.get_sin_left(), 0.0],
            model="Singlet",
        )

    def B_decay_to_zb(self) -> float:
        """Γ(B → Zb) in GeV."""
        s_l = self.get_sin_left()
        return self._width(
            c.MZ, c.Mb, c.Cst2,
            coupling=[s_l * math.sqrt(max(0.0, 1.0 - s_l**2)), 0.0],
            model="Singlet",
        )

    def B_decay_to_hb(self) -> float:
        """Γ(B → Hb) in GeV."""
        s_l = self.get_sin_left()
        coupling = s_l * math.sqrt(max(0.0, 1.0 - s_l**2))
        return self._width(
            c.Mh, c.Mb, c.Cst3,
            coupling=coupling,
            model="Singlet",
            higgs=True,
        )

    # ── Branching ratios ──────────────────────────────────────────────────────

    def _all_widths(self) -> List[float]:
        return [self.B_decay_to_wt(), self.B_decay_to_zb(), self.B_decay_to_hb()]

    def get_brBwt(self) -> float:
        w = self._all_widths(); return self._branching_ratio(w[0], w)

    def get_brBzb(self) -> float:
        w = self._all_widths(); return self._branching_ratio(w[1], w)

    def get_brBhb(self) -> float:
        w = self._all_widths(); return self._branching_ratio(w[2], w)


# ══════════════════════════════════════════════════════════════════════════════
#  SU(2) Doublet B
# ══════════════════════════════════════════════════════════════════════════════

class BottomPartnerDoublet(VectorLikeModelBase):
    """
    SU(2) doublet bottom-quark partner.

    Supports two doublet family configurations:
      (B, Y) — right-handed doublet, mixing via sin θ_R;  no W channel at tree level
      (T, B) — left-handed doublet, mixing via sin θ_dR;  no W channel at tree level

    Parameter input
    ────────────────
      mB    : VLQ mass in GeV  (required)
      k_B   : Universal coupling κ
      w_m   : Γ/M ratio
      s_r   : sin θ_R  (for BY doublet)
      s_d_r : sin θ_dR (for TB doublet)
    """

    species_code  = "B"
    scenario_name = "Doublet"

    def __init__(self) -> None:
        super().__init__(family="BY")

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_mB(self, mB: float) -> None:
        self._set_mass(mB)

    def set_sin_r(self, s_r: float) -> None:
        """Set sin θ_R for the (B, Y) doublet configuration."""
        self._set_primary_mixing(s_r)

    def set_sin_d_r(self, s_d_r: float) -> None:
        """Set sin θ_dR for the (T, B) doublet configuration."""
        self._set_secondary_mixing(s_d_r)

    def set_coupling_strength(self, k: float) -> None:
        self._set_kappa(k)

    def set_width_mass_ratio(self, wr: float) -> None:
        self._set_width_ratio(wr)

    def change_to_TB(self) -> None:
        """Switch the active doublet family from BY to TB."""
        self._state.multiplet_family = "TB"

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_mB(self) -> float:
        return self._get_mass()

    def get_sin_right(self) -> float:
        """Return sin θ_R (BY configuration)."""
        if self._safe_primary_mixing() is not None:
            return abs(self._safe_primary_mixing())
        return self._mixing_from_kappa(self.get_coupling_strength())[0]

    def get_sin_down_right(self) -> float:
        """Return sin θ_dR (TB configuration)."""
        if self._safe_secondary_mixing() is not None:
            return self._safe_secondary_mixing()
        return self._mixing_from_kappa(self.get_coupling_strength())[0]

    def get_coupling_strength(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Doublet")
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        s = (
            self.get_sin_right()
            if self.get_which_doublet() == "BY"
            else self.get_sin_down_right()
        )
        return s * math.sqrt(max(0.0, 1.0 - s**2))

    def get_width_mass_ratio(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._safe_width_ratio()
        if self.get_which_doublet() == "BY":
            return (self.B_decay_to_zb_BY() + self.B_decay_to_hb_BY()) / self.get_mB()
        return (self.B_decay_to_zb_TB() + self.B_decay_to_hb_TB()) / self.get_mB()

    # ── Pair production ───────────────────────────────────────────────────────

    def get_xs_pp_QQ(self) -> float:
        return self._get_pair_production_cross_section()

    # ── Single-production theory cross sections ───────────────────────────────

    def get_xs_pp_Bbq_bHbq(self, i: int) -> Optional[float]:
        if i in (3, 4, 5, 6):
            return self._interpolate_scan_grid(
                grid_key="1802.01486",
                model_label="doublet",
                species="B",
                width_ratio=self.get_width_mass_ratio(),
            )
        return None

    def get_xs_pp_Bj_bHj_ts_channels(self, i: int) -> Optional[float]:
        if i in (13, 14, 15):
            return self._interpolate_scan_grid(
                grid_key="2308.02595",
                model_label="doublet",
                species="B",
                coupling_strength=self.get_coupling_strength(),
            )
        return None

    def get_xs_pp_Bbq_tWbq(self, i: int) -> Optional[float]:
        if i == 18:
            return self._interpolate_curve_file(
                "2111.10216_Fig4_left_pp_B_Wt_doublet_theo.json", species="B"
            )
        return None

    # ── Partial widths — (B, Y) configuration ────────────────────────────────

    def B_decay_to_wt_BY(self) -> float:
        """
        Γ(B → Wt) in the BY doublet.

        Requires sin θ_R to be set (no tree-level W coupling in pure BY scenario,
        but the subleading contribution via sin θ_L is computed here).

        Raises
        ──────
        ValueError — sin θ_R has not been set.
        """
        if self._safe_primary_mixing() is None:
            raise ValueError(
                "sin θ_R (right-handed mixing) must be set for the BY doublet "
                "before computing the Wt partial width."
            )
        s_l = self.sin_left_calc(self.get_sin_right())
        return self._width(
            c.MW, c.MW, c.Cst1,
            coupling=[s_l, 0.0],
            model="Doublet",
        )

    def B_decay_to_zb_BY(self) -> float:
        """Γ(B → Zb) in the BY doublet."""
        s_r = self.get_sin_right()
        return self._width(
            c.MZ, c.Mb, c.Cst2,
            coupling=[s_r * math.sqrt(max(0.0, 1.0 - s_r**2)), 0.0],
            model="Doublet",
        )

    def B_decay_to_hb_BY(self) -> float:
        """Γ(B → Hb) in the BY doublet."""
        s_r = self.get_sin_right()
        return self._width(
            c.Mh, c.Mb, c.Cst3,
            coupling=[s_r * math.sqrt(max(0.0, 1.0 - s_r**2)), 0.0],
            model="Doublet",
            higgs=True,
        )

    # ── Partial widths — (T, B) configuration ────────────────────────────────

    def B_decay_to_zb_TB(self) -> float:
        """Γ(B → Zb) in the TB doublet."""
        s_d_r = self.get_sin_down_right()
        return self._width(
            c.MZ, c.Mb, c.Cst2,
            coupling=[s_d_r * math.sqrt(max(0.0, 1.0 - s_d_r**2)), 0.0],
            model="Doublet",
        )

    def B_decay_to_hb_TB(self) -> float:
        """Γ(B → Hb) in the TB doublet."""
        s_d_r = self.get_sin_down_right()
        return self._width(
            c.Mh, c.Mb, c.Cst3,
            coupling=[s_d_r * math.sqrt(max(0.0, 1.0 - s_d_r**2)), 0.0],
            model="Doublet",
            higgs=True,
        )

    # ── Branching ratios — (B, Y) ─────────────────────────────────────────────

    def get_brBwt_BY(self) -> float:
        w = [self.B_decay_to_wt_BY(), self.B_decay_to_zb_BY(), self.B_decay_to_hb_BY()]
        return self._branching_ratio(w[0], w)

    def get_brBzb_BY(self) -> float:
        w = [self.B_decay_to_zb_BY(), self.B_decay_to_hb_BY()]
        return self._branching_ratio(w[0], w)

    def get_brBhb_BY(self) -> float:
        w = [self.B_decay_to_zb_BY(), self.B_decay_to_hb_BY()]
        return self._branching_ratio(w[1], w)

    # ── Branching ratios — (T, B) ─────────────────────────────────────────────

    def get_brBzb_TB(self) -> float:
        w = [self.B_decay_to_zb_TB(), self.B_decay_to_hb_TB()]
        return self._branching_ratio(w[0], w)

    def get_brBhb_TB(self) -> float:
        w = [self.B_decay_to_zb_TB(), self.B_decay_to_hb_TB()]
        return self._branching_ratio(w[1], w)

    # ── Mixing helper ─────────────────────────────────────────────────────────

    def sin_left_calc(self, s_r: float) -> float:
        """Compute sin θ_L from sin θ_R using the bottom-quark seesaw relation."""
        return self._left_mixing_from_right(s_r, c.Mb)


# ══════════════════════════════════════════════════════════════════════════════
#  Pure B (100% branching ratio scenarios)
# ══════════════════════════════════════════════════════════════════════════════

class BottomPartnerPure(VectorLikeModelBase):
    """
    Pure bottom-quark partner with a fixed 100% branching ratio into one channel.

    Used for model-independent reinterpretation of pair-production limits.
    The active decay channel is set externally via the facade:
      ``pure_B_to_Wt``, ``pure_B_to_Zb``, or ``pure_B_to_Hb``.

    Parameters
    ──────────
    mB : VLQ mass in GeV (only settable parameter).
    """

    species_code  = "B"
    scenario_name = "Pure"

    def __init__(self) -> None:
        super().__init__()

    def set_mB(self, mB: float) -> None:
        """Set the B mass in GeV."""
        self._set_mass(mB)

    def get_mB(self) -> float:
        """Return the B mass in GeV."""
        return self._get_mass()

    def get_xs_pp_QQ(self) -> float:
        """Return σ(pp → BB̄) at NLO+NLL in pb."""
        return self._get_pair_production_cross_section()
