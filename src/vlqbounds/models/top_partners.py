"""
top_partner_models.py
════════════════════════════════════════════════════════════════════════════
Vector-like top-quark partner model classes.

Classes
───────
  TopPartnerSinglet  — SU(2) singlet T partner (mixes with tL)
  TopPartnerDoublet  — SU(2) doublet T partner in the (X,T) or (T,B) representation
  TopPartnerPure     — 100%-branching-ratio ("pure") T partner for limit interpretation
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .model_base import VectorLikeModelBase, c
from vlqbounds.theory.kinematics import evaluate_partial_width


# ══════════════════════════════════════════════════════════════════════════════
#  SU(2) Singlet T
# ══════════════════════════════════════════════════════════════════════════════

class TopPartnerSinglet(VectorLikeModelBase):
    """
    SU(2) singlet top-quark partner.

    Decay channels
    ──────────────
      T → W b   (dominant for large sin θ_L)
      T → Z t   (subleading)
      T → H t   (subleading)

    Branching ratios approach 2:1:1 in the heavy-mass decoupling limit.

    Parameter input (exactly one companion to the mass)
    ────────────────────────────────────────────────────
      mT          : VLQ mass in GeV            (required)
      k_T         : Universal coupling κ
      w_m         : Γ/M ratio
      s_l         : sin θ_L (left-handed mixing angle)
    """

    species_code  = "T"
    scenario_name = "Singlet"

    def __init__(self) -> None:
        super().__init__()

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_mT(self, mT: float) -> None:
        """Set the T mass in GeV."""
        self._set_mass(mT)

    def set_sin_l(self, s_l: float) -> None:
        """Set the left-handed mixing sine sin θ_L."""
        self._set_primary_mixing(s_l)

    def set_coupling_strength(self, k: float) -> None:
        """Set the universal coupling strength κ."""
        self._set_kappa(k)

    def set_width_mass_ratio(self, wr: float) -> None:
        """Set the Γ/M ratio."""
        self._set_width_ratio(wr)

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_mT(self) -> float:
        """Return the T mass in GeV."""
        return self._get_mass()

    def get_sin_left(self) -> float:
        """
        Return sin θ_L.  Derived from κ or Γ/M when not set directly.
        """
        if self._safe_primary_mixing() is not None:
            return self._safe_primary_mixing()
        if self._safe_kappa() is not None:
            return self._safe_kappa() / math.sqrt(2.0)
        return self._infer_kappa_from_width_ratio(model="Singlet") / math.sqrt(2.0)

    def get_coupling_strength(self) -> float:
        """Return κ.  Derived from Γ/M or sin θ_L when not set directly."""
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Singlet")
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        return math.sqrt(2.0) * self.get_sin_left()

    def get_coupling_to_zt_or_ht(self) -> float:
        """
        Return the effective off-diagonal coupling  sin θ_L cos θ_L
        relevant for the Z-t and H-t vertices.
        """
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Singlet") / math.sqrt(2.0)
        if self._safe_kappa() is not None:
            return self._safe_kappa() / math.sqrt(2.0)
        s = self.get_sin_left()
        return s * math.sqrt(max(0.0, 1.0 - s**2))

    def get_width_mass_ratio(self) -> float:
        """
        Return Γ/M.  Computed from the sum of all partial widths when not
        set directly.
        """
        if self._safe_width_ratio() is not None:
            return self._safe_width_ratio()
        return (
            self.T_decay_to_wb()
            + self.T_decay_to_ht()
            + self.T_decay_to_zt()
        ) / self.get_mT()

    # ── Pair production ───────────────────────────────────────────────────────

    def get_xs_pp_QQ(self) -> float:
        """Return σ(pp → TT̄) at NLO+NLL in pb."""
        return self._get_pair_production_cross_section()

    # ── Single-production theory cross sections ───────────────────────────────

    def get_xs_pp_Tbq_Wbbq(self, i: int) -> Optional[float]:
        """σ(pp → Tbq → Wbbq) for channels that proceed via the Wb decay."""
        if i == 30:
            return self._interpolate_curve_file(
                "1602.05606_ATLAS_Fig6_pp_T_Wb_singlet_theo.json", species="T"
            )
        if i == 58:
            return self._interpolate_curve_file(
                "1701.08328_CMS_fig5_pp_T_bW_singlet_C05_theo.json", species="T"
            )
        if i in (59, 60):
            return self._interpolate_scan_grid(
                grid_key="2409.20273",
                model_label="singlet",
                species="T",
                coupling_strength=self.get_coupling_strength(),
            )
        return None

    def get_xs_pp_Tj_Ztj_ts_channels(self, i: int) -> Optional[float]:
        """σ for channels reconstructed via the Zt decay."""
        if i in (32, 33, 34):
            return self._interpolate_scan_grid(
                grid_key="2305.03401",
                model_label="singlet",
                species="T",
                coupling_strength=self.get_coupling_strength(),
            ) / 1000.0
        if i in (43, 44, 45):
            return self._interpolate_scan_grid(
                grid_key="2307.07584",
                model_label="singlet",
                species="T",
                coupling_strength=self.get_coupling_strength(),
            )
        if i == 35:
            return (
                self._interpolate_curve_file(
                    "2402.16561_ATLAS_fig9c_pp_T_Zt_singlet_k05_theo.json", species="T"
                )
                / 1000.0
            )
        return None

    def get_xs_pp_Tbq_tHbq(self, i: int) -> Optional[float]:
        """σ for channels reconstructed via the Ht decay."""
        if i in (37, 38, 39, 40, 41, 42):
            return self._interpolate_scan_grid(
                grid_key="2201.07045",
                model_label="singlet",
                species="T",
                coupling_strength=self.get_coupling_strength(),
            ) / 1000.0
        if i == 17:
            return self._interpolate_curve_file(
                "1612.00999_CMS_Fig10left_pp_T_tH_C05_singlet_theo.json", species="T"
            )
        if i == 46:
            return self._interpolate_curve_file(
                "1612.05336_CMS_fig4upperleft_pp_T_tH_c05_singlet_theo.json", species="T"
            )
        if i == 16:
            return self._interpolate_scan_grid(
                grid_key="2302.12802",
                model_label="singlet",
                species="T",
                coupling_strength=self.get_sin_left(),
            ) / 1000.0
        if i in (18, 19, 24, 25):
            return self._interpolate_scan_grid(
                grid_key="2201.02227",
                model_label="Singlet",
                species="T",
                width_ratio=self.get_width_mass_ratio(),
            )
        return None

    def get_xs_pp_Tbq_tZbq(self, i: int) -> Optional[float]:
        """σ for single-T channels decaying via Zt."""
        if i == 57:
            return self._interpolate_curve_file(
                "1806.10555_ATLAS_fig15_pp_T_Zt_Singlet_k05_theo.json", species="T"
            )
        if i == 31:
            return self._interpolate_curve_file(
                "1701.07409_CMS_Fig4left_pp_T_tZ_singlet_theo.json", species="T"
            )
        if i == 14:
            return self._interpolate_curve_file(
                "1708.01062_CMS_Fig5left_pp_T_tZ_singlet_C05_theo.json", species="T"
            )
        if i == 36:
            return self._interpolate_curve_file(
                "1812.09743_ATLAS_Fig4c_pp_T_tZ_Singlet_theo.json", species="T"
            )
        if i in (10, 11, 12, 13, 20, 21, 26, 27):
            return self._interpolate_scan_grid(
                grid_key="2201.02227",
                model_label="Singlet",
                species="T",
                width_ratio=self.get_width_mass_ratio(),
            )
        if i == 50:
            return self._interpolate_curve_file(
                "2405.05071_CMS_Fig4ur_pp_T_Zt_singlet_theo.json", species="T"
            )
        if i == 51:
            return self._interpolate_curve_file(
                "2405.05071_CMS_Fig4ul_pp_T_tH_singlet_theo.json", species="T"
            )
        return None

    def get_xs_pp_Tbq_tZ_plus_TH(self, i: int) -> Optional[float]:
        """σ for combined (Zt + Ht) decay channels."""
        if i == 52:
            return self._interpolate_curve_file(
                "2405.05071_CMS_Fig4ll_pp_T_ZttH_singlet_theo.json", species="T"
            )
        if i == 53:
            return self._interpolate_curve_file(
                "2405.05071_CMS_Fig4lr_pp_T_ZttH_singlet_theo.json", species="T"
            )
        if i in (22, 23, 28, 29):
            return 2.0 * self._interpolate_scan_grid(
                grid_key="2201.02227",
                model_label="Singlet",
                species="T",
                width_ratio=self.get_width_mass_ratio(),
            )
        return None

    def get_xs_pp_Tbq(self, i: int) -> Optional[float]:
        """σ(pp → Tbq) inclusive single-T production for channel 15."""
        if i == 15:
            xs_tz = self._interpolate_scan_grid(
                grid_key="2201.02227",
                model_label="Singlet",
                species="T",
                width_ratio=self.get_width_mass_ratio(),
            )
            # Divide by BR(T→Zt) ≈ 0.25 to recover inclusive single-T xs
            return xs_tz / 0.25
        return None

    # ── Partial widths ────────────────────────────────────────────────────────

    def T_decay_to_wb(self) -> float:
        """Γ(T → W b) in GeV."""
        return self._width(
            c.MW, c.Mb, c.Cst1,
            coupling=[self.get_sin_left(), 0.0],
            model="Singlet",
        )

    def T_decay_to_zt(self) -> float:
        """Γ(T → Z t) in GeV."""
        s_l = self.get_sin_left()
        return self._width(
            c.MZ, c.Mt, c.Cst2,
            coupling=[s_l * math.sqrt(max(0.0, 1.0 - s_l**2)), 0.0],
            model="Singlet",
        )

    def T_decay_to_ht(self) -> float:
        """Γ(T → H t) in GeV."""
        s_l = self.get_sin_left()
        return self._width(
            c.Mh, c.Mt, c.Cst3,
            coupling=[s_l * math.sqrt(max(0.0, 1.0 - s_l**2)), 0.0],
            model="Singlet",
            higgs=True,
        )

    # ── Branching ratios ──────────────────────────────────────────────────────

    def _all_widths(self) -> List[float]:
        return [self.T_decay_to_wb(), self.T_decay_to_zt(), self.T_decay_to_ht()]

    def get_brTbw(self) -> float:
        """BR(T → Wb)."""
        w = self._all_widths()
        return self._branching_ratio(w[0], w)

    def get_brTzt(self) -> float:
        """BR(T → Zt)."""
        w = self._all_widths()
        return self._branching_ratio(w[1], w)

    def get_brTht(self) -> float:
        """BR(T → Ht)."""
        w = self._all_widths()
        return self._branching_ratio(w[2], w)


# ══════════════════════════════════════════════════════════════════════════════
#  SU(2) Doublet T
# ══════════════════════════════════════════════════════════════════════════════

class TopPartnerDoublet(VectorLikeModelBase):
    """
    SU(2) doublet top-quark partner.

    Supports two doublet family configurations:
      (X, T) — right-handed doublet mixing via sin θ_R
      (T, B) — left-handed doublet mixing via sin θ_uR

    Decay channels depend on the active doublet family.

    Parameter input
    ────────────────
      mT     : VLQ mass in GeV (required)
      k_T    : Universal coupling κ
      w_m    : Γ/M ratio
      s_r    : sin θ_R  (for XT doublet)
      s_u_r  : sin θ_uR (for TB doublet)
    """

    species_code  = "T"
    scenario_name = "Doublet"

    def __init__(self) -> None:
        super().__init__(family="XT")

    # ── Setters ───────────────────────────────────────────────────────────────

    def set_mT(self, mT: float) -> None:
        self._set_mass(mT)

    def set_sin_r(self, s_r: float) -> None:
        """Set sin θ_R for the (X, T) doublet configuration."""
        self._set_primary_mixing(s_r)

    def set_sin_u_r(self, s_u_r: float) -> None:
        """Set sin θ_uR for the (T, B) doublet configuration."""
        self._set_secondary_mixing(s_u_r)

    def set_coupling_strength(self, k: float) -> None:
        self._set_kappa(k)

    def set_width_mass_ratio(self, wr: float) -> None:
        self._set_width_ratio(wr)

    def change_to_TB(self) -> None:
        """Switch the active doublet family from XT to TB."""
        self._state.multiplet_family = "TB"

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_mT(self) -> float:
        return self._get_mass()

    def get_sin_right(self) -> float:
        """Return sin θ_R (XT configuration)."""
        if self._safe_primary_mixing() is not None:
            return abs(self._safe_primary_mixing())
        return self._mixing_from_kappa(self.get_coupling_strength())[0]

    def get_sin_up_right(self) -> float:
        """Return sin θ_uR (TB configuration)."""
        if self._safe_secondary_mixing() is not None:
            return abs(self._safe_secondary_mixing())
        return self._mixing_from_kappa(self.get_coupling_strength())[0]

    def get_coupling_strength(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._infer_kappa_from_width_ratio(model="Doublet")
        if self._safe_kappa() is not None:
            return self._safe_kappa()
        s = (
            self.get_sin_right()
            if self.get_which_doublet() == "XT"
            else self.get_sin_up_right()
        )
        return s * math.sqrt(max(0.0, 1.0 - s**2))

    def get_width_mass_ratio(self) -> float:
        if self._safe_width_ratio() is not None:
            return self._safe_width_ratio()
        if self.get_which_doublet() == "XT":
            return (self.T_decay_to_zt_TX() + self.T_decay_to_ht_TX()) / self.get_mT()
        return (self.T_decay_to_zt_TB() + self.T_decay_to_ht_TB()) / self.get_mT()

    # ── Pair production ───────────────────────────────────────────────────────

    def get_xs_pp_QQ(self) -> float:
        return self._get_pair_production_cross_section()

    # ── Curve/grid helpers ────────────────────────────────────────────────────

    def _interpolate_first_existing_curve(
        self,
        candidates: List[str],
    ) -> float:
        """
        Attempt to load each candidate theory-curve file in order,
        returning the first successful result.

        Raises FileNotFoundError when all candidates fail.
        """
        last_error: Optional[Exception] = None
        for name in candidates:
            try:
                return self._interpolate_curve_file(name, species="T")
            except FileNotFoundError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise FileNotFoundError("No theory-curve candidates were provided.")

    # ── Single-production theory cross sections ───────────────────────────────

    def get_xs_pp_Tj_Ztj_ts_channels(self, i: int) -> Optional[float]:
        """σ for doublet T channels reconstructed via the Zt decay."""
        if i in (23, 24, 25):
            return self._interpolate_scan_grid(
                grid_key="2305.03401",
                model_label="doublet",
                species="T",
                coupling_strength=self.get_coupling_strength(),
            ) / 1000.0
        if i in (26, 27, 28):
            return self._interpolate_scan_grid(
                grid_key="2307.07584",
                model_label="doublet",
                species="T",
                coupling_strength=self.get_coupling_strength(),
            )
        return None

    def get_xs_pp_Ttq_tZtq(self, i: int) -> Optional[float]:
        """σ(pp → Ttq → tZtq) for doublet T."""
        if i == 21:
            return self._interpolate_curve_file(
                "1701.07409_CMS_Fig4right_pp_T_tZ_doublet_C05_theo.json", species="T"
            )
        if i == 7:
            return self._interpolate_curve_file(
                "1708.01062_CMS_Fig5right_pp_T_tZ_doublet_C05_theo.json", species="T"
            )
        if i in (11, 12, 17, 18):
            return self._interpolate_scan_grid(
                grid_key="1909.04721",
                model_label="doublet",
                species="T",
                width_ratio=self.get_width_mass_ratio(),
            ) / 2.0
        return None

    def get_xs_pp_Ttq_tHtq(self, i: int) -> Optional[float]:
        """σ(pp → Ttq → tHtq) for doublet T."""
        if i == 5:
            return self._interpolate_first_existing_curve([
                "1612.00999_CMS_Fig10right_pp_T_tH_C05_doublet_theo.json",
                "1701.07409_CMS_Fig4right_pp_T_tZ_doublet_C05_theo.json",
            ])
        if i == 29:
            return self._interpolate_first_existing_curve([
                "1612.05336_CMS_fig4_lower_right_pp_Tbq_tHbq_RH_coupling_theo.json",
                "1612.00999_CMS_Fig10right_pp_T_tH_C05_doublet_theo.json",
                "1612.05336_CMS_fig4lowerright_pp_T_tH_c05_doublet_theo.json",
                "1612.05336_CMS_fig4lowerright_pp_T_tH_c05_doublet_theo_dat.json",
            ])
        if i in (9, 10, 15, 16):
            return self._interpolate_scan_grid(
                grid_key="1909.04721",
                model_label="doublet",
                species="T",
                width_ratio=self.get_width_mass_ratio(),
            ) / 2.0
        return None

    def get_xs_pp_Ttq_tZ_plus_TH(self, i: int) -> Optional[float]:
        """σ for combined (Zt + Ht) doublet-T decay channels."""
        if i in (13, 14, 19, 20):
            return self._interpolate_scan_grid(
                grid_key="1909.04721",
                model_label="doublet",
                species="T",
                width_ratio=self.get_width_mass_ratio(),
            )
        return None

    # ── Internal width helper ─────────────────────────────────────────────────

    def _evaluate_doublet_width(
        self,
        boson_mass:   float,
        daughter_mass: float,
        prefactor:    float,
        *,
        coupling:     list,
        mixing_sine:  float,
        higgs:        bool = False,
    ) -> float:
        """
        Evaluate one partial width for this doublet T model, passing all
        currently-set coupling inputs into the unified evaluator.
        """
        return evaluate_partial_width(
            [self.get_mT(), boson_mass, daughter_mass],
            mixing_sine=mixing_sine,
            kappa=self._safe_kappa(),
            width_ratio=self._safe_width_ratio(),
            coupling=coupling,
            decay_prefactor=prefactor,
            model="Doublet",
            final_state_is_higgs=higgs,
            species="T",
        )

    # ── Partial widths — (X, T) configuration ────────────────────────────────

    def T_decay_to_wb_TX(self) -> float:
        """Γ(T → Wb) in the (X, T) doublet, using the seesaw sin θ_L."""
        s_r = self.get_sin_right()
        s_l = self.sin_left_calc(s_r)
        return self._evaluate_doublet_width(
            c.MW, c.Mb, c.Cst1,
            coupling=[s_l, 0.0],
            mixing_sine=s_r,
            higgs=False,
        )

    def T_decay_to_zt_TX(self) -> float:
        """Γ(T → Zt) in the (X, T) doublet."""
        s_r = self.get_sin_right()
        return self._evaluate_doublet_width(
            c.MZ, c.Mt, c.Cst2,
            coupling=[s_r * math.sqrt(max(0.0, 1.0 - s_r**2)), 0.0],
            mixing_sine=s_r,
            higgs=False,
        )

    def T_decay_to_ht_TX(self) -> float:
        """Γ(T → Ht) in the (X, T) doublet."""
        s_r = self.get_sin_right()
        return self._evaluate_doublet_width(
            c.Mh, c.Mt, c.Cst3,
            coupling=[s_r * math.sqrt(max(0.0, 1.0 - s_r**2)), 0.0],
            mixing_sine=s_r,
            higgs=True,
        )

    # ── Partial widths — (T, B) configuration ────────────────────────────────

    def T_decay_to_zt_TB(self) -> float:
        """Γ(T → Zt) in the (T, B) doublet."""
        s_u_r = self.get_sin_up_right()
        return self._evaluate_doublet_width(
            c.MZ, c.Mt, c.Cst2,
            coupling=[s_u_r * math.sqrt(max(0.0, 1.0 - s_u_r**2)), 0.0],
            mixing_sine=s_u_r,
            higgs=False,
        )

    def T_decay_to_ht_TB(self) -> float:
        """Γ(T → Ht) in the (T, B) doublet."""
        s_u_r = self.get_sin_up_right()
        return self._evaluate_doublet_width(
            c.Mh, c.Mt, c.Cst3,
            coupling=[s_u_r * math.sqrt(max(0.0, 1.0 - s_u_r**2)), 0.0],
            mixing_sine=s_u_r,
            higgs=True,
        )

    # ── Branching ratios — (X, T) ─────────────────────────────────────────────

    def _widths_XT(self) -> List[float]:
        return [self.T_decay_to_wb_TX(), self.T_decay_to_zt_TX(), self.T_decay_to_ht_TX()]

    def get_brTbw_XT(self) -> float:
        w = self._widths_XT(); return self._branching_ratio(w[0], w)

    def get_brTzt_XT(self) -> float:
        w = self._widths_XT(); return self._branching_ratio(w[1], w)

    def get_brTht_XT(self) -> float:
        w = self._widths_XT(); return self._branching_ratio(w[2], w)

    # ── Branching ratios — (T, B) ─────────────────────────────────────────────

    def _widths_TB(self) -> List[float]:
        return [self.T_decay_to_zt_TB(), self.T_decay_to_ht_TB()]

    def get_brTzt_TB(self) -> float:
        w = self._widths_TB(); return self._branching_ratio(w[0], w)

    def get_brTht_TB(self) -> float:
        w = self._widths_TB(); return self._branching_ratio(w[1], w)

    # ── Mixing helper ─────────────────────────────────────────────────────────

    def sin_left_calc(self, s_r: float) -> float:
        """Compute sin θ_L from sin θ_R using the top-quark seesaw relation."""
        return self._left_mixing_from_right(s_r, c.Mt)


# ══════════════════════════════════════════════════════════════════════════════
#  Pure T (100% branching ratio scenarios)
# ══════════════════════════════════════════════════════════════════════════════

class TopPartnerPure(VectorLikeModelBase):
    """
    Pure top-quark partner with a fixed 100% branching ratio into one channel.

    Used for model-independent reinterpretation of pair-production limits.
    The active decay channel is set externally via the facade:
      ``pure_T_to_Wb``, ``pure_T_to_Zt``, or ``pure_T_to_Ht``.

    Parameters
    ──────────
    mT : VLQ mass in GeV (only settable parameter).
    """

    species_code  = "T"
    scenario_name = "Pure"

    def __init__(self) -> None:
        super().__init__()

    def set_mT(self, mT: float) -> None:
        """Set the T mass in GeV."""
        self._set_mass(mT)

    def get_mT(self) -> float:
        """Return the T mass in GeV."""
        return self._get_mass()

    def get_xs_pp_QQ(self) -> float:
        """Return σ(pp → TT̄) at NLO+NLL in pb."""
        return self._get_pair_production_cross_section()
