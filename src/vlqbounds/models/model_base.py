"""
base.py
════════════════════════════════════════════════════════════════════════════
Shared model state management and theory cross-section access for all VLQ
model classes.

Architecture
────────────
  TheoryPredictionSupport — class-level injectable data root + JSON loaders
  _State                  — dataclass holding all mutable model parameters
  VectorLikeModelBase     — concrete base wiring state, decay widths, and XS
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Sequence, Tuple, Union

import numpy as np

from vlqbounds.theory import (
    evaluate_partial_width,
    infer_kappa_from_width_ratio,
    interpolate_surface_value,
    load_scattered_theory_grid_by_pattern,
    load_theory_cross_section_from_repository,
    mass_ratio,
    solve_mixing_from_kappa,
)

from vlqbounds.theory import constants as c


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_name(file_name: str) -> str:
    """Ensure *file_name* has a ``.json`` suffix."""
    return str(Path(file_name).with_suffix(".json"))


def _resolve_scan_model_labels(model_label: str) -> Tuple[str, ...]:
    """
    Return a deduplicated tuple of case variants for a model label so the
    grid loader can try ``"singlet"``, ``"Singlet"``, ``"SINGLET"``, etc.
    """
    raw = str(model_label)
    return tuple(dict.fromkeys([raw, raw.lower(), raw.upper(), raw.title()]))


# ══════════════════════════════════════════════════════════════════════════════
#  Theory prediction mixin
# ══════════════════════════════════════════════════════════════════════════════

class TheoryPredictionSupport:
    """
    Shared JSON-based theory cross-section access for VLQ model classes.

    The data root is a **class-level** attribute so all instances (and
    subclasses) automatically share the same configured root.  It can be
    overridden at any time via :meth:`set_theory_data_root`.

    Class attributes
    ─────────────────
    _theory_data_root : Optional[Path]
        The resolved data directory; None triggers auto-detection inside
        the theory loaders.
    """

    _theory_data_root: ClassVar[Optional[Path]] = None

    @classmethod
    def set_theory_data_root(cls, data_root: Optional[Union[str, Path]]) -> None:
        """
        Configure the data root used by all subsequent theory-XS queries.

        Pass None to restore auto-detection behaviour.
        """
        cls._theory_data_root = (
            None
            if data_root is None
            else Path(data_root).expanduser().resolve()
        )

    @classmethod
    def get_theory_data_root(cls) -> Optional[Path]:
        """Return the currently configured theory data root (may be None)."""
        return cls._theory_data_root

    # ── Low-level loaders ────────────────────────────────────────────────────

    def _load_pair_production_cross_section(
        self,
        heavy_mass: float,
        species:    str,
    ) -> float:
        """
        Return σ(pp→QQ̄) at NLO+NLL in pb.

        The NNLO JSON files store values in fb; the factor of 1/1000 converts
        to pb to match the experimental limit tables.
        """
        return load_theory_cross_section_from_repository(
            heavy_mass=heavy_mass,
            file_name="pp_QQ_NNLO.json",
            vlq_species=species,
            data_root=self.get_theory_data_root(),
        ) / 1000.0

    def _load_single_curve_cross_section(
        self,
        heavy_mass: float,
        file_name:  str,
        species:    str,
    ) -> float:
        """Return σ(heavy_mass) from a named 1-D theory-curve JSON file."""
        return load_theory_cross_section_from_repository(
            heavy_mass=heavy_mass,
            file_name=_json_name(file_name),
            vlq_species=species,
            data_root=self.get_theory_data_root(),
        )

    def _load_single_grid_cross_section(
        self,
        *,
        heavy_mass:       float,
        grid_key:         str,
        model_label:      str,
        species:          str,
        width_ratio:      Optional[float] = None,
        coupling_strength: Optional[float] = None,
    ) -> float:
        """
        Return σ(heavy_mass, scan_value) from a scattered 2-D theory grid,
        trying multiple case variants of *model_label* in order.

        The scan axis is *width_ratio* when supplied, otherwise
        *coupling_strength*.

        Raises
        ──────
        FileNotFoundError — no grid files match the (key, model) pattern.
        RuntimeError      — none of the candidate labels found a grid.
        """
        last_error: Optional[Exception] = None

        for candidate in _resolve_scan_model_labels(model_label):
            try:
                grid = load_scattered_theory_grid_by_pattern(
                    file_key=grid_key,
                    model_label=candidate,
                    vlq_species=species,
                    data_root=self.get_theory_data_root(),
                )
                # Assemble the theory-input tuple for repository interpolation
                theory_input = (
                    grid.masses,
                    grid.scan_values,
                    None if width_ratio is not None else grid.scan_values,
                    grid.cross_sections,
                    heavy_mass,
                    0.0 if width_ratio is None else width_ratio,
                    coupling_strength,
                )
                return interpolate_surface_value(
                    theory_input=theory_input,
                    mode="theory",
                )
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"Unable to resolve theory grid for "
            f"key={grid_key!r}, model={model_label!r}, species={species!r}."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Model state dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _State:
    """
    Mutable container for all settable model parameters.

    Exactly one of ``mixing_primary``, ``kappa``, or ``width_ratio`` is
    expected to be non-None at any time, though the base class does not
    enforce this invariant (subclasses decide which combinations are legal).
    """

    mass:             Optional[float] = None
    mixing_primary:   Optional[float] = None   # sin θ_L  or  sin θ_R  (primary)
    mixing_secondary: Optional[float] = None   # sin θ_dR etc. (secondary)
    kappa:            Optional[float] = None   # universal coupling κ
    width_ratio:      Optional[float] = None   # Γ/M
    multiplet_family: Optional[str]   = None   # "TB", "XT", "BY", "TBY", …

    def reset_coupling(self) -> None:
        """Clear all coupling parameters (preserve mass and family)."""
        self.mixing_primary   = None
        self.mixing_secondary = None
        self.kappa            = None
        self.width_ratio      = None


# ══════════════════════════════════════════════════════════════════════════════
#  VectorLikeModelBase
# ══════════════════════════════════════════════════════════════════════════════

class VectorLikeModelBase(TheoryPredictionSupport):
    """
    Shared state management, decay-width arithmetic, and theory-XS queries
    for all VLQ model classes.

    Subclasses must define two class attributes
    ───────────────────────────────────────────
      species_code  : str — ``"T"``, ``"B"``, ``"X"``, or ``"Y"``
      scenario_name : str — ``"Singlet"``, ``"Doublet"``, ``"Triplet"``, or ``"Pure"``

    Subclasses then call the protected ``_set_*`` / ``_safe_*`` / ``_get_*``
    helpers and the public ``_width()`` dispatcher rather than accessing
    ``_state`` directly.
    """

    species_code:  str = ""
    scenario_name: str = ""

    def __init__(self, *, family: Optional[str] = None) -> None:
        self._state = _State(multiplet_family=family)

    # ── Identity ──────────────────────────────────────────────────────────────

    def model(self) -> str:
        """Return the scenario name (``"Singlet"``, ``"Doublet"``, etc.)."""
        return self.scenario_name

    def get_which_doublet(self) -> Optional[str]:
        """Return the doublet family tag (e.g. ``"TB"``, ``"XT"``, ``"BY"``) or None."""
        return self._state.multiplet_family

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"species={self.species_code!r}, "
            f"scenario={self.scenario_name!r}, "
            f"family={self._state.multiplet_family!r}, "
            f"mass={self._state.mass!r})"
        )

    # ── Mass ──────────────────────────────────────────────────────────────────

    def _set_mass(self, value: float) -> None:
        """Set the VLQ mass in GeV."""
        self._state.mass = float(value)

    def _get_mass(self) -> float:
        """
        Return the VLQ mass in GeV.

        Raises
        ──────
        ValueError — mass has not been set.
        """
        if self._state.mass is None:
            raise ValueError(
                f"The VLQ mass ({self.species_code}) has not been set.  "
                f"Call the appropriate set_m{self.species_code}() method first."
            )
        return float(self._state.mass)

    # ── Coupling parameters ───────────────────────────────────────────────────

    def _set_primary_mixing(self, value: float) -> None:
        """Set the primary mixing sine (sin θ_L or sin θ_R depending on model)."""
        self._state.mixing_primary = float(value)

    def _set_secondary_mixing(self, value: float) -> None:
        """Set the secondary mixing sine (e.g. sin θ_dR for the TB doublet)."""
        self._state.mixing_secondary = float(value)

    def _set_kappa(self, value: float) -> None:
        """Set the universal coupling strength κ (stored as |value|)."""
        self._state.kappa = float(abs(value))

    def _set_width_ratio(self, value: float) -> None:
        """
        Set the Γ/M ratio.

        Raises
        ──────
        ValueError — value is not positive.
        """
        if value <= 0.0:
            raise ValueError(
                f"Width-to-mass ratio must be positive; got {value}."
            )
        self._state.width_ratio = float(value)

    # ── Safe accessors (return None when unset) ───────────────────────────────

    def _safe_primary_mixing(self) -> Optional[float]:
        return None if self._state.mixing_primary is None else float(self._state.mixing_primary)

    def _safe_secondary_mixing(self) -> Optional[float]:
        return None if self._state.mixing_secondary is None else float(self._state.mixing_secondary)

    def _safe_kappa(self) -> Optional[float]:
        return None if self._state.kappa is None else float(self._state.kappa)

    def _safe_width_ratio(self) -> Optional[float]:
        return None if self._state.width_ratio is None else float(self._state.width_ratio)

    # ── Derived quantities ────────────────────────────────────────────────────

    def _infer_kappa_from_width_ratio(self, *, model: str) -> float:
        """
        Invert Γ/M = κ² × D(M)/M to obtain κ for the given *model*.

        Raises
        ──────
        ValueError — width_ratio has not been set.
        """
        if self._state.width_ratio is None:
            raise ValueError(
                "Cannot infer κ: width_ratio has not been set for this model point."
            )
        return infer_kappa_from_width_ratio(
            heavy_mass=self._get_mass(),
            width_ratio=float(self._state.width_ratio),
            species=self.species_code,
            model=model,
        )

    @staticmethod
    def _mixing_from_kappa(kappa: float) -> Tuple[float, float]:
        """
        Return both solutions (sin_minus, sin_plus) for  κ = sin θ cos θ.

        Delegates to :func:`vlqbounds.theory.solve_mixing_from_kappa`.
        """
        return solve_mixing_from_kappa(float(kappa))

    def _left_mixing_from_right(
        self,
        right_sine:        float,
        light_quark_mass:  float,
    ) -> float:
        """
        Compute sin θ_L from sin θ_R via the seesaw-like relation::

            tan θ_L = (m_q / M) × tan θ_R

        Returns sin θ_L ∈ [0, 1).
        """
        cos_r   = np.sqrt(max(0.0, 1.0 - right_sine**2))
        if cos_r < 1.0e-12:
            return 1.0
        tangent = right_sine / cos_r
        ratio   = mass_ratio(light_quark_mass, self._get_mass()) * tangent
        return float(np.sqrt(ratio**2 / (1.0 + ratio**2)))

    # ── Decay-width dispatcher ────────────────────────────────────────────────

    def _width(
        self,
        boson_mass:  float,
        quark_mass:  float,
        prefactor:   float,
        *,
        coupling:    Optional[Union[float, Sequence[float], np.ndarray]],
        model:       str,
        higgs:       bool = False,
    ) -> float:
        """
        Compute one partial decay width by delegating to the unified
        ``evaluate_partial_width`` function in ``vlqbounds.theory``.

        Parameters
        ──────────
        boson_mass  : Mass of the decay-product boson (W, Z, or H) in GeV.
        quark_mass  : Mass of the decay-product quark in GeV.
        prefactor   : Cst1 (W), Cst2 (Z), or Cst3 (H) from the constants module.
        coupling    : Explicit (c1, c2) decay-coupling pair.  When None, the
                      width is derived from the model's κ or Γ/M setting.
        model       : ``"Singlet"``, ``"Doublet"``, or ``"Triplet"``.
        higgs       : True when the final-state boson is the Higgs.

        Returns
        ───────
        Partial width in GeV.
        """
        return evaluate_partial_width(
            involved_masses=[self._get_mass(), boson_mass, quark_mass],
            mixing_sine=self._safe_primary_mixing(),
            kappa=self._safe_kappa(),
            width_ratio=self._safe_width_ratio(),
            coupling=coupling,
            decay_prefactor=prefactor,
            model=model,
            final_state_is_higgs=higgs,
            species=self.species_code,
        )

    # ── Branching-ratio helper ────────────────────────────────────────────────

    def _branching_ratio(
        self,
        partial_width: float,
        widths:        Sequence[float],
    ) -> float:
        """
        Return  BR = Γ_partial / Σ Γ_i.

        Raises
        ──────
        ZeroDivisionError — total decay width is not positive.
        """
        total = float(sum(widths))
        if total <= 0.0:
            raise ZeroDivisionError(
                f"Total decay width Σ Γ = {total:.6e} GeV is not positive."
            )
        return float(partial_width / total)

    # ── Theory cross-section queries ──────────────────────────────────────────

    def _get_pair_production_cross_section(self) -> float:
        """Return σ(pp→QQ̄) at the model's current mass, converted to pb."""
        return self._load_pair_production_cross_section(self._get_mass(), self.species_code)

    def _interpolate_curve_file(
        self,
        file_name: str,
        *,
        species: str,
    ) -> float:
        """Return σ(m) from a named 1-D theory-curve JSON file."""
        return self._load_single_curve_cross_section(self._get_mass(), file_name, species)

    def _interpolate_scan_grid(
        self,
        *,
        grid_key:          str,
        model_label:       str,
        species:           str,
        coupling_strength: Optional[float] = None,
        width_ratio:       Optional[float] = None,
    ) -> float:
        """
        Return σ(m, s) from a scattered 2-D theory grid.

        Exactly one of *coupling_strength* or *width_ratio* should be supplied.

        Raises
        ──────
        ValueError — neither coupling_strength nor width_ratio is provided.
        """
        if coupling_strength is None and width_ratio is None:
            raise ValueError(
                "_interpolate_scan_grid requires either 'coupling_strength' "
                "or 'width_ratio'."
            )
        return self._load_single_grid_cross_section(
            heavy_mass=self._get_mass(),
            grid_key=grid_key,
            model_label=model_label,
            species=species,
            width_ratio=width_ratio,
            coupling_strength=coupling_strength,
        )
