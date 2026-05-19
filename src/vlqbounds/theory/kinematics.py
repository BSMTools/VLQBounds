from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from . import constants as c


FloatArray = NDArray[np.float64]


class FoundationDataError(RuntimeError):
    """Raised when a JSON payload or a curve layout cannot be interpreted safely."""


class ParameterValidationError(ValueError):
    """Raised when a user-facing parameter contract is violated."""


@dataclass(frozen=True, slots=True)
class LimitCurve:
    """One-dimensional exclusion curve in mass space."""

    masses: FloatArray
    observed: FloatArray
    expected: FloatArray

    def __post_init__(self) -> None:
        if len(self.masses) != len(self.observed) or len(self.masses) != len(self.expected):
            raise FoundationDataError(
                "LimitCurve requires masses, observed, and expected arrays with identical lengths."
            )


@dataclass(frozen=True, slots=True)
class CouplingEnvelope:
    """Observed/expected lower-upper envelopes for coupling limits."""

    observed_masses: FloatArray
    expected_masses: FloatArray
    observed_lower: FloatArray
    observed_upper: FloatArray
    expected_lower: FloatArray
    expected_upper: FloatArray

    def __post_init__(self) -> None:
        observed_lengths = {
            len(self.observed_masses),
            len(self.observed_lower),
            len(self.observed_upper),
        }
        expected_lengths = {
            len(self.expected_masses),
            len(self.expected_lower),
            len(self.expected_upper),
        }
        if len(observed_lengths) != 1 or len(expected_lengths) != 1:
            raise FoundationDataError("CouplingEnvelope arrays have inconsistent lengths.")


def mass_ratio(m1: float, m2: float) -> float:
    """Return the dimensionless mass ratio m1 / m2."""
    return m1 / m2


def compute_limit_ratios(theory_value: float, observed_limit: float, expected_limit: float) -> Tuple[float, float]:
    """Return (theory / observed_limit, theory / expected_limit)."""
    return theory_value / observed_limit, theory_value / expected_limit


def kallen_lambda(x: float, y: float, z: float) -> float:
    """Källén λ polynomial."""
    return x**4 + y**4 + z**4 - 2 * x**2 * y**2 - 2 * x**2 * z**2 - 2 * y**2 * z**2


def validate_nonzero_sine(value: float) -> None:
    """Validate that a mixing sine belongs to [-1, 1] and is not zero."""
    if value == 0:
        raise ParameterValidationError("Invalid mixing sine: the value must not be zero.")
    if not (-1.0 <= value <= 1.0):
        raise ParameterValidationError("Invalid mixing sine: expected a value in the range [-1, 1].")


def append_dataframe_row(frame: DataFrame, **row_payload: Any) -> DataFrame:
    """Return a new DataFrame with one appended row."""
    return pd.concat([frame, pd.DataFrame(row_payload, index=[0])], ignore_index=True)


def extract_global_mass_bounds(
    mass_arrays: Sequence[Optional[Sequence[float]]],
    indices: Sequence[int],
) -> Tuple[float, float]:
    """Return the global (min_mass, max_mass) over a selected set of mass arrays."""
    selected = [np.asarray(mass_arrays[index], dtype=float) for index in indices if mass_arrays[index] is not None]
    if not selected:
        raise FoundationDataError("No mass arrays were found for the requested indices.")
    minima = [float(np.min(arr)) for arr in selected]
    maxima = [float(np.max(arr)) for arr in selected]
    return min(minima), max(maxima)


def is_theory_cross_section_dominant(coupling_ratio: float, cross_section_ratio: float) -> bool:
    """Return True when the cross-section ratio is stronger than the coupling ratio."""
    return cross_section_ratio > coupling_ratio


def compute_vector_boson_decay_width(
    prefactor: float,
    heavy_mass: float,
    boson_mass: float,
    couplings: Sequence[float],
    daughter_quark_mass: float,
) -> float:
    """Generic partial width for Q -> q V, where V is W or Z."""
    c1, c2 = couplings
    lam = kallen_lambda(heavy_mass, daughter_quark_mass, boson_mass)
    rx_v = mass_ratio(boson_mass, heavy_mass)
    rx_q = mass_ratio(daughter_quark_mass, heavy_mass)
    kinematic_factor = (
        1
        + rx_v**2
        - 2 * rx_q**2
        - 2 * rx_v**4
        + rx_q**4
        + rx_q**2 * rx_v**2
    )
    return (
        prefactor
        * heavy_mass
        / (boson_mass**2)
        * np.sqrt(lam)
        * ((c1**2 + c2**2) * kinematic_factor - 12 * rx_v**2 * rx_q * c1 * c2)
    )


def compute_higgs_decay_width_from_kappa(
    prefactor: float,
    heavy_mass: float,
    kappa_effective: float,
    daughter_quark_mass: float,
) -> float:
    """Partial width for Q -> q h using an effective universal coupling."""
    return (
        prefactor
        * heavy_mass
        / (c.MW**2)
        * np.sqrt(kallen_lambda(heavy_mass, daughter_quark_mass, c.Mh))
        * kappa_effective**2
        * (1 + mass_ratio(daughter_quark_mass, heavy_mass) ** 2 - mass_ratio(c.Mh, heavy_mass) ** 2)
    )


def compute_higgs_decay_width_from_mixing(
    prefactor: float,
    heavy_mass: float,
    couplings: Union[float, Sequence[float], FloatArray],
    daughter_quark_mass: float,
) -> float:
    """Partial width for Q -> q h using explicit channel couplings."""
    if isinstance(couplings, np.ndarray):
        if couplings.ndim == 0:
            c1, c2 = float(couplings), 0.0
        else:
            c1, c2 = (float(couplings[0]), float(couplings[1])) if len(couplings) >= 2 else (float(couplings[0]), 0.0)
    elif isinstance(couplings, (list, tuple)):
        c1, c2 = (float(couplings[0]), float(couplings[1])) if len(couplings) >= 2 else (float(couplings[0]), 0.0)
    else:
        c1, c2 = float(couplings), 0.0

    return (
        prefactor
        * heavy_mass
        / (c.MW**2)
        * np.sqrt(kallen_lambda(heavy_mass, daughter_quark_mass, c.Mh))
        * (c1**2 + c2**2)
        * (
            1
            + 6 * mass_ratio(daughter_quark_mass, heavy_mass) ** 2
            - mass_ratio(c.Mh, heavy_mass) ** 2
            + mass_ratio(daughter_quark_mass, heavy_mass) ** 4
            - mass_ratio(daughter_quark_mass, heavy_mass) ** 2 * mass_ratio(c.Mh, heavy_mass) ** 2
        )
    )


def is_missing_sequence(sequence: Optional[Sequence[Any]]) -> bool:
    """Return True when the sequence is None or contains only None values."""
    return sequence is None or all(item is None for item in sequence)


def parse_constraint_expression(expression: str) -> Tuple[List[float], bool]:
    """Parse strings such as 'r==0.01' or '0.05<=k<=0.3'."""
    variable = "r" if "r" in expression else "k"
    if "==" in expression:
        numeric_values = [float(chunk) for chunk in expression.split("==") if chunk != variable]
        return numeric_values, True
    if "<=" in expression:
        numeric_values = [float(chunk) for chunk in expression.split("<=") if chunk != variable]
        return numeric_values, False
    raise ParameterValidationError(f"Unsupported constraint expression: {expression}")


def convert_contact_factor_to_kappa(
    contact_factor: NDArray[Any],
    heavy_mass: NDArray[Any],
    model: str,
) -> FloatArray:
    """Convert a contact-factor grid into the effective κ parameterisation."""
    cf = np.asarray(contact_factor, dtype=np.float64)
    masses = np.asarray(heavy_mass, dtype=np.float64)

    if model == "Singlet":
        return np.sqrt(2.0) * (((c.v / masses) + 2.0) * cf) / 3.0
    if model == "Doublet":
        return (((c.v / masses) + 1.0) * cf) / 2.0
    raise ParameterValidationError(f"Unsupported model for contact-factor conversion: {model}")


def solve_mixing_from_kappa(kappa: float) -> Tuple[float, float]:
    """Return the two physical sine solutions associated with a given κ value."""
    discriminant = np.sqrt(1.0 - 4.0 * kappa**2)
    return (
        np.sqrt(0.5 - 0.5 * discriminant),
        np.sqrt(0.5 + 0.5 * discriminant),
    )


def validate_parameter_contract(
    kwargs: Mapping[str, Any],
    required_mass_key: str,
    valid_companion_keys: Iterable[str],
) -> None:
    """Validate the public two-parameter runtime contract for model setters."""
    valid_companion_keys = set(valid_companion_keys)
    if len(kwargs) != 2:
        raise ParameterValidationError("Exactly two keyword arguments are required.")
    if required_mass_key not in kwargs or kwargs[required_mass_key] is None:
        raise ParameterValidationError(f"Required mass parameter '{required_mass_key}' is missing or None.")
    other_keys = set(kwargs) - {required_mass_key}
    if len(other_keys) != 1:
        raise ParameterValidationError(
            f"Exactly one companion parameter must accompany '{required_mass_key}'."
        )
    companion_key = next(iter(other_keys))
    if companion_key not in valid_companion_keys:
        raise ParameterValidationError(
            f"Invalid companion parameter '{companion_key}'. Allowed keys are: {sorted(valid_companion_keys)}."
        )


def _as_float_array(values: Sequence[Any]) -> FloatArray:
    return np.asarray(list(values), dtype=np.float64)


def _resolve_data_root(data_root: Optional[Union[str, Path]]) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()

    module_dir = Path(__file__).resolve().parent
    package_root = module_dir.parent.parent
    candidates = (
        package_root / "data",
        Path.cwd().resolve() / "data",
        Path.cwd().resolve(),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return package_root / "data"


def _extract_nested_payload(payload: Mapping[str, Any]) -> Optional[Any]:
    for nested_key in (
        "data",
        "dataset",
        "curve",
        "payload",
        "limit_curve",
        "content",
        "records",
        "rows",
        "points",
        "values",
    ):
        if nested_key in payload:
            return payload[nested_key]
    return None


def _parse_limit_curve_payload(payload: Any) -> LimitCurve:
    if isinstance(payload, Mapping):
        nested = _extract_nested_payload(payload)
        if nested is not None and nested is not payload:
            try:
                return _parse_limit_curve_payload(nested)
            except FoundationDataError:
                pass

        if "mass" in payload or "masses" in payload or "x" in payload:
            masses = payload.get("mass", payload.get("masses", payload.get("x")))
            observed = payload.get("observed", payload.get("obs", payload.get("limit")))
            expected = payload.get("expected", payload.get("exp"))
            if masses is None or observed is None:
                raise FoundationDataError("A limit-curve JSON object must define mass and observed values.")
            if expected is None:
                expected = [np.nan] * len(masses)
            return LimitCurve(
                masses=_as_float_array(masses),
                observed=_as_float_array(observed),
                expected=_as_float_array(expected),
            )

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if not payload:
            raise FoundationDataError("Cannot parse an empty limit-curve sequence.")
        if all(isinstance(entry, Mapping) for entry in payload):
            masses = [entry.get("mass", entry.get("m", entry.get("x"))) for entry in payload]
            observed = [entry.get("observed", entry.get("obs", entry.get("limit"))) for entry in payload]
            expected = [entry.get("expected", entry.get("exp", np.nan)) for entry in payload]
            return LimitCurve(
                masses=_as_float_array(masses),
                observed=_as_float_array(observed),
                expected=_as_float_array(expected),
            )

        masses: List[float] = []
        observed: List[float] = []
        expected: List[float] = []
        for row in payload:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                raise FoundationDataError("Each row in a limit curve must be a sequence.")
            if len(row) == 2:
                masses.append(float(row[0]))
                observed.append(float(row[1]))
                expected.append(np.nan)
            elif len(row) >= 3:
                masses.append(float(row[0]))
                observed.append(float(row[1]))
                expected.append(float(row[2]))
            else:
                raise FoundationDataError("Curve rows must contain at least two entries.")
        return LimitCurve(
            masses=_as_float_array(masses),
            observed=_as_float_array(observed),
            expected=_as_float_array(expected),
        )

    raise FoundationDataError("Unsupported limit-curve JSON structure.")


def _parse_coupling_envelope_payload(payload: Any) -> CouplingEnvelope:
    if isinstance(payload, Mapping):
        nested = _extract_nested_payload(payload)
        if nested is not None and nested is not payload:
            try:
                return _parse_coupling_envelope_payload(nested)
            except FoundationDataError:
                pass

        mass_obs = payload.get("observed_mass", payload.get("mass_observed", payload.get("m_obs")))
        mass_exp = payload.get("expected_mass", payload.get("mass_expected", payload.get("m_exp")))
        obs_lo = payload.get("observed_lower", payload.get("obs_lower", payload.get("obs_lo")))
        obs_hi = payload.get("observed_upper", payload.get("obs_upper", payload.get("obs_hi")))
        exp_lo = payload.get("expected_lower", payload.get("exp_lower", payload.get("exp_lo")))
        exp_hi = payload.get("expected_upper", payload.get("exp_upper", payload.get("exp_hi")))

        if obs_lo is not None and obs_hi is not None and exp_lo is not None and exp_hi is not None:
            if mass_obs is None and mass_exp is None and "mass" in payload:
                mass_obs = mass_exp = payload["mass"]
            if mass_obs is None or mass_exp is None:
                raise FoundationDataError("Coupling-envelope JSON requires observed and expected mass arrays.")
            return CouplingEnvelope(
                observed_masses=_as_float_array(mass_obs),
                expected_masses=_as_float_array(mass_exp),
                observed_lower=_as_float_array(obs_lo),
                observed_upper=_as_float_array(obs_hi),
                expected_lower=_as_float_array(exp_lo),
                expected_upper=_as_float_array(exp_hi),
            )

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        rows = [list(row) for row in payload]
        if not rows:
            raise FoundationDataError("Cannot parse an empty coupling-envelope sequence.")
        width = len(rows[0])
        if width == 6:
            columns = np.asarray(rows, dtype=np.float64)
            return CouplingEnvelope(
                observed_masses=columns[:, 0],
                expected_masses=columns[:, 1],
                observed_lower=columns[:, 2],
                observed_upper=columns[:, 3],
                expected_lower=columns[:, 4],
                expected_upper=columns[:, 5],
            )
        if width == 5:
            columns = np.asarray(rows, dtype=np.float64)
            masses = columns[:, 0]
            return CouplingEnvelope(
                observed_masses=masses,
                expected_masses=masses,
                observed_lower=columns[:, 1],
                observed_upper=columns[:, 2],
                expected_lower=columns[:, 3],
                expected_upper=columns[:, 4],
            )
        if width == 4:
            columns = np.asarray(rows, dtype=np.float64)
            return CouplingEnvelope(
                observed_masses=columns[:, 0],
                expected_masses=columns[:, 1],
                observed_lower=columns[:, 2],
                observed_upper=columns[:, 2],
                expected_lower=columns[:, 3],
                expected_upper=columns[:, 3],
            )
        if width == 3:
            columns = np.asarray(rows, dtype=np.float64)
            masses = columns[:, 0]
            return CouplingEnvelope(
                observed_masses=masses,
                expected_masses=masses,
                observed_lower=columns[:, 1],
                observed_upper=columns[:, 1],
                expected_lower=columns[:, 2],
                expected_upper=columns[:, 2],
            )

    raise FoundationDataError("Unsupported coupling-envelope JSON structure.")


@lru_cache(maxsize=256)
def _read_json_payload(file_path: str) -> Any:
    with Path(file_path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_limit_curve_path(
    file_name: str,
    experiment: str,
    vlq_species: str,
    data_root: Optional[Union[str, Path]] = None,
) -> Path:
    root = _resolve_data_root(data_root)

    candidate_names = [file_name]
    if not file_name.endswith(".json"):
        candidate_names.append(str(Path(file_name).with_suffix(".json")))

    species = str(vlq_species).upper()
    subdirectories = (
        "",
        f"{species}/ATLAS_Tables",
        f"{species}/CMS_Tables",
        f"{species}/Theo_Tables",
        "B/ATLAS_Tables", "B/CMS_Tables", "B/Theo_Tables",
        "T/ATLAS_Tables", "T/CMS_Tables", "T/Theo_Tables",
        "X/ATLAS_Tables", "X/CMS_Tables", "X/Theo_Tables",
        "Y/ATLAS_Tables", "Y/CMS_Tables", "Y/Theo_Tables",
    )

    for name in candidate_names:
        absolute = Path(name)
        if absolute.is_absolute() and absolute.exists():
            return absolute.resolve()

    for name in candidate_names:
        for subdir in subdirectories:
            candidate = root / subdir / name if subdir else root / name
            if candidate.exists():
                return candidate.resolve()

    try:
        for found in root.rglob(candidate_names[-1]):
            if found.is_file():
                return found.resolve()
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not resolve experimental limit file '{file_name}' for {experiment} under data root '{root}'."
    )


def load_experimental_limit_curves(
    file_names: Sequence[str],
    experiments: Sequence[str],
    vlq_species: str = "T",
    data_root: Optional[Union[str, Path]] = None,
) -> List[LimitCurve]:
    """Load a list of JSON experimental limit curves."""
    curves: List[LimitCurve] = []
    for file_name, experiment in zip(file_names, experiments):
        path = _resolve_limit_curve_path(file_name, experiment, vlq_species, data_root=data_root)
        curves.append(_parse_limit_curve_payload(_read_json_payload(str(path))))
    return curves


def load_coupling_limit_curves(
    file_names: Sequence[str],
    experiments: Sequence[str],
    vlq_species: str = "T",
    data_root: Optional[Union[str, Path]] = None,
) -> List[CouplingEnvelope]:
    """Load a list of JSON coupling-envelope files."""
    envelopes: List[CouplingEnvelope] = []
    for file_name, experiment in zip(file_names, experiments):
        path = _resolve_limit_curve_path(file_name, experiment, vlq_species, data_root=data_root)
        envelopes.append(_parse_coupling_envelope_payload(_read_json_payload(str(path))))
    return envelopes


def _normalise_coupling_sequence(
    coupling: Union[float, Sequence[float], FloatArray],
) -> List[float]:
    """Return the decay-coupling object as a two-component Python list."""
    if isinstance(coupling, np.ndarray):
        if coupling.ndim == 0:
            return [float(coupling), 0.0]
        values = coupling.astype(np.float64, copy=False).ravel().tolist()
        if not values:
            raise ParameterValidationError("Decay coupling array is empty.")
        return [float(values[0]), float(values[1]) if len(values) >= 2 else 0.0]

    if isinstance(coupling, (list, tuple)):
        values = list(coupling)
        if not values:
            raise ParameterValidationError("Decay coupling sequence is empty.")
        return [float(values[0]), float(values[1]) if len(values) >= 2 else 0.0]

    return [float(coupling), 0.0]


def evaluate_partial_width(
    involved_masses: Sequence[float],
    *,
    mixing_sine: Optional[float] = None,
    kappa: Optional[float] = None,
    width_ratio: Optional[float] = None,
    coupling: Optional[Union[float, Sequence[float], FloatArray]] = None,
    decay_prefactor: float,
    model: str = "Singlet",
    final_state_is_higgs: bool = False,
    species: str = "T",
) -> float:
    """Unified partial-width evaluator for the VLQ decay sector."""
    heavy_mass, boson_mass, daughter_mass = map(float, involved_masses)

    if coupling is not None:
        coupling_sequence = _normalise_coupling_sequence(coupling)
        if final_state_is_higgs:
            return compute_higgs_decay_width_from_mixing(
                decay_prefactor,
                heavy_mass,
                coupling_sequence,
                daughter_mass,
            )
        return compute_vector_boson_decay_width(
            decay_prefactor,
            heavy_mass,
            boson_mass,
            coupling_sequence,
            daughter_mass,
        )

    if kappa is not None:
        effective_kappa = kappa / np.sqrt(2.0) if model in {"Singlet", "Triplet"} else kappa
        if final_state_is_higgs:
            return compute_higgs_decay_width_from_kappa(
                decay_prefactor,
                heavy_mass,
                effective_kappa,
                daughter_mass,
            )
        return compute_vector_boson_decay_width(
            decay_prefactor,
            heavy_mass,
            boson_mass,
            [effective_kappa, 0.0],
            daughter_mass,
        )

    if width_ratio is not None:
        inferred_kappa = infer_kappa_from_width_ratio(
            heavy_mass=heavy_mass,
            width_ratio=width_ratio,
            species=species,
            model=model,
        )
        effective_kappa = inferred_kappa / np.sqrt(2.0) if model in {"Singlet", "Triplet"} else inferred_kappa
        if final_state_is_higgs:
            return compute_higgs_decay_width_from_kappa(
                decay_prefactor,
                heavy_mass,
                effective_kappa,
                daughter_mass,
            )
        return compute_vector_boson_decay_width(
            decay_prefactor,
            heavy_mass,
            boson_mass,
            [effective_kappa, 0.0],
            daughter_mass,
        )

    raise ParameterValidationError(
        "A partial-width evaluation requires an explicit decay coupling, "
        "or a universal coupling kappa, or a width-to-mass ratio."
    )


def _sum_denominator_terms(heavy_mass: float, species: str, model: str) -> float:
    species = species.upper()
    model = model.title()

    if species == "T":
        width_w = compute_vector_boson_decay_width(c.Cst1, heavy_mass, c.MW, [1.0, 0.0], c.Mb)
        width_z = compute_vector_boson_decay_width(c.Cst2, heavy_mass, c.MZ, [1.0, 0.0], c.Mt)
        width_h = compute_higgs_decay_width_from_kappa(c.Cst3, heavy_mass, 1.0, c.Mt)
        if model == "Singlet":
            return width_w + width_z + width_h
        if model == "Doublet":
            return width_z + width_h
    elif species == "B":
        width_w = compute_vector_boson_decay_width(c.Cst1, heavy_mass, c.MW, [1.0, 0.0], c.Mt)
        width_z = compute_vector_boson_decay_width(c.Cst2, heavy_mass, c.MZ, [1.0, 0.0], c.Mb)
        width_h = compute_higgs_decay_width_from_kappa(c.Cst3, heavy_mass, 1.0, c.Mb)
        if model == "Singlet":
            return width_w + width_z + width_h
        if model == "Doublet":
            return width_z + width_h
    elif species == "X":
        return compute_vector_boson_decay_width(c.Cst1, heavy_mass, c.MW, [1.0, 0.0], c.Mt)
    elif species == "Y":
        return compute_vector_boson_decay_width(c.Cst1, heavy_mass, c.MW, [1.0, 0.0], c.Mb)

    raise ParameterValidationError(f"Unsupported species/model combination for width inversion: {species}, {model}")


def infer_kappa_from_width_ratio(
    heavy_mass: float,
    width_ratio: float,
    species: str = "T",
    model: str = "Singlet",
) -> float:
    """Infer the universal κ value from a width-to-mass ratio."""
    denominator = _sum_denominator_terms(float(heavy_mass), species.upper(), model.title())
    prefactor = np.sqrt(2.0) if model.title() in {"Singlet", "Triplet"} else 1.0
    return prefactor * np.sqrt(width_ratio * heavy_mass / denominator)


__all__ = [
    "FloatArray",
    "FoundationDataError",
    "ParameterValidationError",
    "LimitCurve",
    "CouplingEnvelope",
    "mass_ratio",
    "compute_limit_ratios",
    "kallen_lambda",
    "validate_nonzero_sine",
    "append_dataframe_row",
    "extract_global_mass_bounds",
    "is_theory_cross_section_dominant",
    "compute_vector_boson_decay_width",
    "compute_higgs_decay_width_from_kappa",
    "compute_higgs_decay_width_from_mixing",
    "is_missing_sequence",
    "parse_constraint_expression",
    "convert_contact_factor_to_kappa",
    "solve_mixing_from_kappa",
    "validate_parameter_contract",
    "load_experimental_limit_curves",
    "load_coupling_limit_curves",
    "evaluate_partial_width",
    "infer_kappa_from_width_ratio",
]
