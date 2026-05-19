from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate


FloatArray = NDArray[np.float64]


class TheoryDataError(RuntimeError):
    """Raised when theory cross-section JSON data cannot be resolved or parsed."""


@dataclass(frozen=True, slots=True)
class TheoryCurve:
    """One-dimensional theory cross-section curve σ(m)."""

    masses: FloatArray
    cross_sections: FloatArray

    def __post_init__(self) -> None:
        if len(self.masses) != len(self.cross_sections):
            raise TheoryDataError("TheoryCurve requires masses and cross_sections arrays of the same length.")


@dataclass(frozen=True, slots=True)
class TheoryGrid:
    """Scattered two-dimensional theory grid σ(m, s)."""

    masses: FloatArray
    scan_values: FloatArray
    cross_sections: FloatArray

    def __post_init__(self) -> None:
        if not (len(self.masses) == len(self.scan_values) == len(self.cross_sections)):
            raise TheoryDataError("TheoryGrid arrays must have the same length.")


def interpolate_cross_section_1d(mass_grid: FloatArray, target_mass: float, cross_sections: FloatArray) -> float:
    """
    Linear interpolation for σ(m); returns -1.0 when the target mass lies outside the grid.
    """
    if np.min(mass_grid) <= target_mass <= np.max(mass_grid):
        interpolator = interpolate.interp1d(mass_grid, cross_sections, kind="linear")
        return float(interpolator(target_mass))
    return -1.0


def interpolate_curve_dense(x: FloatArray, y: FloatArray, dense_x: FloatArray) -> FloatArray:
    """Dense linear interpolation along one axis."""
    interpolator = interpolate.interp1d(x, y, kind="linear")
    return np.asarray(interpolator(dense_x), dtype=np.float64)


def build_surface_interpolator(
    mass_arrays: Sequence[FloatArray],
    scan_values: Sequence[float],
    surfaces: Sequence[FloatArray],
    indices: Sequence[int],
) -> interpolate.LinearNDInterpolator:
    """
    Build a 2D interpolator from a list of one-dimensional slices.
    """
    stitched_masses: List[FloatArray] = []
    stitched_scan_values: List[FloatArray] = []
    stitched_cross_sections: List[FloatArray] = []

    for index, scan_value in zip(indices, scan_values):
        local_masses = np.asarray(mass_arrays[index], dtype=np.float64)
        extended_mass_grid = np.linspace(np.min(local_masses), np.max(local_masses), 100)
        stitched_masses.append(extended_mass_grid)
        stitched_scan_values.append(np.full_like(extended_mass_grid, float(scan_value)))
        stitched_cross_sections.append(interpolate_curve_dense(local_masses, np.asarray(surfaces[index], dtype=np.float64), extended_mass_grid))

    all_masses = np.concatenate(stitched_masses)
    all_scan_values = np.concatenate(stitched_scan_values)
    all_cross_sections = np.concatenate(stitched_cross_sections)
    return interpolate.LinearNDInterpolator(list(zip(all_masses, all_scan_values)), all_cross_sections)


def build_rectangular_surface_interpolator(
    masses: FloatArray,
    scan_axis: Sequence[float],
    cross_sections: FloatArray,
) -> interpolate.LinearNDInterpolator:
    """
    Build a generic 2D interpolator from flattened (mass, scan_value, sigma) data.
    """
    scan_axis = list(scan_axis)
    repeated_axis: List[FloatArray] = []
    slice_length = int(len(masses) / len(scan_axis))
    for value in scan_axis:
        repeated_axis.append(float(value) * np.ones(len(masses[:slice_length])))
    flattened_scan_axis = np.asarray(repeated_axis, dtype=np.float64).flatten()
    return interpolate.LinearNDInterpolator(list(zip(masses, flattened_scan_axis)), cross_sections)


def _normalize_json_name(file_name: str) -> str:
    return str(Path(file_name).with_suffix(".json"))


def _resolve_data_root(data_root: Optional[Union[str, Path]]) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent.parent
    candidates = (
        cwd / "data",
        project_root / "data",
        module_dir.parent / "data",
        cwd,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return cwd / "data"


def _candidate_theory_paths(file_name: str, vlq_species: str, data_root: Path) -> Iterable[Path]:
    normalized_name = _normalize_json_name(file_name)
    species = vlq_species.upper()

    # New project layout: data/<species>/Theo_Tables/*.json
    yield data_root / species / "Theo_Tables" / normalized_name
    # Additional supported repository layouts
    yield data_root / "theory_predictions" / species / normalized_name
    yield data_root / "theory_predictions" / species.lower() / normalized_name
    yield data_root / f"{species}data" / "Theo_Tables" / normalized_name

    # Shared pair-production grids such as pp_QQ_NNLO.json are global in practice.
    # Search every species Theo_Tables directory so all VLQ species can reuse the same file.
    for shared_species in ("T", "B", "X", "Y"):
        yield data_root / shared_species / "Theo_Tables" / normalized_name

    # Direct root-level lookup paths
    yield data_root / normalized_name
    yield data_root / "Theo_Tables" / normalized_name


@lru_cache(maxsize=512)
def _load_json_payload(file_path: str) -> Any:
    with Path(file_path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_theory_file(file_name: str, vlq_species: str, data_root: Optional[Union[str, Path]] = None) -> Path:
    root = _resolve_data_root(data_root)
    normalized_name = _normalize_json_name(file_name)

    for candidate in _candidate_theory_paths(file_name, vlq_species, root):
        if candidate.exists():
            return candidate.resolve()

    # Recursive repository search for valid JSON matches when the file is stored
    # outside the canonical lookup paths.
    matches = list(root.rglob(normalized_name)) if root.exists() else []
    if matches:
        # Prefer Theo_Tables matches, then exact-species folders.
        def _priority(path: Path) -> tuple[int, int, str]:
            parts = path.parts
            score_theo = 0 if "Theo_Tables" in parts else 1
            score_species = 0 if vlq_species.upper() in parts else 1
            return (score_theo, score_species, str(path))
        matches.sort(key=_priority)
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not resolve theory grid file '{file_name}' for species '{vlq_species}' under data root '{root}'."
    )


def _extract_nested_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        for key in ("data", "dataset", "curve", "grid", "payload", "records", "rows", "points", "entries", "values", "content"):
            if key in payload:
                return payload[key]
    return payload


def _parse_theory_curve(payload: Any) -> TheoryCurve:
    payload = _extract_nested_payload(payload)

    if isinstance(payload, Mapping):
        masses = payload.get("mass", payload.get("masses", payload.get("x")))
        cross_sections = payload.get("cross_section", payload.get("cross_sections", payload.get("sigma", payload.get("xs"))))
        if masses is not None and cross_sections is not None:
            return TheoryCurve(
                masses=np.asarray(masses, dtype=np.float64),
                cross_sections=np.asarray(cross_sections, dtype=np.float64),
            )

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if not payload:
            raise TheoryDataError("Cannot parse an empty theory curve.")
        if all(isinstance(row, Mapping) for row in payload):
            masses = [row.get("mass", row.get("m", row.get("x"))) for row in payload]
            cross_sections = [row.get("cross_section", row.get("cross_sections", row.get("sigma", row.get("xs")))) for row in payload]
            return TheoryCurve(
                masses=np.asarray(masses, dtype=np.float64),
                cross_sections=np.asarray(cross_sections, dtype=np.float64),
            )
        matrix = np.asarray(payload, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise TheoryDataError("A theory curve sequence must contain at least [mass, cross_section].")
        return TheoryCurve(masses=matrix[:, 0], cross_sections=matrix[:, 1])

    raise TheoryDataError("Unsupported theory-curve JSON structure.")


def _parse_theory_grid(payload: Any) -> TheoryGrid:
    payload = _extract_nested_payload(payload)

    if isinstance(payload, Mapping):
        masses = payload.get("mass", payload.get("masses", payload.get("x")))
        scan_values = payload.get("scan", payload.get("coupling", payload.get("width_ratio", payload.get("y"))))
        cross_sections = payload.get("cross_section", payload.get("cross_sections", payload.get("sigma", payload.get("xs"))))
        if masses is not None and scan_values is not None and cross_sections is not None:
            return TheoryGrid(
                masses=np.asarray(masses, dtype=np.float64),
                scan_values=np.asarray(scan_values, dtype=np.float64),
                cross_sections=np.asarray(cross_sections, dtype=np.float64),
            )

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if not payload:
            raise TheoryDataError("Cannot parse an empty theory grid.")
        if all(isinstance(row, Mapping) for row in payload):
            masses = [row.get("mass", row.get("m", row.get("x"))) for row in payload]
            scan_values = [row.get("scan", row.get("coupling", row.get("width_ratio", row.get("y")))) for row in payload]
            cross_sections = [row.get("cross_section", row.get("cross_sections", row.get("sigma", row.get("xs")))) for row in payload]
            return TheoryGrid(
                masses=np.asarray(masses, dtype=np.float64),
                scan_values=np.asarray(scan_values, dtype=np.float64),
                cross_sections=np.asarray(cross_sections, dtype=np.float64),
            )
        matrix = np.asarray(payload, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] < 3:
            raise TheoryDataError("A theory grid sequence must contain at least [mass, scan_value, cross_section].")
        return TheoryGrid(masses=matrix[:, 0], scan_values=matrix[:, 1], cross_sections=matrix[:, 2])

    raise TheoryDataError("Unsupported theory-grid JSON structure.")


def interpolate_cross_section_from_curve_file(file_path: Union[str, Path], target_mass: float) -> float:
    """Resolve σ(m) directly from a concrete JSON curve file."""
    payload = _load_json_payload(str(Path(file_path).resolve()))
    curve = _parse_theory_curve(payload)
    return interpolate_cross_section_1d(curve.masses, target_mass, curve.cross_sections)


def load_theory_cross_section_from_repository(
    heavy_mass: float,
    file_name: str,
    vlq_species: str = "T",
    data_root: Optional[Union[str, Path]] = None,
) -> float:
    """
    Resolve and interpolate a one-dimensional theory cross-section curve from the repository.
    """
    file_path = _resolve_theory_file(file_name, vlq_species, data_root=data_root)
    return interpolate_cross_section_from_curve_file(file_path, heavy_mass)


def read_scattered_theory_grid(file_paths: Sequence[Union[str, Path]]) -> TheoryGrid:
    """
    Read multiple grid fragments and merge them into a single scattered theory grid.
    """
    masses: List[FloatArray] = []
    scan_values: List[FloatArray] = []
    cross_sections: List[FloatArray] = []

    for file_path in file_paths:
        payload = _load_json_payload(str(Path(file_path).resolve()))
        grid = _parse_theory_grid(payload)
        masses.append(grid.masses)
        scan_values.append(grid.scan_values)
        cross_sections.append(grid.cross_sections)

    return TheoryGrid(
        masses=np.concatenate(masses) if masses else np.asarray([], dtype=np.float64),
        scan_values=np.concatenate(scan_values) if scan_values else np.asarray([], dtype=np.float64),
        cross_sections=np.concatenate(cross_sections) if cross_sections else np.asarray([], dtype=np.float64),
    )


def load_scattered_theory_grid_by_pattern(
    file_key: str,
    model_label: str,
    vlq_species: str = "T",
    data_root: Optional[Union[str, Path]] = None,
) -> TheoryGrid:
    """
    Load all JSON theory-grid fragments whose file names match a given key/model pattern.
    """
    root = _resolve_data_root(data_root)
    search_roots = [
        root / vlq_species.upper() / "Theo_Tables",
        root / "theory_predictions" / vlq_species.upper(),
        root / "theory_predictions" / vlq_species.lower(),
        root / f"{vlq_species.upper()}data" / "Theo_Tables",
    ]

    matched_paths: List[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        matched_paths.extend(
            path for path in search_root.glob("*.json")
            if file_key in path.name and model_label in path.name
        )

    if not matched_paths:
        raise FileNotFoundError(
            f"No theory-grid JSON files were found for key='{file_key}', model='{model_label}', species='{vlq_species}'."
        )

    return read_scattered_theory_grid(matched_paths)


def interpolate_surface_value(
    *,
    experimental_input: Optional[Tuple[Sequence[int], Sequence[FloatArray], Sequence[float], Optional[Sequence[float]], Sequence[FloatArray], float, float, Optional[float]]] = None,
    theory_input: Optional[Tuple[FloatArray, FloatArray, Optional[FloatArray], FloatArray, float, float, Optional[float]]] = None,
    mode: str = "experimental",
) -> Any:
    """
    Unified two-dimensional interpolation interface for experimental grids
    and theory surfaces.

    Parameters are explicit and keyword-only in order to keep the evaluation
    logic readable and robust.
    """
    if mode == "experimental":
        if experimental_input is None:
            raise TheoryDataError("experimental_input is required when mode='experimental'.")
        (
            indices,
            mass_arrays,
            relative_width_values,
            coupling_values,
            surfaces,
            target_mass,
            target_width_ratio,
            target_coupling,
        ) = experimental_input

        if coupling_values is None:
            if min(relative_width_values) == 0.05:
                if target_width_ratio >= 0.05:
                    interpolator = build_surface_interpolator(mass_arrays, relative_width_values, surfaces, indices)
                    return interpolator(target_mass, target_width_ratio)
                one_dimensional = interpolate.interp1d(mass_arrays[indices[0]], surfaces[indices[0]], kind="linear")
                return one_dimensional(target_mass)

            if min(relative_width_values) == 0.01:
                if target_width_ratio >= 0.01:
                    interpolator = build_surface_interpolator(mass_arrays, relative_width_values, surfaces, indices)
                    return interpolator(target_mass, target_width_ratio)
                return -1.0

        interpolator = build_surface_interpolator(mass_arrays, coupling_values, surfaces, indices)
        return interpolator(target_mass, target_coupling)

    if mode == "theory":
        if theory_input is None:
            raise TheoryDataError("theory_input is required when mode='theory'.")
        (
            masses,
            scan_axis,
            coupling_values,
            cross_sections,
            target_mass,
            target_width_ratio,
            target_coupling,
        ) = theory_input

        if coupling_values is None:
            if min(scan_axis) == 0.05:
                if target_width_ratio >= 0.05:
                    interpolator = interpolate.LinearNDInterpolator(list(zip(masses, scan_axis)), cross_sections)
                    return interpolator(target_mass, target_width_ratio)
                mask = scan_axis == 0.05
                one_dimensional = interpolate.interp1d(masses[mask], cross_sections[mask], kind="linear")
                return one_dimensional(target_mass)

            if min(scan_axis) == 0.01:
                if target_width_ratio >= 0.01:
                    interpolator = interpolate.LinearNDInterpolator(list(zip(masses, scan_axis)), cross_sections)
                    return interpolator(target_mass, target_width_ratio)
                return -1.0

        interpolator = interpolate.LinearNDInterpolator(list(zip(masses, coupling_values)), cross_sections)
        return interpolator(target_mass, target_coupling)

    raise TheoryDataError(f"Unsupported interpolation mode: {mode}")
