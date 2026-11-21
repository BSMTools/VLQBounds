
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import json
import warnings
import numpy as np
from numpy.typing import NDArray

from vlqbounds.models import (
    BottomPartnerDoublet,
    TopPartnerDoublet,
    XPartnerDoublet,
    YPartnerDoublet,
    BottomPartnerPure,
    TopPartnerPure,
    BottomPartnerSinglet,
    TopPartnerSinglet,
    YPartnerTriplet,
)

VLQModel = Union[TopPartnerPure, TopPartnerSinglet, TopPartnerDoublet, BottomPartnerPure, BottomPartnerSinglet, BottomPartnerDoublet, XPartnerDoublet, YPartnerDoublet, YPartnerTriplet]
FloatArray = NDArray[np.float64]


EXPECTED_DATASET_COUNTS: Dict[Tuple[str, str], int] = {
    ("B", "Singlet"): 23,
    ("B", "Doublet"): 21,
    ("B", "Pure"): 17,
    ("X", "Doublet"): 14,
    ("Y", "Doublet"): 8,
    ("Y", "Triplet"): 4,
    ("T", "Singlet"): 61,
    ("T", "Doublet"): 33,
    ("T", "Pure"): 31,
}


class VLQSpecies(str, Enum):
    TOP_LIKE = "T"
    BOTTOM_LIKE = "B"
    X_LIKE = "X"
    Y_LIKE = "Y"


class ProductionMode(str, Enum):
    PAIR = "pair_prod"
    SINGLE = "single_prod"


class BoundsCatalogError(RuntimeError):
    """Base exception for metadata, I/O, and schema errors raised by this module."""


class RegistryConsistencyError(BoundsCatalogError):
    """Raised when the embedded registry violates strict consistency checks."""


@dataclass(frozen=True, slots=True)
class ExperimentalLimitDescriptor:
    """
    Immutable metadata container describing one experimental limit curve.

    Each dataset is represented by one structured object so the runtime can
    materialize analysis metadata and numerical content from a single source
    of truth.
    """

    index: int
    species: str
    scenario: str
    dataset_key: str
    citation: str
    experiment: str
    file_name: str
    process: str
    energy_tev: int
    luminosity_fbinv: float
    doublet_family: Optional[str] = None

    @property
    def json_file_name(self) -> str:
        return str(Path(self.file_name).with_suffix(".json"))

    @property
    def production_mode(self) -> ProductionMode:
        process_head = self.process[:9]
        if process_head in {"pp --> TT", "pp --> BB", "pp --> XX", "pp --> YY"}:
            return ProductionMode.PAIR
        return ProductionMode.SINGLE

    @property
    def channel_signature(self) -> str:
        return f"{self.experiment}:{self.process}:{self.energy_tev}:{self.luminosity_fbinv}"

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "species": self.species,
            "scenario": self.scenario,
            "dataset_key": self.dataset_key,
            "citation": self.citation,
            "experiment": self.experiment,
            "file_name": self.file_name,
            "json_file_name": self.json_file_name,
            "process": self.process,
            "energy_tev": self.energy_tev,
            "luminosity_fbinv": self.luminosity_fbinv,
            "doublet_family": self.doublet_family,
            "production_mode": self.production_mode.value,
        }


@dataclass(frozen=True, slots=True)
class LimitCurveData:
    """Numerical payload of one limit curve loaded from a JSON file."""

    masses: FloatArray
    expected: FloatArray
    observed: FloatArray

    def __post_init__(self) -> None:
        if len(self.masses) != len(self.expected) or len(self.masses) != len(self.observed):
            raise BoundsCatalogError(
                "Inconsistent curve lengths: masses, expected, and observed must have the same size."
            )

    @property
    def size(self) -> int:
        return int(len(self.masses))

    def to_mapping(self) -> Dict[str, List[float]]:
        return {
            "masses": self.masses.astype(float).tolist(),
            "expected": self.expected.astype(float).tolist(),
            "observed": self.observed.astype(float).tolist(),
        }


@dataclass(frozen=True, slots=True)
class RegistryDiagnostics:
    """Compact summary of registry health checks."""

    duplicate_dataset_keys: Tuple[str, ...]
    duplicate_filenames: Tuple[str, ...]
    missing_doublet_family: Tuple[str, ...]
    non_monotonic_indices: Tuple[str, ...]
    duplicate_channel_signatures: Tuple[str, ...]

    @property
    def has_issues(self) -> bool:
        return any(
            (
                self.duplicate_dataset_keys,
                self.duplicate_filenames,
                self.missing_doublet_family,
                self.non_monotonic_indices,
                self.duplicate_channel_signatures,
            )
        )

    def to_mapping(self) -> Dict[str, List[str]]:
        return {
            "duplicate_dataset_keys": list(self.duplicate_dataset_keys),
            "duplicate_filenames": list(self.duplicate_filenames),
            "missing_doublet_family": list(self.missing_doublet_family),
            "non_monotonic_indices": list(self.non_monotonic_indices),
            "duplicate_channel_signatures": list(self.duplicate_channel_signatures),
        }




@dataclass(frozen=True, slots=True)
class MissingExperimentalLimitRecord:
    """Detailed record for one registry entry whose JSON file could not be located."""

    index: int
    species: str
    scenario: str
    dataset_key: str
    citation: str
    experiment: str
    file_name: str
    process: str
    energy_tev: int
    luminosity_fbinv: float
    doublet_family: Optional[str] = None

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "species": self.species,
            "scenario": self.scenario,
            "dataset_key": self.dataset_key,
            "citation": self.citation,
            "experiment": self.experiment,
            "file_name": self.file_name,
            "process": self.process,
            "energy_tev": self.energy_tev,
            "luminosity_fbinv": self.luminosity_fbinv,
            "doublet_family": self.doublet_family,
        }


class JsonLimitCurveReader:
    """
    Smart JSON reader with flexible schema support and transparent caching.

    Supported payload shapes include:
      - list of points: [{"mass": ..., "expected": ..., "observed": ...}, ...]
      - list of rows: [[mass, obs], ...] or [[mass, obs, exp, ...], ...]
        (the column order used by every table shipped in data/)
      - flat arrays: {"mass": [...], "expected": [...], "observed": [...]}
      - nested payloads under keys such as data, curve, payload, limit_curve
      - column-oriented layouts with explicit field maps
    """

    _CACHE: Dict[Tuple[str, float], LimitCurveData] = {}

    def __init__(self, search_roots: Sequence[Path]):
        self.search_roots: Tuple[Path, ...] = tuple(Path(root).expanduser().resolve() for root in search_roots)

    def resolve(self, file_name: str) -> Path:
        candidate = Path(file_name)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        candidate_names = [file_name]
        if not file_name.endswith(".json"):
            candidate_names.append(str(Path(file_name).with_suffix(".json")))

        subdirectories = (
            "",
            "data",
            "limits",
            "tables",
            "exp_tables",
            "ATLAS",
            "CMS",
            "atlas",
            "cms",
            "B/ATLAS_Tables",
            "B/CMS_Tables",
            "B/Theo_Tables",
            "T/ATLAS_Tables",
            "T/CMS_Tables",
            "T/Theo_Tables",
            "X/ATLAS_Tables",
            "X/CMS_Tables",
            "X/Theo_Tables",
            "Y/ATLAS_Tables",
            "Y/CMS_Tables",
            "Y/Theo_Tables",
            "data/B/ATLAS_Tables",
            "data/B/CMS_Tables",
            "data/B/Theo_Tables",
            "data/T/ATLAS_Tables",
            "data/T/CMS_Tables",
            "data/T/Theo_Tables",
            "data/X/ATLAS_Tables",
            "data/X/CMS_Tables",
            "data/X/Theo_Tables",
            "data/Y/ATLAS_Tables",
            "data/Y/CMS_Tables",
            "data/Y/Theo_Tables",
            "tables/B/ATLAS_Tables",
            "tables/B/CMS_Tables",
            "tables/B/Theo_Tables",
            "tables/T/ATLAS_Tables",
            "tables/T/CMS_Tables",
            "tables/T/Theo_Tables",
            "tables/X/ATLAS_Tables",
            "tables/X/CMS_Tables",
            "tables/X/Theo_Tables",
            "tables/Y/ATLAS_Tables",
            "tables/Y/CMS_Tables",
            "tables/Y/Theo_Tables",
        )
        for root in self.search_roots:
            for name in candidate_names:
                for subdir in subdirectories:
                    path = root / subdir / name if subdir else root / name
                    if path.exists():
                        return path
        raise FileNotFoundError(f"Could not locate JSON limit file: {file_name}")

    @staticmethod
    def _as_float_array(values: Sequence[Any]) -> FloatArray:
        if isinstance(values, np.ndarray):
            return values.astype(np.float64, copy=False)
        return np.asarray(list(values), dtype=np.float64)

    @staticmethod
    def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        raise KeyError(keys[0])

    def _from_point_sequence(self, points: Sequence[Any]) -> LimitCurveData:
        if not points:
            raise BoundsCatalogError("JSON payload contains an empty point sequence.")

        if all(isinstance(point, Mapping) for point in points):
            masses, expected, observed = [], [], []
            for point in points:
                masses.append(self._first_present(point, "mass", "m", "x"))
                expected.append(point.get("expected", point.get("exp", point.get("y_exp", np.nan))))
                observed.append(point.get("observed", point.get("obs", point.get("y_obs", point.get("limit", np.nan)))))
            return LimitCurveData(
                masses=self._as_float_array(masses),
                expected=self._as_float_array(expected),
                observed=self._as_float_array(observed),
            )

        masses, expected, observed = [], [], []
        for row in points:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)) or len(row) < 2:
                raise BoundsCatalogError(
                    "Each point row must contain at least [mass, observed] or [mass, observed, expected]."
                )
            masses.append(row[0])
            observed.append(row[1])
            expected.append(np.nan if len(row) == 2 else row[2])

        return LimitCurveData(
            masses=self._as_float_array(masses),
            expected=self._as_float_array(expected),
            observed=self._as_float_array(observed),
        )

    def _try_nested_payloads(self, payload: Mapping[str, Any]) -> Optional[LimitCurveData]:
        for nested_key in ("data", "dataset", "curve", "payload", "limit_curve", "content"):
            nested = payload.get(nested_key)
            if isinstance(nested, Mapping):
                try:
                    return self.parse(nested)
                except BoundsCatalogError:
                    pass
            elif isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
                try:
                    return self._from_point_sequence(nested)
                except BoundsCatalogError:
                    pass
        return None

    def parse(self, payload: Any) -> LimitCurveData:
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            return self._from_point_sequence(payload)

        if not isinstance(payload, Mapping):
            raise BoundsCatalogError("Unsupported JSON payload structure for a limit curve.")

        nested = self._try_nested_payloads(payload)
        if nested is not None:
            return nested

        for point_key in ("points", "rows", "entries", "limits", "values"):
            points = payload.get(point_key)
            if isinstance(points, Sequence) and not isinstance(points, (str, bytes, bytearray)):
                return self._from_point_sequence(points)

        masses = None
        for key in ("mass", "masses", "x", "x_values"):
            if key in payload:
                masses = payload[key]
                break

        if masses is not None:
            expected = payload.get("expected", payload.get("exp", payload.get("y_exp")))
            observed = payload.get("observed", payload.get("obs", payload.get("y_obs", payload.get("limit"))))
            if observed is None:
                raise BoundsCatalogError("JSON payload contains masses but no observed limit array.")
            if expected is None:
                expected = [np.nan] * len(masses)
            return LimitCurveData(
                masses=self._as_float_array(masses),
                expected=self._as_float_array(expected),
                observed=self._as_float_array(observed),
            )

        columns = payload.get("columns")
        records = payload.get("records")
        if isinstance(columns, Sequence) and isinstance(records, Sequence):
            index_by_name = {str(name): index for index, name in enumerate(columns)}
            translated_rows = []
            for row in records:
                if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                    raise BoundsCatalogError("Column-oriented JSON records must be row sequences.")
                mass_index = index_by_name.get("mass", index_by_name.get("m", index_by_name.get("x")))
                obs_index = index_by_name.get("observed", index_by_name.get("obs", index_by_name.get("limit")))
                exp_index = index_by_name.get("expected")
                if mass_index is None or obs_index is None:
                    raise BoundsCatalogError("Column-oriented JSON records must define mass and observed columns.")
                translated_rows.append({
                    "mass": row[mass_index],
                    "expected": row[exp_index] if exp_index is not None else np.nan,
                    "observed": row[obs_index],
                })
            return self._from_point_sequence(translated_rows)

        raise BoundsCatalogError("Unable to infer mass, expected, and observed arrays from the JSON payload.")

    def read(self, file_name: str) -> LimitCurveData:
        path = self.resolve(file_name)
        cache_key = (str(path), path.stat().st_mtime)
        cached = self._CACHE.get(cache_key)
        if cached is not None:
            return cached

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        curve = self.parse(payload)
        self._CACHE[cache_key] = curve
        return curve


class ExperimentalBoundsRepository:
    """
    Structured repository for experimental VLQ limit inputs.

    The repository loads JSON limit curves, materializes the runtime arrays
    used by the evaluation engines, and exposes grouped metadata for pair and
    single production channels.
    """

    def __init__(
        self,
        model: VLQModel,
        data_directory: Optional[Union[str, Path]] = None,
        *,
        strict_registry: bool = False,
    ) -> None:
        self.model: VLQModel = model
        self.m: VLQModel = model
        self.strict_registry = strict_registry

        self.data_directory = self._resolve_data_directory(data_directory)
        self.reader = JsonLimitCurveReader(self._build_search_roots(self.data_directory))

        self.MY: Optional[List[Optional[FloatArray]]] = None
        self.MX: Optional[List[Optional[FloatArray]]] = None
        self.MB: Optional[List[Optional[FloatArray]]] = None
        self.MT: Optional[List[Optional[FloatArray]]] = None

        self.exp: Optional[List[Optional[FloatArray]]] = None
        self.obs: Optional[List[Optional[FloatArray]]] = None
        self.which_doublet: Optional[List[Optional[str]]] = None

        self.file_name: Optional[List[Optional[str]]] = None
        self.key: Optional[List[Optional[str]]] = None
        self.label: Optional[List[Optional[str]]] = None
        self.expt: Optional[List[Optional[str]]] = None
        self.process: Optional[List[Optional[str]]] = None
        self.energy: Optional[List[Optional[int]]] = None
        self.luminosity: Optional[List[Optional[float]]] = None

        self.to_W: Union[bool, List[Optional[bool]]] = False
        self.to_Z: Union[bool, List[Optional[bool]]] = False
        self.to_H: Union[bool, List[Optional[bool]]] = False

        self.cs_keys: Dict[str, List[str]] = {}
        self.TB_XT_keys: Dict[str, List[str]] = {}
        self.TB_YB_keys: Dict[str, List[str]] = {}

        self.VLB: bool = False
        self.VLX: bool = False
        self.VLY: bool = False

        self._active_descriptors: List[ExperimentalLimitDescriptor] = []
        self._descriptor_by_key: Dict[str, ExperimentalLimitDescriptor] = {}
        self._curve_by_key: Dict[str, LimitCurveData] = {}
        self._missing_descriptors: List[MissingExperimentalLimitRecord] = []

        self._sync_species_flags()
        self._initialize_storage_for_active_model()

        diagnostics = self.registry_diagnostics()
        if strict_registry and diagnostics.has_issues:
            raise RegistryConsistencyError(
                "Embedded experimental registry failed strict validation:\n"
                + json.dumps(diagnostics.to_mapping(), indent=2)
            )

    # ------------------------------------------------------------------
    # Model classification
    # ------------------------------------------------------------------
    def _sync_species_flags(self) -> None:
        self.VLB = isinstance(self.model, (BottomPartnerSinglet, BottomPartnerDoublet, BottomPartnerPure))
        self.VLX = isinstance(self.model, XPartnerDoublet)
        self.VLY = isinstance(self.model, (YPartnerDoublet, YPartnerTriplet))

    @property
    def species(self) -> str:
        if self.VLB:
            return VLQSpecies.BOTTOM_LIKE.value
        if self.VLX:
            return VLQSpecies.X_LIKE.value
        if self.VLY:
            return VLQSpecies.Y_LIKE.value
        return VLQSpecies.TOP_LIKE.value

    @property
    def scenario(self) -> str:
        return str(self.model.model())

    @property
    def active_mass_container(self) -> Optional[List[Optional[FloatArray]]]:
        return {
            VLQSpecies.TOP_LIKE.value: self.MT,
            VLQSpecies.BOTTOM_LIKE.value: self.MB,
            VLQSpecies.X_LIKE.value: self.MX,
            VLQSpecies.Y_LIKE.value: self.MY,
        }.get(self.species)

    # ------------------------------------------------------------------
    # Path and storage setup
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_data_directory(data_directory: Optional[Union[str, Path]]) -> Path:
        if data_directory is not None:
            return Path(data_directory).expanduser().resolve()

        module_dir = Path(__file__).resolve().parent
        project_root = module_dir.parent.parent
        cwd = Path.cwd().resolve()

        candidates = (
            project_root / "data",
            cwd / "data",
            module_dir / "data",
            module_dir / "limits",
            module_dir / "tables",
            module_dir.parent / "data",
            module_dir.parent / "limits",
            module_dir.parent / "tables",
            project_root,
            cwd,
            module_dir,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return project_root / "data"

    @staticmethod
    def _build_search_roots(data_directory: Path) -> Tuple[Path, ...]:
        module_dir = Path(__file__).resolve().parent
        project_root = module_dir.parent.parent
        cwd = Path.cwd().resolve()
        return (
            data_directory,
            project_root,
            cwd,
            module_dir,
            module_dir.parent,
            data_directory / "ATLAS",
            data_directory / "CMS",
            data_directory / "atlas",
            data_directory / "cms",
            data_directory / "exp_tables",
        )

    def _initialize_storage_for_active_model(self) -> None:
        count = self.get_expected_dataset_count(self.species)
        self._allocate_runtime_arrays(count)

    def _allocate_runtime_arrays(self, count: int) -> None:
        self.file_name = [None] * count
        self.key = [None] * count
        self.label = [None] * count
        self.expt = [None] * count
        self.process = [None] * count
        self.energy = [None] * count
        self.luminosity = [None] * count
        self.exp = [None] * count
        self.obs = [None] * count

        self.MT = None
        self.MB = None
        self.MX = None
        self.MY = None
        self.which_doublet = None
        self.to_W = False
        self.to_Z = False
        self.to_H = False

        if isinstance(self.model, (TopPartnerPure, TopPartnerSinglet)):
            self.MT = [None] * count
        elif isinstance(self.model, TopPartnerDoublet):
            self.MT = [None] * count
            self.which_doublet = [None] * count
        elif isinstance(self.model, BottomPartnerPure):
            self.MB = [None] * count
            self.to_W = [None] * count
            self.to_Z = [None] * count
            self.to_H = [None] * count
        elif isinstance(self.model, BottomPartnerSinglet):
            self.MB = [None] * count
        elif isinstance(self.model, BottomPartnerDoublet):
            self.MB = [None] * count
            self.which_doublet = [None] * count
        elif isinstance(self.model, XPartnerDoublet):
            self.MX = [None] * count
        elif isinstance(self.model, (YPartnerDoublet, YPartnerTriplet)):
            self.MY = [None] * count

    # ------------------------------------------------------------------
    # Registry access
    # ------------------------------------------------------------------
    @classmethod
    @lru_cache(maxsize=1)
    def _registry(cls) -> Tuple[ExperimentalLimitDescriptor, ...]:
        return tuple(ExperimentalLimitDescriptor(*row) for row in _RAW_LIMIT_REGISTRY)

    def select_descriptors(
        self,
        *,
        species: Optional[str] = None,
        scenario: Optional[str] = None,
        experiment: Optional[str] = None,
        production_mode: Optional[Union[str, ProductionMode]] = None,
    ) -> List[ExperimentalLimitDescriptor]:
        selected: List[ExperimentalLimitDescriptor] = []
        production_value = production_mode.value if isinstance(production_mode, ProductionMode) else production_mode

        for descriptor in self._registry():
            if species is not None and descriptor.species != species:
                continue
            if scenario is not None and descriptor.scenario != scenario:
                continue
            if experiment is not None and descriptor.experiment != experiment:
                continue
            if production_value is not None and descriptor.production_mode.value != production_value:
                continue
            selected.append(descriptor)

        selected.sort(key=lambda item: item.index)
        return selected

    def iter_descriptors(self) -> Iterator[ExperimentalLimitDescriptor]:
        return iter(self.select_descriptors(species=self.species, scenario=self.scenario))

    def get_expected_dataset_count(self, vlq: Optional[str] = None) -> int:
        species = vlq or self.species
        scenario = self.scenario
        return EXPECTED_DATASET_COUNTS.get((species, scenario), len(self.select_descriptors(species=species, scenario=scenario)))

    def get_registry_descriptor_count(self, vlq: Optional[str] = None) -> int:
        species = vlq or self.species
        return len(self.select_descriptors(species=species, scenario=self.scenario))

    def get_descriptor_by_key(self, dataset_key: str) -> ExperimentalLimitDescriptor:
        if dataset_key in self._descriptor_by_key:
            return self._descriptor_by_key[dataset_key]
        for descriptor in self._registry():
            if descriptor.dataset_key == dataset_key:
                return descriptor
        raise KeyError(dataset_key)

    def registry_diagnostics(self) -> RegistryDiagnostics:
        seen_keys: Dict[str, int] = {}
        seen_files: Dict[str, int] = {}
        seen_signatures: Dict[str, int] = {}

        duplicate_dataset_keys: List[str] = []
        duplicate_filenames: List[str] = []
        missing_doublet_family: List[str] = []
        non_monotonic_indices: List[str] = []
        duplicate_channel_signatures: List[str] = []

        previous_index: Optional[int] = None
        descriptors = self.select_descriptors(species=self.species, scenario=self.scenario)

        for descriptor in descriptors:
            if descriptor.dataset_key in seen_keys:
                duplicate_dataset_keys.append(descriptor.dataset_key)
            else:
                seen_keys[descriptor.dataset_key] = descriptor.index

            if descriptor.file_name in seen_files:
                duplicate_filenames.append(descriptor.file_name)
            else:
                seen_files[descriptor.file_name] = descriptor.index

            if descriptor.channel_signature in seen_signatures:
                duplicate_channel_signatures.append(descriptor.channel_signature)
            else:
                seen_signatures[descriptor.channel_signature] = descriptor.index

            if self.scenario == "Doublet" and self.species in {"T", "B"} and not descriptor.doublet_family:
                missing_doublet_family.append(descriptor.dataset_key)

            if previous_index is not None and descriptor.index <= previous_index:
                non_monotonic_indices.append(
                    f"{descriptor.dataset_key}: index {descriptor.index} follows {previous_index}"
                )
            previous_index = descriptor.index

        return RegistryDiagnostics(
            duplicate_dataset_keys=tuple(duplicate_dataset_keys),
            duplicate_filenames=tuple(duplicate_filenames),
            missing_doublet_family=tuple(missing_doublet_family),
            non_monotonic_indices=tuple(non_monotonic_indices),
            duplicate_channel_signatures=tuple(duplicate_channel_signatures),
        )

    # ------------------------------------------------------------------
    # JSON loading and state materialisation
    # ------------------------------------------------------------------
    def load_experimental_limits(self) -> None:
        self._sync_species_flags()
        descriptors = self.select_descriptors(species=self.species, scenario=self.scenario)

        loaded_descriptors: List[ExperimentalLimitDescriptor] = []
        loaded_curves: List[LimitCurveData] = []
        missing_files: List[str] = []
        self._missing_descriptors = []

        for descriptor in descriptors:
            try:
                curve = self.reader.read(descriptor.file_name)
            except FileNotFoundError:
                missing_files.append(descriptor.file_name)
                self._missing_descriptors.append(
                    MissingExperimentalLimitRecord(
                        index=descriptor.index,
                        species=descriptor.species,
                        scenario=descriptor.scenario,
                        dataset_key=descriptor.dataset_key,
                        citation=descriptor.citation,
                        experiment=descriptor.experiment,
                        file_name=descriptor.file_name,
                        process=descriptor.process,
                        energy_tev=descriptor.energy_tev,
                        luminosity_fbinv=descriptor.luminosity_fbinv,
                        doublet_family=descriptor.doublet_family,
                    )
                )
                continue
            loaded_descriptors.append(descriptor)
            loaded_curves.append(curve)

        self._allocate_runtime_arrays(len(loaded_descriptors))
        self._active_descriptors = loaded_descriptors
        self._descriptor_by_key = {}
        self._curve_by_key = {}

        mass_container = self.active_mass_container
        if mass_container is None:
            raise BoundsCatalogError("No mass container is allocated for the active model.")

        for position, (descriptor, curve) in enumerate(zip(loaded_descriptors, loaded_curves)):
            self._descriptor_by_key[descriptor.dataset_key] = descriptor
            self._curve_by_key[descriptor.dataset_key] = curve

            self.key[position] = descriptor.dataset_key
            self.label[position] = descriptor.citation
            self.expt[position] = descriptor.experiment
            self.file_name[position] = descriptor.json_file_name
            self.process[position] = descriptor.process
            self.energy[position] = descriptor.energy_tev
            self.luminosity[position] = descriptor.luminosity_fbinv
            if self.which_doublet is not None:
                self.which_doublet[position] = descriptor.doublet_family

            mass_container[position] = curve.masses
            self.exp[position] = curve.expected
            self.obs[position] = curve.observed
            self._populate_decay_flags(position, descriptor)

        if missing_files:
            preview = ", ".join(missing_files[:5])
            suffix = "" if len(missing_files) <= 5 else f" ... (+{len(missing_files) - 5} more)"
            warnings.warn(
                (
                    f"Registry/data mismatch for {self.species}-{self.scenario}: loaded {len(loaded_descriptors)} of "
                    f"{len(descriptors)} registry entries. Missing files: {preview}{suffix}"
                ),
                RuntimeWarning,
            )

        self.build_production_mode_index()
        self.build_doublet_family_index()

    def _populate_decay_flags(self, position: int, descriptor: ExperimentalLimitDescriptor) -> None:
        if not isinstance(self.to_W, list) or not isinstance(self.to_Z, list) or not isinstance(self.to_H, list):
            return
        process = descriptor.process
        self.to_W[position] = "W" in process
        self.to_Z[position] = "Z" in process
        self.to_H[position] = "H" in process

    def get_curve_by_key(self, dataset_key: str) -> LimitCurveData:
        if dataset_key in self._curve_by_key:
            return self._curve_by_key[dataset_key]
        descriptor = self.get_descriptor_by_key(dataset_key)
        curve = self.reader.read(descriptor.file_name)
        self._curve_by_key[dataset_key] = curve
        return curve

    def missing_descriptors(self) -> List[MissingExperimentalLimitRecord]:
        return list(self._missing_descriptors)

    def missing_descriptor_rows(self) -> List[Dict[str, Any]]:
        return [record.to_mapping() for record in self._missing_descriptors]

    def registry_reference_report(self) -> Dict[str, Any]:
        registry_count = self.get_registry_descriptor_count()
        expected_count = self.get_expected_dataset_count()
        loaded_count = len(self._active_descriptors)
        missing_count = len(self._missing_descriptors)
        return {
            "species": self.species,
            "scenario": self.scenario,
            "reference_expected_count": expected_count,
            "registry_count": registry_count,
            "loaded_count": loaded_count,
            "missing_count": missing_count,
            "counts_match_reference_layout": registry_count == expected_count,
        }

    def export_missing_descriptor_manifest(self, output_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        if not self._missing_descriptors:
            return None
        if output_path is None:
            output_path = Path(__file__).resolve().parents[2] / "reports" / "manifests" / "missing_experimental_limits.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "species": self.species,
            "scenario": self.scenario,
            "registry_reference_report": self.registry_reference_report(),
            "missing": self.missing_descriptor_rows(),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path

    # ------------------------------------------------------------------
    # Index builders and summaries
    # ------------------------------------------------------------------
    def build_production_mode_index(self) -> None:
        pair_prod: List[str] = []
        single_prod: List[str] = []

        for descriptor in self._active_descriptors:
            if descriptor.production_mode is ProductionMode.PAIR:
                pair_prod.append(descriptor.dataset_key)
            else:
                single_prod.append(descriptor.dataset_key)

        self.cs_keys = {
            ProductionMode.PAIR.value: pair_prod,
            ProductionMode.SINGLE.value: single_prod,
        }

    @staticmethod
    def _family_tokens(tag: Optional[str]) -> Tuple[str, ...]:
        if not tag:
            return tuple()
        return tuple(part.strip() for part in tag.split("or"))

    def build_doublet_family_index(self) -> None:
        self.TB_XT_keys = {}
        self.TB_YB_keys = {}

        if self.which_doublet is None:
            return

        tb_keys: List[str] = []
        xt_keys: List[str] = []
        by_keys: List[str] = []

        for key, family in zip(self.key or [], self.which_doublet):
            if key is None or not family:
                continue
            tokens = self._family_tokens(family)
            if any(token.endswith("TB") or token == "TB" for token in tokens):
                tb_keys.append(key)
            if any(token.startswith("XT") or token == "XT" for token in tokens):
                xt_keys.append(key)
            if any(token.startswith("BY") or token == "BY" for token in tokens):
                by_keys.append(key)

        if self.species == "T":
            self.TB_XT_keys = {"(T,B)": tb_keys, "(X,T)": xt_keys}
        elif self.species == "B":
            self.TB_YB_keys = {"(T,B)": tb_keys, "(B,Y)": by_keys}
            self.TB_XT_keys = dict(self.TB_YB_keys)

    def manifest_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "channel": index,
                "dataset_key": descriptor.dataset_key,
                "citation": descriptor.citation,
                "experiment": descriptor.experiment,
                "process": descriptor.process,
                "energy_tev": descriptor.energy_tev,
                "luminosity_fbinv": descriptor.luminosity_fbinv,
                "file_name": descriptor.json_file_name,
                "doublet_family": descriptor.doublet_family,
                "production_mode": descriptor.production_mode.value,
            }
            for index, descriptor in enumerate(self._active_descriptors)
        ]

    def summary(self) -> Dict[str, Any]:
        diagnostics = self.registry_diagnostics()
        return {
            "species": self.species,
            "scenario": self.scenario,
            "dataset_count": len(self._active_descriptors) if self._active_descriptors else self.get_expected_dataset_count(),
            "production_modes": self.cs_keys,
            "doublet_groups_tb_xt": self.TB_XT_keys,
            "doublet_groups_tb_yb": self.TB_YB_keys,
            "diagnostics": diagnostics.to_mapping(),
            "registry_reference_report": self.registry_reference_report(),
            "missing_descriptor_count": len(self._missing_descriptors),
        }

    def export_channel_manifest(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        if not self._active_descriptors:
            raise BoundsCatalogError("No active descriptors are loaded. Call load_experimental_limits() first.")

        if output_path is None:
            output_path = Path(__file__).resolve().parent.parent / "keys" / "coupling_info.dat"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as stream:
            stream.write("************* File for each cross section limit information*****************\n")
            stream.write("Generated by ExperimentalBoundsRepository\n")
            stream.write(f"Active species: {self.species}\n")
            stream.write(f"Active scenario: {self.scenario}\n")
            for channel_index, descriptor in enumerate(self._active_descriptors):
                stream.write("***********************************************************************\n")
                stream.write(f"channel {channel_index}:\n")
                stream.write(
                    f"{descriptor.process}\t\t {descriptor.citation} ({descriptor.experiment})"
                    f"\t sqrt(s) = {descriptor.energy_tev} TeV"
                    f"\t luminosity = {descriptor.luminosity_fbinv} fb-1\n"
                )
        return output_path

    # ------------------------------------------------------------------
    # Runtime interface
    # ------------------------------------------------------------------
    def bounds_properties_arrays(self, number_of_atlas_cms_tables: int) -> None:
        self._allocate_runtime_arrays(number_of_atlas_cms_tables)

    def get_number_of_cms_atlas_tables(self, vlq: str = "T") -> int:
        return self.get_expected_dataset_count(vlq)

    def initialize_bounds_arrays(self) -> None:
        self._initialize_storage_for_active_model()

    def initialize_tables_cms_and_atlas(self) -> None:
        self.load_experimental_limits()

    def all_processes(self) -> Path:
        return self.export_channel_manifest()

    def cs_dict(self) -> None:
        self.build_production_mode_index()

    def tb_xt_dict(self) -> None:
        self.build_doublet_family_index()


Tables = ExperimentalBoundsRepository


_RAW_LIMIT_REGISTRY: Tuple[Tuple[int, str, str, str, str, str, str, str, int, float, Optional[str]], ...] = (
    (0, 'B', 'Singlet', '01762fb', 'arXiv:1806.01762', 'ATLAS', '1806.01762_ATLAS_fig4b_pp_BB_Wt_singlet.json', 'pp --> BB --> Wt --> 1l', 13, 36.1, None),
    (1, 'B', 'Singlet', '10555f13b', 'arXiv:1806.10555', 'ATLAS', '1806.10555_ATLAS_fig13b_pp_BB_Zb_singlet.json', 'pp --> BB --> Zb --> l+l-+>=3l', 13, 36.1, None),
    (2, 'B', 'Singlet', '15413f7b', 'arXiv:2210.15413', 'ATLAS', '2210.15413_ATLAS_fig7b_pp_BB_Zb_singlet.json', 'pp --> BB --> Zb --> l+l-+3l', 13, 139.0, None),
    (3, 'B', 'Singlet', '05263f7d', 'arXiv:2212.05263', 'ATLAS', '2212.05263_ATLAS_fig7d_pp_BB_Wt_singlet.json', 'pp --> BB --> Wt --> 1l', 13, 139.0, None),
    (4, 'B', 'Singlet', '01486Fig6a', 'arXiv:1802.01486', 'CMS', '1802.01486_Fig6a_CMS_upper_left_pp_B_bH_Gamma_M1_singlet_or_doublet.json', 'pp --> Bbq --> bHbq --> 0l', 13, 35.9, None),
    (5, 'B', 'Singlet', '01486Fig6b', 'arXiv:1802.01486', 'CMS', '1802.01486_Fig6b_CMS_upper_right_pp_B_bH_Gamma_M10_singlet_or_doublet.json', 'pp --> Bbq --> bHbq --> 0l', 13, 35.9, None),
    (6, 'B', 'Singlet', '01486Fig6c', 'arXiv:1802.01486', 'CMS', '1802.01486_Fig6c_CMS_lower_left_pp_B_bH_Gamma_M20_singlet_or_doublet.json', 'pp --> Bbq --> bHbq --> 0l', 13, 35.9, None),
    (7, 'B', 'Singlet', '01486Fig6d', 'arXiv:1802.01486', 'CMS', '1802.01486_Fig6d_CMS_lower_right_pp_B_bH_Gamma_M30_singlet_or_doublet.json', 'pp --> Bbq --> bHbq --> 0l', 13, 35.9, None),
    (8, 'B', 'Singlet', '04758f8ll', 'arXiv:1805.04758', 'CMS', '1805.04758_CMS_Fig8_lower_left_singlet.json', 'pp --> BB --> 1l+l+l++3l', 13, 35.9, None),
    (9, 'B', 'Singlet', '07327f9c', 'arXiv:2209.07327', 'CMS', '2209.07327_CMS_f9c_pp_BB_singlet.json', 'pp --> BB --> 1l+l+l++3l', 13, 138.0, None),
    (10, 'B', 'Singlet', '5500f12a', 'arXiv:1409.5500', 'ATLAS', '1409.5500_ATLAS_Fig12a_pp_BB_singlet.json', 'pp --> BB --> l+l-+3l', 8, 20.3, None),
    (11, 'B', 'Singlet', '05425f8', 'arXiv:1503.05425', 'ATLAS', '1503.05425_ATLAS_Fig8_pp_BB_Wt_Zb_HB_singlet.json', 'pp --> BB --> 1l', 8, 20.3, None),
    (12, 'B', 'Singlet', '04605f7a', 'arXiv:1504.04605', 'ATLAS', '1504.04605_ATLAS_Fig7a_pp_BB_singlet.json', 'pp --> BB --> l+l+', 8, 20.3, None),
    (13, 'B', 'Singlet', '04306f22b', 'arXiv:1505.04306', 'ATLAS', '1505.04306_ATLAS_Fig22b_pp_BB_Hb_singlet.json', 'pp --> BB --> 1l', 8, 20.3, None),
    (14, 'B', 'Singlet', '02343f3a', 'arXiv:1808.02343', 'ATLAS', '1808.02343_ATLAS_Fig3a_pp_BB_singlet.json', 'pp --> BB --> 1l+l+l++3l', 13, 36.1, None),
    (15, 'B', 'Singlet', '02595f8a', 'arXiv:2308.02595', 'ATLAS', '2308.02595_ATLAS_Fig8a_pp_B_bH_k03_singlet.json', 'pp --> Bb(t)q --> bHb(t)q --> 0l', 13, 139.0, None),
    (16, 'B', 'Singlet', '02595f8c', 'arXiv:2308.02595', 'ATLAS', '2308.02595_ATLAS_Fig8c_pp_B_bH_k04_singlet.json', 'pp --> Bb(t)q --> bHb(t)q --> 0l', 13, 139.0, None),
    (17, 'B', 'Singlet', '02595f8e', 'arXiv:2308.02595', 'ATLAS', '2308.02595_ATLAS_Fig8e_pp_B_bH_k05_singlet.json', 'pp --> Bb(t)q --> bHb(t)q --> 0l', 13, 139.0, None),
    (18, 'B', 'Singlet', '07409f5l', 'arXiv:1701.07409', 'CMS', '1701.07409_CMS_Fig5_left_pp_Bt_bZ_cWt05_singlet.json', 'pp --> Btq --> bZtq --> 0l', 13, 2.3, None),
    (19, 'B', 'Singlet', '07409f5r', 'arXiv:1701.07409', 'CMS', '1701.07409_CMS_Fig5_right_pp_Bb_bZ_cbZ05_singlet.json', 'pp --> Bbq --> bZbq --> 0l', 13, 2.3, None),
    (20, 'B', 'Singlet', '10216f4l', 'arXiv:2111.10216', 'CMS', '2111.10216_CMS_Fig4_left_pp_B_Wt_singlet_doublet.json', 'pp --> Bbq --> tWbq --> 1l ', 13, 138.0, None),
    (21, 'B', 'Singlet', '10216f4r', 'arXiv:2111.10216', 'CMS', '2111.10216_CMS_Fig4_right_pp_B_Wt_singlet_doublet.json', 'pp --> Btq --> tWtq --> 1l ', 13, 138.0, None),
    (22, 'B', 'Singlet', '17605f39ll', 'arXiv:2405.17605', 'CMS', '2405.17605_CMS_Fig39_lower_left_pp_BB_singlet.json', 'pp --> BB --> 0l', 13, 138.0, None),
    (0, 'B', 'Doublet', '10555f13d', 'arXiv:1806.10555', 'ATLAS', '1806.10555_ATLAS_fig13d_pp_BB_BY_doublet.json', 'pp --> BB --> l+l-+>=3l', 13, 36.1, 'BY'),
    (1, 'B', 'Doublet', '01771f13c', 'arXiv:1808.01771', 'ATLAS', '1808.01771_ATLAS_fig13c_pp_BB_BY_doublet.json', 'pp --> BB --> 0l', 13, 36.1, 'BY'),
    (2, 'B', 'Doublet', '05263f7f', 'arXiv:2212.05263', 'ATLAS', '2212.05263_ATLAS_fig7f_pp_QQ_mass_degenerate_BYorTB_doublet.json', 'pp --> BB --> 1l', 13, 139.0, 'TB'),
    (3, 'B', 'Doublet', '01486Fig6a', 'arXiv:1802.01486', 'CMS', '1802.01486_Fig6a_CMS_upper_left_pp_B_bH_Gamma_M1_singlet_or_doublet.json', 'pp --> Bbq --> bHbq --> 0l', 13, 35.9, 'BYorTB'),
    (4, 'B', 'Doublet', '01486Fig6b', 'arXiv:1802.01486', 'CMS', '1802.01486_Fig6b_CMS_upper_right_pp_B_bH_Gamma_M10_singlet_or_doublet.json', 'pp --> Bbq --> bHbq --> 0l', 13, 35.9, 'BYorTB'),
    (5, 'B', 'Doublet', '01486Fig6c', 'arXiv:1802.01486', 'CMS', '1802.01486_Fig6c_CMS_lower_left_pp_B_bH_Gamma_M20_singlet_or_doublet.json', 'pp --> Bbq --> bHbq --> 0l', 13, 35.9, 'BYorTB'),
    (6, 'B', 'Doublet', '01486Fig6d', 'arXiv:1802.01486', 'CMS', '1802.01486_Fig6d_CMS_lower_right_pp_B_bH_Gamma_M30_singlet_or_doublet.json', 'pp --> Bbq --> bHbq --> 0l', 13, 35.9, 'BYorTB'),
    (7, 'B', 'Doublet', '04758Fig8lr', 'arXiv:1805.04758', 'CMS', '1805.04758_CMS_Fig8_lower_right_pp_BB_doublet.json', 'pp --> BB --> 1l+l+l++3l', 13, 35.9, 'BYorTB'),
    (8, 'B', 'Doublet', '09768Fig7ur', 'arXiv:1812.09768', 'CMS', '1812.09768_CMS_Fig7_upper_right_pp_BB_doublet.json', 'pp --> BB --> l+l-', 13, 35.9, 'BYorTB'),
    (9, 'B', 'Doublet', '09835Fig11l', 'arXiv:2008.09835', 'CMS', '2008.09835_CMS_Fig11_lower_pp_BB_doublet.json', 'pp --> BB --> 0l', 13, 137.0, 'BYorTB'),
    (10, 'B', 'Doublet', '07327f9d', 'arXiv:2209.07327', 'CMS', '2209.07327_CMS_f9d_pp_BB_doublet.json', 'pp --> BB --> 1l+l+l++3l', 13, 137.0, 'BYorTB'),
    (11, 'B', 'Doublet', '02343f3b', 'arXiv:1808.02343', 'ATLAS', '1808.02343_ATLAS_Fig3b_pp_BB_TB_doublet.json', 'pp --> BB --> 1l+l+l++3l', 13, 36.1, 'TB'),
    (12, 'B', 'Doublet', '02343f3c', 'arXiv:1808.02343', 'ATLAS', '1808.02343_ATLAS_Fig3c_pp_BB_BY_doublet.json', 'pp --> BB --> 1l+l+l++3l', 13, 36.1, 'BY'),
    (13, 'B', 'Doublet', '02595f8b', 'arXiv:2308.02595', 'ATLAS', '2308.02595_ATLAS_Fig8b_pp_B_bH_k03_BY_doublet.json', 'pp --> Bb(t)q --> bHb(t)q --> 0l', 13, 139.0, 'BY'),
    (14, 'B', 'Doublet', '02595f8d', 'arXiv:2308.02595', 'ATLAS', '2308.02595_ATLAS_Fig8d_pp_B_bH_k04_BY_doublet.json', 'pp --> Bb(t)q --> bHb(t)q --> 0l', 13, 139.0, 'BY'),
    (15, 'B', 'Doublet', '02595f8f', 'arXiv:2308.02595', 'ATLAS', '2308.02595_ATLAS_Fig8f_pp_B_bH_k05_BY_doublet.json', 'pp --> Bb(t)q --> bHb(t)q --> 0l', 13, 139.0, 'BY'),
    (16, 'B', 'Doublet', '03408f13r', 'arXiv:1706.03408', 'CMS', '1706.03408_CMS_Fig13_right_pp_BB_doublet.json', 'pp --> BB --> 1l', 13, 2.6, 'BYorTB'),
    (17, 'B', 'Doublet', '13808f18ll', 'arXiv:2402.13808', 'CMS', '2402.13808_CMS_Fig18_lower_left_pp_BB_TB_doublet.json', 'pp --> BB --> 0l+l+l-', 13, 138.0, 'TB'),
    (18, 'B', 'Doublet', '10216f4ld', 'arXiv:2111.10216', 'CMS', '2111.10216_CMS_Fig4_left_pp_B_Wt_singlet_doublet.json', 'pp --> Bbq --> tWbq --> 1l ', 13, 138.0, 'BYorTB'),
    (19, 'B', 'Doublet', '17605f39lr', 'arXiv:2405.17605', 'CMS', '2405.17605_CMS_Fig39_lower_right_pp_BB_doublet.json', 'pp --> BB --> 0l ', 13, 138.0, 'BYorTB'),
    (20, 'B', 'Doublet', '15413f7d', 'arXiv:2210.15413', 'ATLAS', '2210.15413_ATLAS_fig7d_pp_BB_BY_doublet.json', 'pp --> BB --> l+l-+3l', 13, 139.0, 'BY'),
    (0, 'B', 'Pure', '01762f4a', 'arXiv:1806.01762', 'ATLAS', '1806.01762_ATLAS_fig4a_pp_BB_Wt.json', 'pp --> BB --> WtWt --> 1l', 13, 36.1, None),
    (1, 'B', 'Pure', '15413fig7f', 'arXiv:2210.15413', 'ATLAS', '2210.15413_ATLAS_fig7f_pp_BB_Zb.json', 'pp --> BB --> ZbZb --> >= l+l-', 13, 139.0, None),
    (2, 'B', 'Pure', '09768fig7ul', 'arXiv:1812.09768', 'CMS', '1812.09768_CMS_Fig7_upper_left_pp_BB_bZ.json', 'pp --> BB --> ZbZb --> l+l-', 13, 35.9, None),
    (3, 'B', 'Pure', '09835fig11ul', 'arXiv:2008.09835', 'CMS', '2008.09835_CMS_fig11_upper_left_pp_B_bH.json', 'pp --> BB --> HbHb --> 0l', 13, 137.0, None),
    (4, 'B', 'Pure', '09835fig11ur', 'arXiv:2008.09835', 'CMS', '2008.09835_CMS_fig11_upper_right_pp_B_bZ.json', 'pp --> BB --> ZbZb --> 0l', 13, 137.0, None),
    (5, 'B', 'Pure', '1265f5', 'arXiv:1204.1265', 'ATLAS', '1204.1265_ATLAS_Fig5_pp_BB_Zb.json', 'pp --> BB --> ZbZb --> l+l-', 13, 137.0, None),
    (6, 'B', 'Pure', '04605f6a', 'arXiv:1504.04605', 'ATLAS', '1504.04605_ATLAS_Fig6a_pp_BB_Wt.json', 'pp --> BB --> WtWt --> l+l+', 8, 20.3, None),
    (7, 'B', 'Pure', '04306f22a', 'arXiv:1505.04306', 'ATLAS', '1505.04306_ATLAS_Fig22a_pp_BB_Hb.json', 'pp --> BB --> HbHb --> 1l', 8, 20.3, None),
    (8, 'B', 'Pure', '13808f18ul', 'arXiv:2402.13808', 'CMS', '2402.13808_CMS_Fig18_upper_left_pp_BB_bH.json', 'pp --> BB --> HbHb --> 0l+l+l-', 13, 138.0, None),
    (9, 'B', 'Pure', '13808f18ur', 'arXiv:2402.13808', 'CMS', '2402.13808_CMS_Fig18_upper_right_pp_BB_bZ.json', 'pp --> BB --> ZbZb --> 0l+l+l-', 13, 138.0, None),
    (10, 'B', 'Pure', '07129f13l', 'arXiv:1507.07129', 'CMS', '1507.07129_Fig13_left_pp_BB_Wt.json', 'pp --> BB --> WtWt --> 1l+2l+>=2l', 8, 19.7, None),
    (11, 'B', 'Pure', '07129f13m', 'arXiv:1507.07129', 'CMS', '1507.07129_Fig13_middle_pp_BB_bZ.json', 'pp --> BB --> ZbZb --> 1l+2l+>=2l', 8, 19.7, None),
    (12, 'B', 'Pure', '07129f13r', 'arXiv:1507.07129', 'CMS', '1507.07129_Fig13_right_pp_BB_bH.json', 'pp --> BB --> HbHb --> 1l+2l+>=2l', 8, 19.7, None),
    (13, 'B', 'Pure', '17605f38l', 'arXiv:2405.17605', 'CMS', '2405.17605_CMS_Fig38_lower_pp_BB_tW.json', 'pp --> BB --> WtWt --> 0l+>=1l+2l', 13, 19.7, None),
    (14, 'B', 'Pure', '17605f38l', 'arXiv:2405.17605', 'CMS', '2405.17605_CMS_Fig38_upper_left_pp_BB_bZ.json', 'pp --> BB --> ZbZb --> 0l+2l', 13, 138.0, None),
    (15, 'B', 'Pure', '17605f38l', 'arXiv:2405.17605', 'CMS', '2405.17605_CMS_Fig38_upper_right_pp_BB_bH.json', 'pp --> BB --> HbHb --> 0l+2l', 13, 138.0, None),
    (16, 'B', 'Pure', '05263f7f', 'arXiv:2212.05263', 'ATLAS', '2212.05263_ATLAS_fig7b_pp_BB_BYorTB_doublet.json', 'pp --> BB --> WtWt --> 1l', 13, 139.0, None),
    (0, 'X', 'Doublet', '03188f7a', 'arXiv:1810.03188', 'CMS', '1810.03188_CMS_Fig7a_pp_XX_tW_left_handed.json', 'pp --> XX --> Wt --> 1l+l+l+', 13, 35.9, None),
    (1, 'X', 'Doublet', '03188f7b', 'arXiv:1810.03188', 'CMS', '1810.03188_CMS_Fig7b_pp_XX_tW_right_handed.json', 'pp --> XX --> Wt --> 1l+l+l+', 13, 35.9, None),
    (2, 'X', 'Doublet', '03188f7c', 'arXiv:1810.03188', 'CMS', '1810.03188_CMS_Fig7c_pp_XX_tW_left_handed.json', 'pp --> XX --> Wt --> 1l+l+l+', 13, 35.9, None),
    (3, 'X', 'Doublet', '03188f7d', 'arXiv:1810.03188', 'CMS', '1810.03188_CMS_Fig7d_pp_XX_tW_right_handed.json', 'pp --> XX --> Wt --> 1l+l+l+', 13, 35.9, None),
    (4, 'X', 'Doublet', '03188f7d', 'arXiv:1810.03188', 'CMS', '1810.03188_CMS_Fig7d_pp_XX_tW_right_handed.json', 'pp --> XX --> Wt --> 1l+l+l+', 13, 35.9, None),
    (5, 'X', 'Doublet', '03188f8a', 'arXiv:1810.03188', 'CMS', '1810.03188_CMS_Fig8a_pp_XX_tW_left_handed_combination.json', 'pp --> XX --> Wt --> 1l+l+l+', 13, 35.9, None),
    (6, 'X', 'Doublet', '03188f8b', 'arXiv:1810.03188', 'CMS', '1810.03188_CMS_Fig8b_pp_XX_tW_right_handed_combination.json', 'pp --> XX --> Wt --> 1l+l+l+', 13, 35.9, None),
    (7, 'X', 'Doublet', '03347f4a', 'arXiv:1707.03347', 'ATLAS', '1707.03347_ATLAS_fig4a_pp_XX_Wb.json', 'pp --> XX --> Wb --> 1l', 13, 36.1, None),
    (8, 'X', 'Doublet', '03347f4a', 'arXiv:1806.01762', 'ATLAS', '1806.01762_ATLAS_Fig4a_pp_XX_Wt.json', 'pp --> XX --> Wt --> 1l', 13, 36.1, None),
    (9, 'X', 'Doublet', '11883f10a', 'arXiv:1807.11883', 'ATLAS', '1807.11883_ATLAS_Fig10a_pp_XX_Wt.json', 'pp --> XX --> l+l+', 13, 36.1, None),
    (10, 'X', 'Doublet', '05263f7b', 'arXiv:2212.05263', 'ATLAS', '2212.05263_ATLAS_fig7b_pp_XX_Wt_doublet.json', 'pp --> XX --> Wt --> 1l', 13, 139.0, None),
    (11, 'X', 'Doublet', '05263f7f', 'arXiv:2212.05263', 'ATLAS', '2212.05263_ATLAS_fig7f_pp_XX_Wt_doublet.json', 'pp --> XX --> Wt --> 1l', 13, 139.0, None),
    (12, 'X', 'Doublet', '04605f10a', 'arXiv:1504.04605', 'ATLAS', '1504.04605_ATLAS_fig10a_pp_XX_Wt.json', 'pp --> XX --> Wt --> l+l+', 8, 20.3, None),
    (13, 'X', 'Doublet', '08597f8ur', 'arXiv:1809.08597', 'CMS', '1809.08597_Fig8_upper_right_pp_X_Wt_gamma_M1.json', 'pp --> Xtq --> Wt --> 1l', 13, 35.9, None),
    (0, 'Y', 'Doublet', '08328f5', 'arXiv:1701.08328', 'CMS', '1701.08328_CMS_Fig5_pp_Ybq_bWbq_c05.json', 'pp --> Ybq --> bW --> 1l', 13, 2.3, None),
    (1, 'Y', 'Doublet', '01539f4', 'arXiv:1710.01539', 'CMS', '1710.01539_CMS_Fig4_upper_pp_YY_Wb.json', 'pp --> YY --> Wb --> 1l', 13, 35.8, None),
    (2, 'Y', 'Doublet', '03347f4a', 'arXiv:1707.03347', 'ATLAS', '1707.03347_ATLAS_fig4a_pp_YY_Wb.json', 'pp --> YY --> Wb --> 1l ', 13, 36.1, None),
    (3, 'Y', 'Doublet', '17165f5a', 'arXiv:2401.17165', 'ATLAS', '2401.17165_ATLAS_Fig5a_pp_YY_bW.json', 'pp --> YY --> Wb --> 1l', 13, 140.0, None),
    (4, 'Y', 'Doublet', '05606f6', 'arXiv:1602.05606', 'ATLAS', '1602.05606_ATLAS_Fig6_pp_Ybj_Wb.json', 'pp --> Ybq --> bW --> 1l', 8, 20.3, None),
    (5, 'Y', 'Doublet', '07343f9', 'arXiv:1812.07343', 'ATLAS', '1812.07343_fig9_pp_Y_Wb.json', 'pp --> Ybq --> bW --> 1l', 13, 36.1, None),
    (6, 'Y', 'Doublet', '20273f5a', 'arXiv:2409.20273', 'ATLAS', '2409.20273_ATLAS_Fig5a_pp_Qbq_Wb_k05_Tsinglet_Ydoublet.json', 'pp --> Ybq --> bW --> 0l', 13, 139.0, None),
    (7, 'Y', 'Doublet', '20273f5b', 'arXiv:2409.20273', 'ATLAS', '2409.20273_ATLAS_Fig5b_pp_Qbq_Wb_k07_Tsinglet_YDoublet.json', 'pp --> Ybq --> bW --> 0l', 13, 139.0, None),
    (0, 'Y', 'Triplet', '08328f5', 'arXiv:1701.08328', 'CMS', '1701.08328_CMS_Fig5_pp_Ybq_bWbq_c05.json', 'pp --> Ybq --> bW --> 1l', 13, 2.3, None),
    (1, 'Y', 'Triplet', '01539f4', 'arXiv:1710.01539', 'CMS', '1710.01539_CMS_Fig4_upper_pp_YY_Wb.json', 'pp --> YY --> Wb --> 1l', 13, 35.8, None),
    (2, 'Y', 'Triplet', '03347f4a', 'arXiv:1707.03347', 'ATLAS', '1707.03347_ATLAS_fig4a_pp_YY_Wb.json', 'pp --> YY --> Wb --> 1l', 13, 36.1, None),
    (3, 'Y', 'Triplet', '17165f5a', 'arXiv:2401.17165', 'ATLAS', '2401.17165_ATLAS_Fig5a_pp_YY_bW.json', 'pp --> YY --> Wb --> 1l', 13, 140.0, None),
    (0, 'T', 'Singlet', '10751fig6b', 'arXiv:1705.10751', 'ATLAS', '1705.10751_ATLAS_fig6b_pp_TT_Zt_Singlet.json', 'pp --> TT --> 1l', 13, 36.1, None),
    (1, 'T', 'Singlet', '73270', 'arXiv:2209.07327', 'CMS', '2209.07327_CMS_f9a_pp_TTbar_Singlet.json', 'pp --> TT --> 1l+l+l++3l', 13, 138.0, None),
    (2, 'T', 'Singlet', '33470fig4b', 'arXiv:1707.03347', 'ATLAS', '1707.03347_ATLAS_fig4b_pp_TT_Singlet.json', 'pp --> TT --> 1l', 13, 36.1, None),
    (3, 'T', 'Singlet', '05263', 'arXiv:2212.05263', 'ATLAS', '2212.05263_ATLAS_fig7c_pp_TT_Zt_Singlet.json', 'pp --> TT --> 1l', 13, 139.0, None),
    (4, 'T', 'Singlet', '10555', 'arXiv:1806.10555', 'ATLAS', '1806.10555_ATLAS_Fig-13_a_pp_TT_Zt_singlet.json', 'pp --> TT --> l+l-,≥3l', 13, 36.1, None),
    (5, 'T', 'Singlet', '04758', 'arXiv:1805.04758', 'CMS', '1805.04758_CMS_Fig8-upper-row-left_pp_TT_bW_Zt_tH_Singlet.json', 'pp --> TT --> 1l+l+l++3l', 13, 35.9, None),
    (6, 'T', 'Singlet', '11883', 'arXiv:1807.11883', 'ATLAS', '1807.11883_ATLAS_pp_TT_Wb_Zt_Ht_Fig_8_b_Singlet.json', 'pp --> TT --> l+l+', 13, 36.1, None),
    (7, 'T', 'Singlet', '03408', 'arXiv:1706.03408', 'CMS', '1706.03408_CMS_Fig-12-left_pp_TT_bW_Zt_tH_Singlet.json', 'pp --> TT --> 1l', 13, 2.6, None),
    (8, 'T', 'Singlet', '5500fc', 'arXiv:1409.5500', 'ATLAS', '1409.5500_ATLAS_Fig-12-c_pp_TT_Zt_Singlet.json', 'pp --> TT --> l+l-+3l', 8, 20.3, None),
    (9, 'T', 'Singlet', '04306fb', 'arXiv:1505.04306', 'ATLAS', '1505.04306_ATLAS_Fig18-b_pp_TT_Ht+X-Wb+X_Singlet.json', 'pp --> TT --> 1l', 8, 20.3, None),
    (10, 'T', 'Singlet', '02227fa', 'arXiv:2201.02227', 'CMS', '2201.02227_CMS_f8a_pp_Tbq_tZ_gamma_mT_0.05_Singlet.json', 'pp --> Tbq --> tZbq --> 0l', 13, 137.0, None),
    (11, 'T', 'Singlet', '02227fb', 'arXiv:2201.02227', 'CMS', '2201.02227_CMS_f8b_pp_Tbq_tZ_gamma_mT_0.1_Singlet.json', 'pp --> Tbq --> tZbq --> 0l', 13, 137.0, None),
    (12, 'T', 'Singlet', '02227fc', 'arXiv:2201.02227', 'CMS', '2201.02227_CMS_fig8c_pp_Tbq_tZ_gamma_mT_0.2_Singlet.json', 'pp --> Tbq --> tZbq --> 0l', 13, 137.0, None),
    (13, 'T', 'Singlet', '02227fd', 'arXiv:2201.02227', 'CMS', '2201.02227_CMS_fig8d_pp_Tbq_tZ_gamma_mT_0.3_Singlet.json', 'pp --> Tbq --> tZbq --> 0l', 13, 137.0, None),
    (14, 'T', 'Singlet', '01062f5a', 'arXiv:1708.01062', 'CMS', '1708.01062_CMS_Fig5-left_pp_Tbq_tZbq_Singlet-LH.json', 'pp --> Tbq --> tZbq --> l+l-', 13, 35.9, None),
    (15, 'T', 'Singlet', '17605f35', 'arXiv:2405.17605', 'CMS', '2405.17605_CMS_fig35_pp_Tbq_singlet.json', 'pp --> Tbq --> 0l', 13, 138.0, None),
    (16, 'T', 'Singlet', '12802', 'arXiv:2302.12802', 'CMS', '2302.12802_CMS_f4_pp_Tbq_tH_Singlet.json', 'pp --> Tbq --> tHbq --> 0l+1l', 13, 138.0, None),
    (17, 'T', 'Singlet', '00999f10a', 'arXiv:1612.00999', 'CMS', '1612.00999_CMS_Fig10-left_pp_TT_tH.json', 'pp --> Tbq --> tHbq --> 0l', 13, 2.3, None),
    (18, 'T', 'Singlet', '04721f8a', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig8_upper-row-left_pp_Tbq_tHbq.json', 'pp --> Tbq --> tHbq --> 0l', 13, 35.9, None),
    (19, 'T', 'Singlet', '04721f8b', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig8_upper-row-right_pp_Tbq_tHbq.json', 'pp --> Tbq --> tHbq --> 0l', 13, 35.9, None),
    (20, 'T', 'Singlet', '04721f8c', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig8_middle-row-left_pp_Tbq_tZbq.json', 'pp --> Tbq --> tZbq --> 0l', 13, 35.9, None),
    (21, 'T', 'Singlet', '04721f8d', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig8_middle-row-right_pp_Tbq_tZbq.json', 'pp --> Tbq --> tZbq --> 0l', 13, 35.9, None),
    (22, 'T', 'Singlet', '04721f8e', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig8_lower-row-left_pp_Tbq_tH+tZ_bq.json', 'pp --> Tbq --> (tZ + tH)bq --> 0l', 13, 35.9, None),
    (23, 'T', 'Singlet', '04721f8f', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig8_lower-row-right_pp_Tbq_tH+tZ_bq.json', 'pp --> Tbq --> (tZ + tH)bq --> 0l', 13, 35.9, None),
    (24, 'T', 'Singlet', '04721f9a', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig9_upper-row-left_pp_Tbq_tHbq.json', 'pp --> Tbq --> tHbq --> 0l', 13, 35.9, None),
    (25, 'T', 'Singlet', '04721f9b', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig9_upper-row-right_pp_Tbq_tHbq.json', 'pp --> Tbq --> tHbq --> 0l', 13, 35.9, None),
    (26, 'T', 'Singlet', '04721f9c', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig9_middle-row-left_pp_Tbq_tZbq.json', 'pp --> Tbq --> tZbq --> 0l', 13, 35.9, None),
    (27, 'T', 'Singlet', '04721f9d', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig9_middle-row-right_pp_Tbq_tZbq.json', 'pp --> Tbq --> tZbq --> 0l', 13, 35.9, None),
    (28, 'T', 'Singlet', '04721f9e', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig9_lower-row-left_pp_Tbq_tHbq_tZbq.json', 'pp --> Tbq --> (tZ + tH)bq --> 0l', 13, 35.9, None),
    (29, 'T', 'Singlet', '04721f9f', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig9_lower-row-right_pp_Tbq_tHbq_tZbq.json', 'pp --> Tbq --> (tZ + tH)bq --> 0l', 13, 35.9, None),
    (30, 'T', 'Singlet', '05606f6', 'arXiv:1602.05606', 'ATLAS', '1602.05606_ATLAS_Fig6_pp_Tbj_Wb_singlet.json', 'pp --> Tbq --> bWbq --> 1l', 8, 20.3, None),
    (31, 'T', 'Singlet', '07409f4a', 'arXiv:1701.07409', 'CMS', '1701.07409_CMS_Fig4left_pp_Tb_tZ_singlet.json', 'pp --> Tbq --> tZbq --> 0l ', 13, 2.3, None),
    (32, 'T', 'Singlet', '03401f11a', 'arXiv:2305.03401', 'ATLAS', '2305.03401_ATLAS_Fig11a_pp_Tqt_Wb_Ht_Zt_singlet-k02.json', 'pp --> Tb(t)q --> tZ(H)b(t)q --> 1l', 13, 139.0, None),
    (33, 'T', 'Singlet', '03401f11b', 'arXiv:2305.03401', 'ATLAS', '2305.03401_ATLAS_Fig11b_pp_Tqt_Wb_Ht_Zt_singlet-k04.json', 'pp --> Tb(t)q --> tZ(H)b(t)q --> 1l', 13, 139.0, None),
    (34, 'T', 'Singlet', '03401f11c', 'arXiv:2305.03401', 'ATLAS', '2305.03401_ATLAS_Fig11c_pp_Tqt_Wb_Ht_Zt_singlet-k06.json', 'pp --> Tb(t)q --> tZ(H)b(t)q --> 1l', 13, 139.0, None),
    (35, 'T', 'Singlet', '16561f9c', 'arXiv:2402.16561', 'ATLAS', '2402.16561_ATLAS_Fig9c_pp_T_Ht_Zt_singlet.json', 'pp --> Tb(t)q --> tZb(t)q --> 0l', 13, 139.0, None),
    (36, 'T', 'Singlet', '09743f4c', 'arXiv:1812.09743', 'ATLAS', '1812.09743_ATLAS_Fig4c_pp_Tbq_Wb_Ht_Zt_singlet.json', 'pp --> Tbq --> tZbq --> 0l+1l', 13, 36.1, None),
    (37, 'T', 'Singlet', '07045f8a', 'arXiv:2201.07045', 'ATLAS', '2201.07045_ATLAS_Fig8-a_pp_TT_tH_k_T-0.1_singlet.json', 'pp --> Tbq --> tHbq --> 0l', 13, 139.0, None),
    (38, 'T', 'Singlet', '07045f8b', 'arXiv:2201.07045', 'ATLAS', '2201.07045_ATLAS_Fig8-b_pp_TT_tH_k_T-0.3_singlet.json', 'pp --> Tbq --> tHbq --> 0l', 13, 139.0, None),
    (39, 'T', 'Singlet', '07045f8c', 'arXiv:2201.07045', 'ATLAS', '2201.07045_ATLAS_Fig8-c_pp_TT_tH_k_T-0.5_singlet.json', 'pp --> Tbq --> tHbq --> 0l', 13, 139.0, None),
    (40, 'T', 'Singlet', '07045f8d', 'arXiv:2201.07045', 'ATLAS', '2201.07045_ATLAS_Fig8-d_pp_TT_tH_k_T-0.7_singlet.json', 'pp --> Tbq --> tHbq --> 0l', 13, 139.0, None),
    (41, 'T', 'Singlet', '07045f8e', 'arXiv:2201.07045', 'ATLAS', '2201.07045_ATLAS_Fig8-e_pp_TT_tH_k_T-0.9_singlet.json', 'pp --> Tbq --> tHbq --> 0l', 13, 139.0, None),
    (42, 'T', 'Singlet', '07045f8f', 'arXiv:2201.07045', 'ATLAS', '2201.07045_ATLAS_Fig8-f_pp_TT_tH_k_T-1.1_singlet.json', 'pp --> Tbq --> tHbq --> 0l', 13, 139.0, None),
    (43, 'T', 'Singlet', '07584f8a', 'arXiv:2307.07584', 'ATLAS', '2307.07584_ATLAS_Fig8a_pp_T_Wb_Ht_Zt_singlet_k03.json', 'pp --> Tb(t)q --> tZb(t)q --> l+l-+3l', 13, 139.0, None),
    (44, 'T', 'Singlet', '07584f8c', 'arXiv:2307.07584', 'ATLAS', '2307.07584_ATLAS_Fig8c_pp_T_Wb_Ht_Zt_singlet_k05.json', 'pp --> Tb(t)q --> tZb(t)q --> l+l-+3l', 13, 139.0, None),
    (45, 'T', 'Singlet', '07584f8e', 'arXiv:2307.07584', 'ATLAS', '2307.07584_ATLAS_Fig8e_pp_T_Wb_Ht_Zt_singlet_k07.json', 'pp --> Tb(t)q --> tZb(t)q --> l+l-+3l', 13, 139.0, None),
    (46, 'T', 'Singlet', '05336f4ul', 'arXiv:1612.05336', 'CMS', '1612.05336_CMS_fig4_upper_left_pp_Tbq_tHbq_LH_coupling.json', 'pp --> Tbq --> tHbq --> 0l', 13, 2.3, None),
    (47, 'T', 'Singlet', '032f6b', 'ATLAS-CONF-2016-032', 'ATLAS', 'ATLAS-CONF-2016-032_ATLAS_Fig6b_pp_TT_Wb_Zt_Ht_Singlet.json', 'pp --> TT --> l+l+', 13, 3.2, None),
    (48, 'T', 'Singlet', '104f16a', 'ATLAS-CONF-2016-104', 'ATLAS', 'ATLAS-CONF-2016-104_ATLAS_Fig16b_pp_TT_HtX_singlet.json', 'pp --> TT --> 0l', 13, 13.2, None),
    (49, 'T', 'Singlet', '7667f6', 'arXiv:1311.7667', 'CMS', '1311.7667_CMS_Fig6_pp_TT_bW_tH_Zt.json', 'pp --> TT --> 1l', 8, 19.5, None),
    (50, 'T', 'Singlet', '05071f4ur', 'arXiv:2405.05071', 'CMS', '2405.05071_CMS_Fig7_upper_right_pp_Tbq_tZbq.json', 'pp --> Tbq --> tZbq --> tbbbq --> 0l', 13, 138.0, None),
    (51, 'T', 'Singlet', '05071f4ul', 'arXiv:2405.05071', 'CMS', '2405.05071_CMS_Fig7_upper_left_pp_Tbq_tHbq.json', 'pp --> Tbq --> tHbq --> tbbbq --> 0l', 13, 138.0, None),
    (52, 'T', 'Singlet', '05071f4ll', 'arXiv:2405.05071', 'CMS', '2405.05071_CMS_Fig7_lower_left_pp_Tbq_tZbq_tHbq.json', 'pp --> Tbq --> (tH + tZ)bq --> tbbbq --> 0l', 13, 138.0, None),
    (53, 'T', 'Singlet', '05071f4lr', 'arXiv:2405.05071', 'CMS', '2405.05071_CMS_Fig7_lower_right_pp_Tbq_tZbq_tHbq.json', 'pp --> Tbq --> (tH + tZ)bq --> tbbbq --> 0l', 13, 138.0, None),
    (54, 'T', 'Singlet', '04605f7b', 'arXiv:1504.04605', 'ATLAS', '1504.04605_ATLAS_Fig7b_pp_TT_bW_tH_Zt_singlet.json', 'pp --> TT --> l+l+', 8, 20.3, None),
    (55, 'T', 'Singlet', '09678f17b', 'arXiv:1803.09678', 'ATLAS', '1803.09678_ATLAS_Fig17-b_pp_TT_Wb_Ht_Zt_Singlet.json', 'pp --> TT --> 1l', 13, 36.1, None),
    (56, 'T', 'Singlet', '15413f7a', 'arXiv:2210.15413', 'ATLAS', '2210.15413_ATLAS_Fig7-a_pp_TT_Wb-Zt_Ht_Singlet.json', 'pp --> TT --> l+l-+3l', 13, 139.0, None),
    (57, 'T', 'Singlet', '10555f15', 'arXiv:1806.10555', 'ATLAS', '1806.10555_ATLAS_Fig15_pp_T_Zt_Singlet.json', 'pp --> Tbq --> tZbq --> l+l-+>=3l', 13, 36.1, None),
    (58, 'T', 'Singlet', '08328', 'arXiv:1701.08328', 'CMS', '1701.08328_CMS_fig5_pp_Tbq_or_Ybq_bW.json', 'pp --> Tbq --> bWbq --> 1l', 13, 2.3, None),
    (59, 'T', 'Singlet', '20273f5a', 'arXiv:2409.20273', 'ATLAS', '2409.20273_ATLAS_Fig5a_pp_Qbq_Wb_k05_Tsinglet_Ydoublet.json', 'pp --> Tbq --> bWbq --> 0l', 13, 139.0, None),
    (60, 'T', 'Singlet', '20273f5b', 'arXiv:2409.20273', 'ATLAS', '2409.20273_ATLAS_Fig5b_pp_Qbq_Wb_k07_Tsinglet_YDoublet.json', 'pp --> Tbq --> bWbq --> 0l', 13, 139.0, None),
    (0, 'T', 'Doublet', '10751', 'arXiv:1705.10751', 'ATLAS', '1705.10751_ATLAS_fig6c_pp_TT_Doublet.json', 'pp --> TT --> 1l', 13, 36.1, 'XTorTB'),
    (1, 'T', 'Doublet', '07327', 'arXiv:2209.07327', 'CMS', '2209.07327_CMS_f9b_pp_TTbar_Doublet.json', 'pp --> TT --> 1l+l+l++3l', 13, 138.0, 'XTorTB'),
    (2, 'T', 'Doublet', '05263', 'arXiv:2212.05263', 'ATLAS', '2212.05263_ATLAS_fig7e_pp_TT_Zt_T,B_or_X,T_Doublet.json', 'pp --> TT --> 1l', 13, 36.1, 'XTorTB'),
    (3, 'T', 'Doublet', '10555', 'arXiv:1806.10555', 'ATLAS', '1806.10555_ATLAS_Fig-13_c_pp_TT_Zt_doublet.json', 'pp --> TT --> l+l-+>=3l', 13, 139.0, 'XT'),
    (4, 'T', 'Doublet', '04758', 'arXiv:1805.04758', 'CMS', '1805.04758_CMS_Fig8-upper-row-right_pp_TT_Zt_tH_Doublet.json', 'pp --> TT --> 1l+l+l++3l', 13, 35.9, 'XTorTB'),
    (5, 'T', 'Doublet', '00999f10b', 'arXiv:1612.00999', 'CMS', '1612.00999_CMS_Fig10-right_pp_Ttq_tH.json', 'pp --> Ttq --> tHtq --> 1l', 13, 2.3, 'XTorTB'),
    (6, 'T', 'Doublet', '03408', 'arXiv:1706.03408', 'CMS', '1706.03408_CMS_Fig-12-right_pp_TT_Zt_tH_Doublet.json', 'pp --> TT --> 1l', 13, 2.6, 'XTorTB'),
    (7, 'T', 'Doublet', '01062f5b', 'arXiv:1708.01062', 'CMS', '1708.01062_CMS_Fig5-right_pp_Ttq_tZtq_Doublet-RH.json', 'pp --> Ttq --> tZtq --> l+l-', 13, 35.9, 'XTorTB'),
    (8, 'T', 'Doublet', '5500fd', 'arXiv:1409.5500', 'ATLAS', '1409.5500_ATLAS_Fig-12-d_pp_TT_Zt_doublet.json', 'pp --> TT --> l+l-+3l', 8, 20.3, 'TB'),
    (9, 'T', 'Doublet', '04721f10a', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig10_upper-row-left_pp_Ttq_tHtq.json', 'pp --> Ttq --> tHtq --> 0l', 13, 35.9, 'TB'),
    (10, 'T', 'Doublet', '04721f10b', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig10_upper-row-right_pp_Ttq_tHtq.json', 'pp --> Ttq --> tHtq --> 0l', 13, 35.9, 'TB'),
    (11, 'T', 'Doublet', '04721f10c', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig10_middle-row-left_pp_Ttq_tZtq.json', 'pp --> Ttq --> tZtq --> 0l', 13, 35.9, 'TB'),
    (12, 'T', 'Doublet', '04721f10d', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig10_middle-row-right_pp_Ttq_tZtq.json', 'pp --> Ttq --> tZtq --> 0l', 13, 35.9, 'TB'),
    (13, 'T', 'Doublet', '04721f10e', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig10_lower-row-left_pp_Ttq_tH-tZ_tq.json', 'pp --> Ttq --> (tZ + tH)tq --> 0l', 13, 35.9, 'TB'),
    (14, 'T', 'Doublet', '04721f10f', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig10_lower-row-right_pp_Ttq_tH-tZ_tq.json', 'pp --> Ttq --> (tZ + tH)tq --> 0l', 13, 35.9, 'TB'),
    (15, 'T', 'Doublet', '04721f11a', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig11_upper-row-left_pp_Ttq_tHtq.json', 'pp --> Ttq --> tHtq --> 0l', 13, 35.9, 'TB'),
    (16, 'T', 'Doublet', '04721f11b', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig11_upper-row-right_pp_Ttq_tHtq.json', 'pp --> Ttq --> tHtq --> 0l', 13, 35.9, 'TB'),
    (17, 'T', 'Doublet', '04721f11c', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig11_middle-row-left_pp_Ttq_tZtq.json', 'pp --> Ttq --> tZtq --> 0l', 13, 35.9, 'TB'),
    (18, 'T', 'Doublet', '04721f11d', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig11_middle-row-right_pp_Ttq_tZtq.json', 'pp --> Ttq --> tZtq --> 0l', 13, 35.9, 'TB'),
    (19, 'T', 'Doublet', '04721f11e', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig11_lower-row-left_pp_Ttq_tZ-tH_tq.json', 'pp --> Ttq --> (tZ + tH)tq --> 0l', 13, 35.9, 'TB'),
    (20, 'T', 'Doublet', '04721f11f', 'arXiv:1909.04721', 'CMS', '1909.04721_CMS_Fig11_lower-row-right_pp_Ttq_tZ-tH_tq.json', 'pp --> Ttq --> (tZ + tH)tq --> 0l', 13, 35.9, 'TB'),
    (21, 'T', 'Doublet', '07409f4b', 'arXiv:1701.07409', 'CMS', '1701.07409_CMS_Fig4right_pp_Tt_tZ_doublet.json', 'pp --> Ttq --> tZtq --> 0l', 13, 2.3, 'XTorTB'),
    (22, 'T', 'Doublet', '04306fc', 'arXiv:1505.04306', 'ATLAS', '1505.04306_ATLAS_Fig18-c_pp_TT_Ht+X_Doublet.json', 'pp --> TT --> 1l', 8, 20.3, 'XTorTB'),
    (23, 'T', 'Doublet', '03401f12a', 'arXiv:2305.03401', 'ATLAS', '2305.03401_ATLAS_Fig12a_pp_Tqt_Ht_Zt_doublet-k02.json', 'pp --> Tb(t)q --> tZ(H)b(t)q --> 1l', 13, 139.0, 'TB'),
    (24, 'T', 'Doublet', '03401f12b', 'arXiv:2305.03401', 'ATLAS', '2305.03401_ATLAS_Fig12b_pp_Tqt_Ht_Zt_doublet-k04.json', 'pp --> Tb(t)q --> tZ(H)b(t)q --> 1l', 13, 139.0, 'TB'),
    (25, 'T', 'Doublet', '03401f12c', 'arXiv:2305.03401', 'ATLAS', '2305.03401_ATLAS_Fig12c_pp_Tqt_Ht_Zt_doublet-k06.json', 'pp --> Tb(t)q --> tZ(H)b(t)q --> 1l', 13, 139.0, 'TB'),
    (26, 'T', 'Doublet', '07584f8b', 'arXiv:2307.07584', 'ATLAS', '2307.07584_ATLAS_Fig8b_pp_T_Ht_Zt_doublet_k03.json', 'pp --> Ttq --> tZtq --> l+l-+3l', 13, 139.0, 'XTorTB'),
    (27, 'T', 'Doublet', '07584f8d', 'arXiv:2307.07584', 'ATLAS', '2307.07584_ATLAS_Fig8d_pp_T_Ht_Zt_doublet_k05.json', 'pp --> Ttq --> tZtq --> l+l-+3l', 13, 139.0, 'XTorTB'),
    (28, 'T', 'Doublet', '07584f8f', 'arXiv:2307.07584', 'ATLAS', '2307.07584_ATLAS_Fig8f_pp_T_Ht_Zt_doublet_k07.json', 'pp --> Ttq --> tZtq --> l+l-+3l', 13, 139.0, 'XTorTB'),
    (29, 'T', 'Doublet', '05336f4lr', 'arXiv:1612.05336', 'CMS', '1612.05336_CMS_fig4_lower_right_pp_Tbq_tHbq_RH_coupling.json', 'pp --> Ttq --> tHtq --> 0l', 13, 2.3, 'XTorTB'),
    (30, 'T', 'Doublet', '104f16a', 'ATLAS-CONF-2016-104', 'ATLAS', 'ATLAS-CONF-2016-104_ATLAS_Fig16a_pp_TT_HtX_doublet.json', 'pp --> TT --> 0l+1l', 13, 13.2, 'XTorTB'),
    (31, 'T', 'Doublet', '09678f17a', 'arXiv:1803.09678', 'ATLAS', '1803.09678_ATLAS_Fig17-a_pp_TT_Ht_Zt_Doublet.json', 'pp --> TT --> 1l', 13, 36.1, 'XTorTB'),
    (32, 'T', 'Doublet', '15413f7c', 'arXiv:2210.15413', 'ATLAS', '2210.15413_ATLAS_Fig7-c_pp_TT_Zt_Ht_Doublet-X-T.json', 'pp --> TT --> l+l-+3l', 13, 139.0, 'XT'),
    (0, 'T', 'Pure', '05263', 'arXiv:2212.05263', 'ATLAS', '2212.05263_ATLAS_fig7a_pp_TT_tZ.json', 'pp --> TT --> ZtZt --> 1l', 13, 139.0, None),
    (1, 'T', 'Pure', '97680', 'arXiv:1812.09768', 'CMS', '1812.09768_CMS_fig4a_pp_TT_tZ.json', 'pp --> TT --> ZtZt --> l+l-', 13, 36.1, None),
    (2, 'T', 'Pure', '10751fig6a', 'arXiv:1705.10751', 'ATLAS', '1705.10751_ATLAS_fig6a_pp_TT_Zt.json', 'pp --> TT --> ZtZt --> 1l', 13, 36.1, None),
    (3, 'T', 'Pure', '17710', 'arXiv:1808.01771', 'ATLAS', '1808.01771_ATLAS_fig13a_pp_TT_Ht.json', 'pp --> TT --> HtHt --> 0l', 13, 36.1, None),
    (4, 'T', 'Pure', '19520', 'arXiv:1503.01952', 'CMS', '1503.01952_CMS_fig13_pp_TT_tH.json', 'pp --> TT --> HtHt --> 0l', 8, 19.7, None),
    (5, 'T', 'Pure', '33470fig4a', 'arXiv:1707.03347', 'ATLAS', '1707.03347_ATLAS_fig4a_pp_TT_Wb.json', 'pp --> TT --> WbWb --> 1l', 13, 36.1, None),
    (6, 'T', 'Pure', '10555', 'arXiv:1806.10555', 'ATLAS', '1806.10555_ATLAS_Fig-13e_pp_TT_Zt.json', 'pp --> TT --> ZtZt --> l+l-', 13, 36.1, None),
    (7, 'T', 'Pure', '04177', 'arXiv:1509.04177', 'CMS', '1509.04177_CMS_fig8_pp_TT_bW.json', 'pp --> TT --> WbWb --> 0l+1l+2l', 8, 19.7, None),
    (8, 'T', 'Pure', '04177', 'arXiv:1509.04177', 'CMS', '1509.04177_CMS_fig8_pp_TT_tH.json', 'pp --> TT --> HtHt --> 0l+1l+2l', 8, 19.7, None),
    (9, 'T', 'Pure', '04177', 'arXiv:1509.04177', 'CMS', '1509.04177_CMS_fig8_pp_TT_tZ.json', 'pp --> TT --> ZtZt --> 0l+1l+2l', 8, 19.7, None),
    (10, 'T', 'Pure', '03408', 'arXiv:1706.03408', 'CMS', '1706.03408_CMS_Fig-11-left_pp_TT_bW.json', 'pp --> TT --> WbWb --> 1l', 13, 2.6, None),
    (11, 'T', 'Pure', '03408', 'arXiv:1706.03408', 'CMS', '1706.03408_CMS_Fig-11-right_pp_TT_tH.json', 'pp --> TT --> HtHt --> 1l', 13, 2.6, None),
    (12, 'T', 'Pure', '0471fa', 'arXiv:1209.0471', 'CMS', '1209.0471_CMS_Fig6_upper_pp_TT_bW.json', 'pp --> TT --> WbWb --> 1l', 13, 2.6, None),
    (13, 'T', 'Pure', '0471fb', 'arXiv:1209.0471', 'CMS', '1209.0471_CMS_Fig6_middle_pp_TT_bW.json', 'pp --> TT --> WbWb --> 1l', 7, 5.0, None),
    (14, 'T', 'Pure', '0471fc', 'arXiv:1209.0471', 'CMS', '1209.0471_CMS_Fig6_lower_pp_TT_bW.json', 'pp --> TT --> WbWb --> 1l', 7, 5.0, None),
    (15, 'T', 'Pure', '04306fa', 'arXiv:1505.04306', 'ATLAS', '1505.04306_ATLAS_Fig18-a_pp_TT_Wb+X.json', 'pp --> TT --> WbWb --> 1l', 8, 20.3, None),
    (16, 'T', 'Pure', '5468', 'arXiv:1210.5468', 'ATLAS', '1210.5468_ATLAS_Fig-3_pp_tt_Wb.json', 'pp --> TT --> WbWb --> 1l ', 7, 4.7, None),
    (17, 'T', 'Pure', '01539', 'arXiv:1710.01539', 'CMS', '1710.01539_CMS_Fig4_upper_pp_TT_bW.json', 'pp --> TT --> WbWb --> 1l', 13, 35.8, None),
    (18, 'T', 'Pure', '104f15a', 'ATLAS-CONF-2016-104', 'ATLAS', 'ATLAS-CONF-2016-104_ATLAS_Fig15a_pp_TT_tH.json', 'pp --> TT --> HtHt --> 0l+1l ', 13, 13.2, None),
    (19, 'T', 'Pure', '104f15b', 'ATLAS-CONF-2016-104', 'ATLAS', 'ATLAS-CONF-2016-104_ATLAS_Fig15b_pp_TT_tZ.json', 'pp --> TT --> ZtZt --> 0l+1l ', 13, 13.2, None),
    (20, 'T', 'Pure', '11903fa', 'arXiv:1906.11903', 'CMS', '1906.11903_CMS_Fig6_lower_left_pp_TT_bW.json', 'pp --> TT --> WbWb --> 0l', 13, 35.9, None),
    (21, 'T', 'Pure', '11903fc', 'arXiv:1906.11903', 'CMS', '1906.11903_CMS_Fig6_upper_left_pp_TT_tZ.json', 'pp --> TT --> ZtZt --> 0l', 13, 35.9, None),
    (22, 'T', 'Pure', '11903ff', 'arXiv:1906.11903', 'CMS', '1906.11903_CMS_Fig6_middle_left_pp_TT_tH.json', 'pp --> TT --> HtHt --> 0l', 13, 35.9, None),
    (23, 'T', 'Pure', '5410f2', 'arXiv:1203.5410', 'CMS', '1203.5410_CMS_Fig2_pp_tt_bWbW.json', 'pp --> TT --> WbWb --> l+l-', 7, 5.0, None),
    (24, 'T', 'Pure', '3076f2', 'arXiv:1202.3076', 'ATLAS', '1202.3076_ATLAS_Fig2_pp_tt_WbWb.json', 'pp --> TT --> WbWb --> 1l', 7, 1.04, None),
    (25, 'T', 'Pure', '03903f9', 'arXiv:1606.03903', 'ATLAS', '1606.03903_ATLAS_Fig9_pp_TT_Zt.json', 'pp --> TT --> ZtZt --> 1l', 13, 3.2, None),
    (26, 'T', 'Pure', '15413f7e', 'arXiv:2210.15413', 'ATLAS', '2210.15413_ATLAS_Fig7-e_pp_TT_Zt.json', 'pp --> TT --> ZtZt --> l+l-', 13, 139.0, None),
    (27, 'T', 'Pure', '17165f5a', 'arXiv:2401.17165', 'ATLAS', '2401.17165_ATLAS_Fig5a_pp_TT_bW.json', 'pp --> TT --> WbWb --> 1l + jets', 13, 140.0, None),
    (28, 'T', 'Pure', '07327fwb', 'arXiv:2209.07327', 'CMS', '2209.07327_CMS_Table11_pp_TT_Wb.json', 'pp --> TT --> WbWb --> >=1l + jets', 13, 138.0, None),
    (29, 'T', 'Pure', '07327fzt', 'arXiv:2209.07327', 'CMS', '2209.07327_CMS_Table11_pp_TT_Zt.json', 'pp --> TT --> ZtZt --> >=1l + jets', 13, 138.0, None),
    (30, 'T', 'Pure', '07327fht', 'arXiv:2209.07327', 'CMS', '2209.07327_CMS_Table11_pp_TT_Ht.json', 'pp --> TT --> HtHt --> >=1l + jets', 13, 138.0, None),
)


__all__ = [
    "VLQSpecies",
    "ProductionMode",
    "BoundsCatalogError",
    "RegistryConsistencyError",
    "ExperimentalLimitDescriptor",
    "LimitCurveData",
    "RegistryDiagnostics",
    "JsonLimitCurveReader",
    "ExperimentalBoundsRepository",
    "Tables",
]
