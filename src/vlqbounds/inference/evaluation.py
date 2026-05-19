from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from vlqbounds.theory import (
    append_dataframe_row,
    compute_limit_ratios,
    convert_contact_factor_to_kappa,
    is_missing_sequence,
    is_theory_cross_section_dominant,
    load_coupling_limit_curves,
    parse_constraint_expression,
    validate_nonzero_sine,
    validate_parameter_contract,
)
from vlqbounds.theory.theory_repository import interpolate_surface_value


# ======================================================================================
# Core types
# ======================================================================================

FloatArray = NDArray[np.float64]
ConstraintIdentifier = str


class VLQSpecies(str, Enum):
    T = "T"
    B = "B"
    X = "X"
    Y = "Y"


class Representation(str, Enum):
    SINGLET = "Singlet"
    DOUBLET = "Doublet"
    TRIPLET = "Triplet"


class Observable(str, Enum):
    KAPPA = "kappa"
    WIDTH_MASS_RATIO = "width_mass_ratio"


class InterpolationMode(str, Enum):
    EXPERIMENTAL = "experimental"
    THEORY = "theory"


# ======================================================================================
# Immutable constraint model
# ======================================================================================

@dataclass(frozen=True, slots=True)
class ConstraintRecord:
    species: VLQSpecies
    observable: Observable
    representation: Representation
    condition: str
    identifiers: tuple[ConstraintIdentifier, ...]

    @property
    def is_empty(self) -> bool:
        return len(self.identifiers) == 0


@dataclass(frozen=True, slots=True)
class ConstraintQuery:
    species: VLQSpecies
    observable: Observable
    representation: Representation
    condition: str


@dataclass(slots=True)
class ConstraintRepository:
    _records: tuple[ConstraintRecord, ...]
    _index: dict[tuple[VLQSpecies, Observable, Representation, str], ConstraintRecord] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._index = {
            (record.species, record.observable, record.representation, record.condition): record
            for record in self._records
        }

    def fetch(self, query: ConstraintQuery) -> ConstraintRecord:
        try:
            return self._index[
                (query.species, query.observable, query.representation, query.condition)
            ]
        except KeyError as exc:
            raise KeyError(
                "No constraint record found for "
                f"{query.species.value}/{query.observable.value}/"
                f"{query.representation.value}/{query.condition}"
            ) from exc

    def identifiers(
        self,
        *,
        species: VLQSpecies,
        observable: Observable,
        representation: Representation,
        condition: str,
    ) -> tuple[ConstraintIdentifier, ...]:
        return self.fetch(
            ConstraintQuery(
                species=species,
                observable=observable,
                representation=representation,
                condition=condition,
            )
        ).identifiers

    def select(
        self,
        *,
        species: Optional[VLQSpecies] = None,
        observable: Optional[Observable] = None,
        representation: Optional[Representation] = None,
    ) -> tuple[ConstraintRecord, ...]:
        return tuple(
            record
            for record in self._records
            if (species is None or record.species is species)
            and (observable is None or record.observable is observable)
            and (representation is None or record.representation is representation)
        )

    def conditions(
        self,
        *,
        species: VLQSpecies,
        observable: Observable,
        representation: Representation,
    ) -> tuple[str, ...]:
        return tuple(
            record.condition
            for record in self.select(
                species=species,
                observable=observable,
                representation=representation,
            )
        )


# ======================================================================================
# Declarative builders
# ======================================================================================

def _unique(*identifiers: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(identifiers))


def _record(
    species: VLQSpecies,
    observable: Observable,
    representation: Representation,
    condition: str,
    *identifiers: str,
) -> ConstraintRecord:
    return ConstraintRecord(
        species=species,
        observable=observable,
        representation=representation,
        condition=condition,
        identifiers=_unique(*identifiers),
    )


CONSTRAINT_RECORDS: Final[tuple[ConstraintRecord, ...]] = (
    # ------------------------------------------------------------------
    # T : width/mass ratio
    # ------------------------------------------------------------------
    _record(VLQSpecies.T, Observable.WIDTH_MASS_RATIO, Representation.SINGLET, "r==0.01",
            "05071f4ur", "05071f4ul", "05071f4ll", "05071f4lr"),
    _record(VLQSpecies.T, Observable.WIDTH_MASS_RATIO, Representation.DOUBLET, "r==0.01"),

    _record(VLQSpecies.T, Observable.WIDTH_MASS_RATIO, Representation.SINGLET, "r<=0.05",
            "02227fa", "12802", "04721f8a", "04721f8c", "04721f8e", "17605f35", "17605f40u"),
    _record(VLQSpecies.T, Observable.WIDTH_MASS_RATIO, Representation.DOUBLET, "r<=0.05",
            "04721f10a", "04721f10c", "04721f10e"),

    _record(VLQSpecies.T, Observable.WIDTH_MASS_RATIO, Representation.SINGLET, "0.05<=r<=0.3",
            "02227fb", "02227fc", "02227fd",
            "04721f8b", "04721f9a", "04721f9b",
            "04721f8d", "04721f9c", "04721f9d",
            "04721f8f", "04721f9e", "04721f9f"),
    _record(VLQSpecies.T, Observable.WIDTH_MASS_RATIO, Representation.DOUBLET, "0.05<=r<=0.3",
            "04721f10b", "04721f11a", "04721f11b",
            "04721f10d", "04721f11c", "04721f11d",
            "04721f10f", "04721f11e", "04721f11f"),

    # ------------------------------------------------------------------
    # T : kappa
    # ------------------------------------------------------------------
    _record(VLQSpecies.T, Observable.KAPPA, Representation.SINGLET, "0.2<=k<=0.6",
            "03401f11a", "03401f11b", "03401f11c"),
    _record(VLQSpecies.T, Observable.KAPPA, Representation.DOUBLET, "0.2<=k<=0.6",
            "03401f12a", "03401f12b", "03401f12c"),

    _record(VLQSpecies.T, Observable.KAPPA, Representation.SINGLET, "0.1<=k<=1.1",
            "07045f8a", "07045f8b", "07045f8c", "07045f8d", "07045f8e", "07045f8f"),
    _record(VLQSpecies.T, Observable.KAPPA, Representation.DOUBLET, "0.1<=k<=1.1"),

    _record(VLQSpecies.T, Observable.KAPPA, Representation.SINGLET, "0.3<=k<=0.7",
            "07584f8a", "07584f8c", "07584f8e"),
    _record(VLQSpecies.T, Observable.KAPPA, Representation.DOUBLET, "0.3<=k<=0.7",
            "07584f8b", "07584f8d", "07584f8f"),

    _record(VLQSpecies.T, Observable.KAPPA, Representation.SINGLET, "0.5<=k<=0.7",
            "20273f5a", "20273f5b"),
    _record(VLQSpecies.T, Observable.KAPPA, Representation.DOUBLET, "0.5<=k<=0.7"),

    _record(VLQSpecies.T, Observable.KAPPA, Representation.SINGLET, "k==0.5",
            "10555f15", "16561f9c", "09743f4c", "01062f5a",
            "00999f10a", "05336f4ul", "07409f4a", "09743f4c", "08328"),
    _record(VLQSpecies.T, Observable.KAPPA, Representation.DOUBLET, "k==0.5",
            "00999f10b", "05336f4lr", "07409f4b", "01062f5b"),

    _record(VLQSpecies.T, Observable.KAPPA, Representation.SINGLET, "k==1",
            "05606f6"),
    _record(VLQSpecies.T, Observable.KAPPA, Representation.DOUBLET, "k==1"),

    # ------------------------------------------------------------------
    # B : kappa
    # ------------------------------------------------------------------
    _record(VLQSpecies.B, Observable.KAPPA, Representation.SINGLET, "0.3<=k<=0.5",
            "02595f8a", "02595f8c", "02595f8e"),
    _record(VLQSpecies.B, Observable.KAPPA, Representation.DOUBLET, "0.3<=k<=0.5",
            "02595f8b", "02595f8d", "02595f8f"),

    _record(VLQSpecies.B, Observable.KAPPA, Representation.SINGLET, "k==0.5",
            "07409f5r", "07409f5l"),
    _record(VLQSpecies.B, Observable.KAPPA, Representation.DOUBLET, "k==0.5"),

    # ------------------------------------------------------------------
    # B : width/mass ratio
    # ------------------------------------------------------------------
    _record(VLQSpecies.B, Observable.WIDTH_MASS_RATIO, Representation.SINGLET, "0.01<=r<=0.3",
            "01486Fig6a", "01486Fig6b", "01486Fig6c", "01486Fig6d"),
    _record(VLQSpecies.B, Observable.WIDTH_MASS_RATIO, Representation.DOUBLET, "0.01<=r<=0.3",
            "01486Fig6a", "01486Fig6b", "01486Fig6c", "01486Fig6d"),

    _record(VLQSpecies.B, Observable.WIDTH_MASS_RATIO, Representation.SINGLET, "r<=0.1",
            "10216f4l", "10216f4r"),
    _record(VLQSpecies.B, Observable.WIDTH_MASS_RATIO, Representation.DOUBLET, "r<=0.1",
            "10216f4ld", "10216f4rd"),

    # ------------------------------------------------------------------
    # X : width/mass ratio / kappa
    # ------------------------------------------------------------------
    _record(VLQSpecies.X, Observable.WIDTH_MASS_RATIO, Representation.DOUBLET, "r==0.01",
            "10216f4ld"),

    # ------------------------------------------------------------------
    # Y : kappa
    # ------------------------------------------------------------------
    _record(VLQSpecies.Y, Observable.KAPPA, Representation.DOUBLET, "0.5<=k<=0.7",
            "20273f5a", "20273f5b"),
    _record(VLQSpecies.Y, Observable.KAPPA, Representation.TRIPLET, "0.5<=k<=0.7"),

    _record(VLQSpecies.Y, Observable.KAPPA, Representation.DOUBLET, "k==0.35",
            "08328f5"),
    _record(VLQSpecies.Y, Observable.KAPPA, Representation.TRIPLET, "k==0.35",
            "08328f5"),

    _record(VLQSpecies.Y, Observable.KAPPA, Representation.DOUBLET, "k==0.7",
            "05606f6"),
    _record(VLQSpecies.Y, Observable.KAPPA, Representation.TRIPLET, "k==0.7"),
)

CONSTRAINTS: Final[ConstraintRepository] = ConstraintRepository(CONSTRAINT_RECORDS)


# ======================================================================================
# Result abstraction
# ======================================================================================

@dataclass(slots=True)
class ExclusionResult(ABC):
    result_flag: Optional[int] = None
    dominant_channel: Optional[int] = None
    observed_ratio: Optional[float] = None
    expected_ratio: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def finalize(self, position: int) -> None:
        raise NotImplementedError

    @property
    def allowed(self) -> Optional[bool]:
        return None if self.result_flag is None else bool(self.result_flag)

    @property
    def excluded(self) -> Optional[bool]:
        return None if self.result_flag is None else not bool(self.result_flag)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_flag": self.result_flag,
            "dominant_channel": self.dominant_channel,
            "observed_ratio": self.observed_ratio,
            "expected_ratio": self.expected_ratio,
            "metadata": dict(self.metadata),
        }

    def __str__(self) -> str:
        return (
            "Exclusion result\n"
            "----------------\n"
            f"result_flag      : {self.result_flag if self.result_flag is not None else 'N/A'} (1 = allowed, 0 = excluded)\n"
            f"dominant_channel : {self.dominant_channel if self.dominant_channel is not None else 'N/A'}\n"
            f"observed_ratio   : {self.observed_ratio if self.observed_ratio is not None else 'N/A'}\n"
            f"expected_ratio   : {self.expected_ratio if self.expected_ratio is not None else 'N/A'}"
        )


# ======================================================================================
# Utility interface
# ======================================================================================

def validate_sine_parameter(value: float) -> None:
    validate_nonzero_sine(value)


def validate_model_parameter_contract(
    parameters: Mapping[str, Any],
    required_mass_key: str,
    valid_keys: Iterable[str],
) -> None:
    validate_parameter_contract(parameters, required_mass_key, valid_keys)


def calculate_observed_expected_ratios(
    theory_value: float,
    observed_limit: float,
    expected_limit: float,
) -> tuple[float, float]:
    return compute_limit_ratios(theory_value, observed_limit, expected_limit)


def is_coupling_constraint_more_restrictive(
    coupling_ratio: float,
    cross_section_ratio: float,
) -> bool:
    return is_theory_cross_section_dominant(coupling_ratio, cross_section_ratio)


def parse_constraint_condition(expression: str) -> Any:
    return parse_constraint_expression(expression)


def transform_contact_factor_to_kappa(
    contact_factor: FloatArray,
    heavy_mass: FloatArray,
    model_label: str,
) -> FloatArray:
    return convert_contact_factor_to_kappa(contact_factor, heavy_mass, model_label)


def append_scan_dataframe_row(dataframe: pd.DataFrame, **row_data: Any) -> pd.DataFrame:
    return append_dataframe_row(dataframe, **row_data)


def sequence_is_missing_or_empty(values: Optional[Sequence[Any]]) -> bool:
    return is_missing_sequence(values)


def to_json_resource_name(resource_name: str) -> str:
    return str(Path(resource_name).with_suffix(".json"))


def populate_coupling_limit_arrays(
    resource_names: Sequence[str],
    file_count: int,
    mass_arrays: Sequence[list[Any]],
    expected_arrays: Sequence[list[Any]],
    observed_arrays: Sequence[list[Any]],
    experiment_labels: Sequence[str],
    species: VLQSpecies = VLQSpecies.T,
    data_root: Optional[str | Path] = None,
) -> None:
    selected_resources = tuple(resource_names[:file_count])
    selected_experiments = tuple(experiment_labels[:file_count])

    if len(selected_resources) != len(selected_experiments):
        raise ValueError(
            "The selected coupling resources and experiment labels must have identical length."
        )

    envelopes = load_coupling_limit_curves(
        [to_json_resource_name(name) for name in selected_resources],
        selected_experiments,
        vlq_species=species.value,
        data_root=data_root,
    )

    for index, envelope in enumerate(envelopes):
        mass_arrays[0][index] = envelope.observed_masses
        mass_arrays[1][index] = envelope.expected_masses
        observed_arrays[0][index] = envelope.observed_lower
        observed_arrays[1][index] = envelope.observed_upper
        expected_arrays[0][index] = envelope.expected_lower
        expected_arrays[1][index] = envelope.expected_upper


def interpolate_surface(
    *,
    mode: InterpolationMode,
    experimental_input: Optional[tuple[Any, ...]] = None,
    theory_input: Optional[tuple[Any, ...]] = None,
) -> Any:
    if mode is InterpolationMode.EXPERIMENTAL:
        return interpolate_surface_value(
            experimental_input=experimental_input,
            mode=InterpolationMode.EXPERIMENTAL.value,
        )

    return interpolate_surface_value(
        theory_input=theory_input,
        mode=InterpolationMode.THEORY.value,
    )


__all__ = [
    "FloatArray",
    "ConstraintIdentifier",
    "VLQSpecies",
    "Representation",
    "Observable",
    "InterpolationMode",
    "ConstraintRecord",
    "ConstraintQuery",
    "ConstraintRepository",
    "ConstraintRecord",
    "CONSTRAINT_RECORDS",
    "CONSTRAINTS",
    "ExclusionResult",
    "validate_sine_parameter",
    "validate_model_parameter_contract",
    "calculate_observed_expected_ratios",
    "is_coupling_constraint_more_restrictive",
    "parse_constraint_condition",
    "transform_contact_factor_to_kappa",
    "append_scan_dataframe_row",
    "sequence_is_missing_or_empty",
    "to_json_resource_name",
    "populate_coupling_limit_arrays",
    "interpolate_surface",
]
