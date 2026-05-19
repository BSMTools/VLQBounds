from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from vlqbounds.theory import constants as c

from .evaluation import (
    CONSTRAINTS,
    ExclusionResult,
    InterpolationMode,
    Observable,
    Representation,
    VLQSpecies,
    interpolate_surface,
    parse_constraint_condition,
)
from vlqbounds.constraints.experimental_repository import ExperimentalBoundsRepository


class CrossSectionLimitEngine(ExperimentalBoundsRepository, ExclusionResult):
    """
    Engine responsible for confronting model predictions with cross-section limits.

    """

    def __init__(self, m):
        ExperimentalBoundsRepository.__init__(self, m)
        ExclusionResult.__init__(self)
        self.m = m
        self.most_sensitive_channels: List[int] = []

    # ------------------------------------------------------------------
    # Internal classification helpers
    # ------------------------------------------------------------------
    def _active_species(self) -> VLQSpecies:
        if getattr(self, "VLB", False):
            return VLQSpecies.B
        if getattr(self, "VLX", False):
            return VLQSpecies.X
        if getattr(self, "VLY", False):
            return VLQSpecies.Y
        return VLQSpecies.T

    def _active_representation(self) -> Optional[Representation]:
        model_name = self.m.model()
        if model_name == "Singlet":
            return Representation.SINGLET
        if model_name == "Doublet":
            return Representation.DOUBLET
        if model_name == "Triplet":
            return Representation.TRIPLET
        return None

    def _constraint_conditions(self, observable: Observable) -> tuple[str, ...]:
        representation = self._active_representation()
        if representation is None:
            return ()
        return CONSTRAINTS.conditions(
            species=self._active_species(),
            observable=observable,
            representation=representation,
        )

    def _constraint_identifiers(self, observable: Observable, condition: str) -> tuple[str, ...]:
        representation = self._active_representation()
        if representation is None:
            return ()
        return CONSTRAINTS.identifiers(
            species=self._active_species(),
            observable=observable,
            representation=representation,
            condition=condition,
        )

    @staticmethod
    def _condition_is_satisfied(condition: str, parameter: float) -> bool:
        values, is_exact = parse_constraint_condition(condition)
        if len(values) == 2:
            lower, upper = values
            return lower <= parameter <= upper
        if len(values) == 1:
            target = values[0]
            if is_exact:
                return abs(parameter - target) <= c.Threshold
            return parameter <= target
        return False

    def _find_condition_for_identifier(self, identifier: str, observable: Observable) -> Optional[str]:
        for condition in self._constraint_conditions(observable):
            if identifier in self._constraint_identifiers(observable, condition):
                return condition
        return None

    # ------------------------------------------------------------------
    # Public helpers retained for engine use
    # ------------------------------------------------------------------
    def get_ratio_and_universal_coupling(self):
        if self.m.model() != "Pure":
            return self.m.get_width_mass_ratio(), self.m.get_coupling_strength()
        return None, None

    def get_pair_prod_channels(self, key: str) -> Optional[int]:
        if key in self.cs_keys["pair_prod"]:
            return self.key.index(key)
        return None

    def process_condition(self, channel_key: str, observable: Observable, parameter: float) -> Optional[int]:
        if channel_key in self.cs_keys["pair_prod"]:
            return self.get_pair_prod_channels(channel_key)

        for condition in self._constraint_conditions(observable):
            identifiers = self._constraint_identifiers(observable, condition)
            if channel_key in identifiers and self._condition_is_satisfied(condition, parameter):
                return self.key.index(channel_key)
        return None

    def get_channels_from_limit_condition(
        self,
        channel_key: str,
        width_ratio: Optional[float],
        coupling_strength: Optional[float],
    ) -> Optional[int]:
        if width_ratio is not None:
            channel = self.process_condition(channel_key, Observable.WIDTH_MASS_RATIO, width_ratio)
            if channel is not None:
                return channel
        if coupling_strength is not None:
            channel = self.process_condition(channel_key, Observable.KAPPA, coupling_strength)
            if channel is not None:
                return channel
        return None

    def get_channel(self, channel_key: str, width_ratio: Optional[float], kappa: Optional[float]) -> Optional[int]:
        if self.m.model() == "Pure":
            return self.get_pair_prod_channels(channel_key)
        return self.get_channels_from_limit_condition(channel_key, width_ratio, kappa)

    # ------------------------------------------------------------------
    # Cross-section numerator selection
    # ------------------------------------------------------------------
    def numerator(self, index: int):
        def pure_vlq_decay():
            if self.to_W:
                return self.m.get_xs_pp_QQ() if "W" in self.process[index] else -1
            if self.to_Z:
                return self.m.get_xs_pp_QQ() if "Z" in self.process[index] else -1
            if self.to_H:
                return self.m.get_xs_pp_QQ() if "H" in self.process[index] else -1
            raise Exception("Error, final state must be indicated")

        width_ratio, kappa = self.get_ratio_and_universal_coupling()
        channel = self.get_channel(self.key[index], width_ratio, kappa)
        if channel is None:
            return -1

        if getattr(self, "VLB", False):
            if self.m.model() == "Singlet":
                if channel == 20:
                    return self.m.get_xs_pp_Bbq_tWbq(channel)
                if channel == 21:
                    return self.m.get_xs_pp_Btq_tWtq(channel)
                if channel in [15, 16, 17]:
                    return self.m.get_xs_pp_Bj_bHj_ts_channels(channel)
                if channel in [4, 5, 6, 7]:
                    return self.m.get_xs_pp_Bbq_bHbq(channel)
                if channel == 19:
                    return self.m.get_xs_pp_Bbq_bZbq(channel)
                if channel == 18:
                    return self.m.get_xs_pp_Btq_bZtq(channel)
                return self.m.get_xs_pp_QQ()
            if self.m.model() == "Doublet":
                if channel == 18:
                    return self.m.get_xs_pp_Bbq_tWbq(channel)
                if channel in [13, 14, 15]:
                    return self.m.get_xs_pp_Bj_bHj_ts_channels(channel)
                if channel in [3, 4, 5, 6]:
                    return self.m.get_xs_pp_Bbq_bHbq(channel)
                return self.m.get_xs_pp_QQ()
            return pure_vlq_decay()

        if getattr(self, "VLX", False):
            if channel == 14:
                return self.m.get_xs_pp_Xtq_twtq(channel)
            return self.m.get_xs_pp_QQ()

        if getattr(self, "VLY", False):
            if self.m.model() == "Doublet":
                if channel in [0, 4, 6, 7]:
                    return self.m.get_xs_pp_Ybq_Wbbq(channel)
                if channel == 5:
                    return -1
                return self.m.get_xs_pp_QQ()
            if self.m.model() == "Triplet":
                if channel == 0:
                    return self.m.get_xs_pp_Ybq_Wbbq(channel)
                return self.m.get_xs_pp_QQ()

        if self.m.model() == "Singlet":
            if channel in [32, 33, 34, 35, 43, 44, 45]:
                return self.m.get_xs_pp_Tj_Ztj_ts_channels(channel)
            if channel in [37, 38, 39, 40, 41, 42, 17, 46, 16, 18, 19, 24, 25]:
                return self.m.get_xs_pp_Tbq_tHbq(channel)
            if channel in [57, 31, 36, 14, 10, 11, 12, 13, 50, 51, 20, 21, 26, 27]:
                return self.m.get_xs_pp_Tbq_tZbq(channel)
            if channel in [58, 30, 59, 60]:
                return self.m.get_xs_pp_Tbq_Wbbq(channel)
            if channel == 15:
                return self.m.get_xs_pp_Tbq(channel)
            if channel in [52, 53, 22, 23, 28, 29]:
                return self.m.get_xs_pp_Tbq_tZ_plus_TH(channel)
            if channel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 47, 48, 49, 54, 55, 56]:
                return self.m.get_xs_pp_QQ()
            return -1

        if self.m.model() == "Doublet":
            if channel in [23, 24, 25, 26, 27, 28]:
                return self.m.get_xs_pp_Tj_Ztj_ts_channels(channel)
            if channel in [5, 29, 9, 10, 15, 16]:
                return self.m.get_xs_pp_Ttq_tHtq(channel)
            if channel in [21, 7, 11, 12, 17, 18]:
                return self.m.get_xs_pp_Ttq_tZtq(channel)
            if channel in [13, 14, 19, 20]:
                return self.m.get_xs_pp_Ttq_tZ_plus_TH(channel)
            if channel in [0, 1, 2, 3, 4, 6, 8, 22, 30, 31, 32]:
                return self.m.get_xs_pp_QQ()
            return -1

        if self.m.model() == "Pure":
            return pure_vlq_decay()
        return -1

    # ------------------------------------------------------------------
    # Cross-section denominator evaluation
    # ------------------------------------------------------------------
    def denominator(self, numerator_value, index, table_values):
        if numerator_value < 0:
            return -1

        if getattr(self, "VLB", False):
            if min(self.MB[index]) <= self.m.get_mB() <= max(self.MB[index]):
                return self._b_denominator(index, table_values)
            return -1
        if getattr(self, "VLX", False):
            if min(self.MX[index]) <= self.m.get_mX() <= max(self.MX[index]):
                return float(interp1d(self.MX[index], table_values[index], "linear")(self.m.get_mX()))
            return -1
        if getattr(self, "VLY", False):
            if min(self.MY[index]) <= self.m.get_mY() <= max(self.MY[index]):
                return float(interp1d(self.MY[index], table_values[index], "linear")(self.m.get_mY()))
            return -1
        return self.get_expt_xs_from_2d_interpolation(index, table_values)

    def _b_denominator(self, index, table_values):
        def width_interp(indices):
            width_values = [0.01, 0.1, 0.2, 0.3]
            expt_input = (
                indices,
                self.MB,
                width_values,
                None,
                table_values,
                self.m.get_mB(),
                self.m.get_width_mass_ratio(),
                None,
            )
            return interpolate_surface(mode=InterpolationMode.EXPERIMENTAL, experimental_input=expt_input)

        def kappa_interp(indices):
            kappa_values = [0.3, 0.4, 0.5]
            expt_input = (
                indices,
                self.MB,
                None,
                kappa_values,
                table_values,
                self.m.get_mB(),
                None,
                self.m.get_coupling_strength(),
            )
            return interpolate_surface(mode=InterpolationMode.EXPERIMENTAL, experimental_input=expt_input)

        if self.m.model() == "Singlet":
            if index in [4, 5, 6, 7]:
                return width_interp([4, 5, 6, 7])
            if index in [15, 16, 17]:
                return kappa_interp([15, 16, 17])
        elif self.m.model() == "Doublet":
            if index in [5, 6, 7, 8]:
                return width_interp([5, 6, 7, 8])
            if index in [13, 14, 15]:
                return kappa_interp([13, 14, 15])

        return float(interp1d(self.MB[index], table_values[index], "linear")(self.m.get_mB()))

    def expected_ratio_calc(self):
        maximum_ratio = float("-inf")
        position = -1
        for index, _ in enumerate(self.key):
            numerator_value = self.numerator(index)
            observed_limit = self.denominator(numerator_value, index, self.obs)
            if observed_limit == -1 or numerator_value == -1:
                continue
            observed_ratio = numerator_value / observed_limit
            if observed_ratio >= maximum_ratio:
                maximum_ratio = observed_ratio
                position = index
        return position

    def check_channel(self) -> Tuple[int, float, float, int]:
        position = self.expected_ratio_calc()
        predicted_xs = self.numerator(position) if position != -1 else -1
        observed_xs = self.denominator(predicted_xs, position, self.obs) if position != -1 else -1
        expected_xs = self.denominator(predicted_xs, position, self.exp) if position != -1 else -1

        if predicted_xs == -1 or observed_xs == -1:
            observed_ratio = float("-inf")
            expected_ratio = float("-inf")
        else:
            observed_ratio = predicted_xs / observed_xs
            expected_ratio = predicted_xs / expected_xs

        self.observed_ratio = observed_ratio
        self.expected_ratio = expected_ratio
        self._obs_ratio = observed_ratio
        self._exp_ratio = expected_ratio
        self.finalize(position)
        self.most_sensitive_channels.append(self._channel)
        return self._result, self._obs_ratio, self._exp_ratio, self._channel

    # ------------------------------------------------------------------
    # 2D interpolation helpers for T limits
    # ------------------------------------------------------------------
    def _t_kappa_interpolation_indices(self, key: str) -> Optional[tuple[list[int], list[float]]]:
        for condition in self._constraint_conditions(Observable.KAPPA):
            identifiers = self._constraint_identifiers(Observable.KAPPA, condition)
            if key not in identifiers:
                continue
            values, is_exact = parse_constraint_condition(condition)
            if is_exact or len(values) != 2:
                return None
            indices = [self.key.index(identifier) for identifier in identifiers if identifier in self.key]
            coupling_values = list(np.linspace(values[0], values[1], len(indices)))
            return indices, coupling_values
        return None

    def _t_width_interpolation_indices(self, key: str) -> Optional[tuple[list[int], list[float]]]:
        condition = self._find_condition_for_identifier(key, Observable.WIDTH_MASS_RATIO)
        if condition != "0.05<=r<=0.3":
            return None
        identifiers = self._constraint_identifiers(Observable.WIDTH_MASS_RATIO, condition)
        if key not in identifiers:
            return None
        local_position = identifiers.index(key)
        group_start = (local_position // 3) * 3
        group_identifiers = identifiers[group_start:group_start + 3]
        if not group_identifiers:
            return None
        first_group_index = self.key.index(group_identifiers[0])
        stitched_indices = [first_group_index - 1] + [self.key.index(identifier) for identifier in group_identifiers]
        return stitched_indices, [0.05, 0.1, 0.2, 0.3]

    def get_expt_xs_from_2d_interpolation(self, index, expt_table):
        if not (min(self.MT[index]) <= self.m.get_mT() <= max(self.MT[index])):
            return -1

        key = self.key[index]
        if self.m.model() in {"Singlet", "Doublet"}:
            kappa_bundle = self._t_kappa_interpolation_indices(key)
            if kappa_bundle is not None:
                t_indices, coupling_values = kappa_bundle
                expt_input = (
                    t_indices,
                    self.MT,
                    None,
                    coupling_values,
                    expt_table,
                    self.m.get_mT(),
                    None,
                    self.m.get_coupling_strength(),
                )
                return interpolate_surface(mode=InterpolationMode.EXPERIMENTAL, experimental_input=expt_input)

            width_bundle = self._t_width_interpolation_indices(key)
            if width_bundle is not None:
                t_indices, width_values = width_bundle
                expt_input = (
                    t_indices,
                    self.MT,
                    width_values,
                    None,
                    expt_table,
                    self.m.get_mT(),
                    self.m.get_width_mass_ratio(),
                    None,
                )
                return interpolate_surface(mode=InterpolationMode.EXPERIMENTAL, experimental_input=expt_input)

        return float(interp1d(self.MT[index], expt_table[index], "linear")(self.m.get_mT()))

    # ------------------------------------------------------------------
    # Result finalization
    # ------------------------------------------------------------------
    def finalize(self, position: int) -> None:
        self.dominant_channel = position
        self._channel = position

        if position == -1:
            self.result_flag = -1
            self._result = -1
            self._obs_ratio = -1
            self._exp_ratio = -1
            self.observed_ratio = -1
            self.expected_ratio = -1
            return

        if self._obs_ratio >= 1:
            result = 0
        elif self._obs_ratio < 0:
            result = -1
        else:
            result = 1

        self.result_flag = result
        self._result = result
        self.observed_ratio = self._obs_ratio
        self.expected_ratio = self._exp_ratio
