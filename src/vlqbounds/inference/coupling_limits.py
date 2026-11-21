from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional, Sequence

import json
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
from .cross_section_limits import CrossSectionLimitEngine
from .evaluation import (
    VLQSpecies,
    calculate_observed_expected_ratios,
    populate_coupling_limit_arrays,
    transform_contact_factor_to_kappa,
)


FloatArray = NDArray[np.float64]
VLQModel = TopPartnerSinglet | TopPartnerDoublet | TopPartnerPure | BottomPartnerSinglet | BottomPartnerDoublet | BottomPartnerPure | XPartnerDoublet | YPartnerDoublet | YPartnerTriplet


@dataclass(frozen=True, slots=True)
class CouplingLimitDescriptor:
    index: int
    key: str
    label: str
    experiment: str
    file_name: str
    process: str
    coupling_type: str
    energy_tev: int
    luminosity_fbinv: float
    # Multiplies the tabulated limits to express them in the package's
    # coupling normalisation (k_T, k_B = sqrt(2) sin(theta_L) for singlets).
    # Use sqrt(2) for experimental couplings that approximate sin(theta),
    # i.e. L ⊃ (g/sqrt2) kappa Q W q.
    limit_scale: float = 1.0

    def to_mapping(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "key": self.key,
            "label": self.label,
            "experiment": self.experiment,
            "file_name": self.file_name,
            "process": self.process,
            "coupling_type": self.coupling_type,
            "energy_tev": self.energy_tev,
            "luminosity_fbinv": self.luminosity_fbinv,
            "limit_scale": self.limit_scale,
        }


_RAW_COUPLING_LIMIT_REGISTRY: Final[dict[tuple[str, str], list[dict[str, Any]]]] = {('B', 'Singlet'): [{'index': 0,
                     'file_name': '2308.02595_ATLAS_Fig9a_pp_B_bH_k_singlet.json',
                     'key': '02595f9a',
                     'label': 'arXiv:2308.02595',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Bbq --> bH --> 0l',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 1,
                     'file_name': '2405.17605_Fig43_upper_pp_B_bH_singlet_1802.01486_cf.json',
                     'key': '01486f43',
                     'label': 'arXiv:1802.01486',
                     'experiment': 'CMS',
                     'process': 'pp --> Bqq --> bH --> b,bb,bq',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 2,
                     'file_name': '2405.17605_Fig43_upper_pp_Bbq_tW_singlet_1809.08597_cf.json',
                     'key': '01486f43b',
                     'label': 'arXiv:1809.08597',
                     'experiment': 'CMS',
                     'process': 'pp --> Bbq --> tW --> bqq,lnu/blnu,qq',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 3,
                     'file_name': '2405.17605_Fig43_upper_pp_Bbq_tW_singlet_2111.10216_cf.json',
                     'key': '10216f43',
                     'label': 'arXiv:2111.10216',
                     'experiment': 'CMS',
                     'process': 'pp --> Bbq --> tW --> bqq,lnu/qq',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 138},
                    {'index': 4,
                     'file_name': '2405.17605_Fig43_upper_pp_Btq_tW_singlet_1809.08597_cf.json',
                     'key': '01486f43t',
                     'label': 'arXiv:1809.08597',
                     'experiment': 'CMS',
                     'process': 'pp --> Btq --> tW --> bqq,lnu/blnu,qq',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 5,
                     'file_name': '2405.17605_Fig43_upper_pp_Btq_tW_singlet_2111.10216_cf.json',
                     'key': '10216f43t',
                     'label': 'arXiv:2111.10216',
                     'experiment': 'CMS',
                     'process': 'pp --> Btq --> tW --> bqq,lnu/qq',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 138},
                    {'index': 6,
                     'file_name': '2606.01423_CMS_Fig10a_pp_Bbq_wtbq_singlet.json',
                     'key': '01423f10a',
                     'label': 'arXiv:2606.01423',
                     'experiment': 'CMS',
                     'process': 'pp --> Bbq --> tW --> l',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 138,
                     'limit_scale': 2 ** 0.5},
                    {'index': 7,
                     'file_name': '2606.01423_CMS_Fig10b_pp_Btq_wttq_singlet.json',
                     'key': '01423f10b',
                     'label': 'arXiv:2606.01423',
                     'experiment': 'CMS',
                     'process': 'pp --> Btq --> tW --> l',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 138,
                     'limit_scale': 2 ** 0.5}],
 ('B', 'Doublet'): [{'index': 0,
                     'file_name': '2308.02595_ATLAS_Fig9b_pp_B_bH_k_doublet.json',
                     'key': '02595f9b',
                     'label': 'arXiv:2308.02595',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Bbq --> bH --> 0l',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 1,
                     'file_name': '2405.17605_Fig43_lower_pp_B_bH_doublet_1802.01486_cf.json',
                     'key': '01486f43l',
                     'label': 'arXiv:1802.01486',
                     'experiment': 'CMS',
                     'process': 'pp --> Bbq --> bH --> 0l',
                     'coupling_type': 'k_B',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9}],
 ('X', 'Doublet'): [{'index': 0,
                     'file_name': '2405.17605_CMS_Fig44_left_pp_tqX_tW_1809.08597.json',
                     'key': '08597Fig44l',
                     'label': 'arXiv:1809.08597',
                     'experiment': 'CMS',
                     'process': 'pp --> Xtq --> tW --> bqq,lnu/blnu,qq ',
                     'coupling_type': 'k_x',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 1,
                     'file_name': '2405.17605_CMS_Fig44_left_pp_tqX_tW_2111.10216.json',
                     'key': '10216Fig44l',
                     'label': 'arXiv:2111.10216',
                     'experiment': 'CMS',
                     'process': 'pp --> Xtq --> tW --> bqq,lnu/blnu,qq ',
                     'coupling_type': 'k_x',
                     'energy_tev': 13,
                     'luminosity_fbinv': 138},
                    {'index': 2,
                     'file_name': '1807.11883_ATLAS_Fig10b_pp_XX_Wt_coupling.json',
                     'key': '11883f10b',
                     'label': 'arXiv:1807.11883',
                     'experiment': 'ATLAS',
                     'process': 'pp --> XX(Xtq) --> tW --> l+l+ ',
                     'coupling_type': 'k_x',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1}],
 ('Y', 'Doublet'): [{'index': 0,
                     'file_name': '2405.17605_CMS_Fig44_right_pp_tqY_bW_1701.08328.json',
                     'key': '08328Fig44r',
                     'label': 'arXiv:1701.08328',
                     'experiment': 'CMS',
                     'process': 'pp --> Ytq --> bW --> bqq,lnu/blnu,qq ',
                     'coupling_type': 'k_y',
                     'energy_tev': 13,
                     'luminosity_fbinv': 2.3},
                    {'index': 1,
                     'file_name': '1602.05606_ATLAS_Fig8b_pp_Ybj_Wb_Doublet_sinR.json',
                     'key': '05606f8b',
                     'label': 'arXiv:1602.05606',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Ybq --> bW --> b,lnu ',
                     'coupling_type': 'k_y',
                     'energy_tev': 8,
                     'luminosity_fbinv': 20.3},
                    {'index': 2,
                     'file_name': 'ATLAS_CONF_2016_072_fig10b_doublet.json',
                     'key': '072f10b',
                     'label': 'ATLAS_CONF_2016_072',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Ybq --> bW --> b,lnu ',
                     'coupling_type': 'k_y',
                     'energy_tev': 13,
                     'luminosity_fbinv': 3.2},
                    {'index': 3,
                     'file_name': '1812.07343_ATLAS_pp_Ybq_Wbbq_Fig8c_doublet_Y_RH_sinR.json',
                     'key': '07343f8c',
                     'label': 'arXiv:1812.07343',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Ybq --> bW --> b,lnu ',
                     'coupling_type': 'k_y',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1},
                    {'index': 4,
                     'file_name': '2409.20273_ATLAS_Fig6_pp_Ybq_Wb_doublet_BY.json',
                     'key': '20273',
                     'label': 'arXiv:2409.20273',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Ybq --> bW --> b, qq',
                     'coupling_type': 'k_y',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1}],
 ('Y', 'Triplet'): [{'index': 0,
                     'file_name': '1812.07343_ATLAS_Fig8b_pp_Ybq_Wb_triplet_Y_LH_sinL.json',
                     'key': '07343f8b',
                     'label': 'ArXiv:1812.07343',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Ybq --> bW --> b,lnu',
                     'coupling_type': 's_d_l',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1}],
 ('T', 'Singlet'): [{'index': 0,
                     'file_name': '1602.05606_ATLAS_f7b_pp_Tbj_Wbbj_s_L_Singlet.json',
                     'key': '05606f7b',
                     'label': 'arXiv:1602.05606',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> Wb --> lnu,b ',
                     'coupling_type': 's_l',
                     'energy_tev': 8,
                     'luminosity_fbinv': 20.3},
                    {'index': 1,
                     'file_name': '1812.07343_ATLAS_f8a_pp_Tbq_wbbq_Singlet_s_L.json',
                     'key': '07343f8as_L',
                     'label': 'arXiv:1812.07343',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> Wb --> 1l ',
                     'coupling_type': 's_l',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1},
                    {'index': 2,
                     'file_name': '1812.09743_ATLAS_f6b_pp_Tbq_tZbq_s_L_Singlet.json',
                     'key': '09743f6b_s_L',
                     'label': 'arXiv:1812.09743',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> tZ --> 0l+1l ',
                     'coupling_type': 's_l',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1},
                    {'index': 3,
                     'file_name': '1806.10555_ATLAS_f16b_pp_Tbq_Zt_s_L_Singlet.json',
                     'key': '10555f16b_s_L',
                     'label': 'arXiv:1806.10555',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> tZ --> l+l-+>=3l',
                     'coupling_type': 's_l',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1},
                    {'index': 4,
                     'file_name': 'ATLAS-CONF-2016-072_ATLAS_f10a_pp_Tqb_Wb_singlet_s_L.json',
                     'key': '072f10b',
                     'label': 'ATLAS-CONF-2016-072',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> bW --> 1l',
                     'coupling_type': 's_l',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1},
                    {'index': 5,
                     'file_name': '2302.12802_CMS_Fig5_pp_Tbq_tH_cf.json',
                     'key': '12802f5',
                     'label': 'arXiv:2302.12802',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tH --> gamma gamma, 0l/1l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 36.1},
                    {'index': 6,
                     'file_name': '2402.16561_ATLAS_Fig12a_pp_T_Ht_Zt_k_T_singlet.json',
                     'key': '16561f12a_k_T',
                     'label': 'arXiv:2402.16561',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> tZ --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 7,
                     'file_name': '2201.07045_ATLAS_f9_pp_Tbq_Htbq_k_T_singlet.json',
                     'key': '07045f9_k_T',
                     'label': 'arXiv:2201.07045',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> tH --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 8,
                     'file_name': '2305.03401_ATLAS_f13a_pp_Tbq_Ztbq_singlet_k_T.json',
                     'key': '03401f13a',
                     'label': 'arXiv:2305.03401',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> tZ(H) --> 1l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 9,
                     'file_name': '2307.07584_ATLAS_f9a_pp_Tbq_Ztbq_k_T_singlet.json',
                     'key': '07584f9a',
                     'label': 'arXiv:2307.07584',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> tZ --> l+l-+3l ',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 10,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_Tbq_tZ_ll_singlet_1708.01062_cf.json',
                     'key': '01062f42u',
                     'label': 'arXiv:1708.01062',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> l+l-',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 11,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_T_bW_singlet_1701.08328_cf.json',
                     'key': '08328f42u',
                     'label': 'arXiv:1701.08328',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> 1l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 2.3},
                    {'index': 12,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_T_combination_singlet_cf.json',
                     'key': '17605f42u',
                     'label': 'arXiv:2405.17605',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 138},
                    {'index': 13,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_Ttq_tZ_tH_singlet_part1_1909.04721_cf.json',
                     'key': '04721f42up1',
                     'label': 'arXiv:1909.04721',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 14,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_Ttq_tZ_tH_part2_singlet_1909.04721_cf.json',
                     'key': '04721f42up2',
                     'label': 'arXiv:1909.04721',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 15,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_Ttq_tZ_singlet_1708.01062_cf.json',
                     'key': '01062f42utq',
                     'label': 'arXiv:1708.01062',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> l+l-',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 16,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_T_tZ_singlet_2201.02227_cf.json',
                     'key': '02227f42u',
                     'label': 'arXiv:2201.02227',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 137},
                    {'index': 17,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_T_tZ_tH_singlet_2405.05071_cf.json',
                     'key': '05071f42u',
                     'label': 'arXiv:2405.05071',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 138},
                    {'index': 18,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_Tbq_tZ_tH_singlet_part1_1909.04721_cf.json',
                     'key': '04721f42ubq1',
                     'label': 'arXiv:1909.04721',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> (tZ + tH) --> bqq, bb',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 19,
                     'file_name': '2405.17605_CMS_Fig42_upper_pp_Tbq_tZ_tH_singlet_part2_1909.04721_cf.json',
                     'key': '04721f42ubq2',
                     'label': 'arXiv:1909.04721',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> (tZ + tH) --> bqq, bb',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 20,
                     'file_name': '2408.08789_ATLAS_Fig6a_pp_Tbq_Ht_Zt_singlet.json',
                     'key': '08789f6a',
                     'label': 'arXiv:2408.08789',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tb(t)q --> tZ/H --> 0l+1l+>=2l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 21,
                     'file_name': '2506.15515_ATLAS_Fig6a_pp_Tbq_Wb_singlet.json',
                     'key': '15515f6a',
                     'label': 'arXiv:2506.15515',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> Wbbq --> 1l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 140,
                     'limit_scale': 2 ** 0.5},
                    {'index': 22,
                     'file_name': '2604.17564_CMS_Fig4b_pp_VLQ_Wb_Tsinglet.json',
                     'key': '17564f4b',
                     'label': 'arXiv:2604.17564',
                     'experiment': 'CMS',
                     'process': 'pp --> T --> Wb --> 1l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 138}],
 ('T', 'Doublet'): [{'index': 0,
                     'file_name': '2305.03401_ATLAS_f13b_pp_Tbq_Ztbq_k_T_doublet.json',
                     'key': '03401f13b',
                     'label': 'arXiv:2305.03401',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> tZbq --> 1l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 1,
                     'file_name': '2307.07584_ATLAS_f9b_pp_Tbq_Ztbq_k_T_doublet.json',
                     'key': '07584f9b',
                     'label': 'arXiv:2307.07584',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tbq --> tZbq --> l+l-+3l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139},
                    {'index': 2,
                     'file_name': '2405.17605_CMS_Fig42_lower_pp_T_tZ_doublet_1708.01062_cf.json',
                     'key': '01062f42l',
                     'label': 'arXiv:1708.01062',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> l+l-',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 3,
                     'file_name': '2405.17605_CMS_Fig42_lower_pp_T_tZ_tH_doublet_part1_1909.04721_cf.json',
                     'key': '04721f42lp1',
                     'label': 'arXiv:1909.04721',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 4,
                     'file_name': '2405.17605_Fig42_lower_pp_T_tZ_tH_doublet_part2_cf_1909.04721.json',
                     'key': '04721f42lp2',
                     'label': 'arXiv:1909.04721',
                     'experiment': 'CMS',
                     'process': 'pp --> Tbq --> tZbq --> 0l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 35.9},
                    {'index': 5,
                     'file_name': '2408.08789_ATLAS_Fig6b_pp_Tbq_Ht_Zt_doublet.json',
                     'key': '08789f6b',
                     'label': 'arXiv:2408.08789',
                     'experiment': 'ATLAS',
                     'process': 'pp --> Tb(t)q --> tZ/H --> 1l + 2l + >=3l',
                     'coupling_type': 'k_T',
                     'energy_tev': 13,
                     'luminosity_fbinv': 139}]}


def _materialize_registry() -> dict[tuple[str, str], tuple[CouplingLimitDescriptor, ...]]:
    registry: dict[tuple[str, str], tuple[CouplingLimitDescriptor, ...]] = {}
    for key, records in _RAW_COUPLING_LIMIT_REGISTRY.items():
        registry[key] = tuple(CouplingLimitDescriptor(**record) for record in records)
    return registry


COUPLING_LIMIT_REGISTRY: Final[dict[tuple[str, str], tuple[CouplingLimitDescriptor, ...]]] = _materialize_registry()


class CouplingLimitEngine(CrossSectionLimitEngine):
    """
    Coupling-limit evaluation engine.

    The engine loads coupling-limit metadata, interpolates the published
    envelopes, compares model predictions against observed and expected bounds,
    and writes JSON manifests for downstream reporting.
    """

    def __init__(self, model: VLQModel):
        super().__init__(model)
        if not isinstance(
            model,
            (TopPartnerSinglet, TopPartnerDoublet, TopPartnerPure, BottomPartnerSinglet, BottomPartnerDoublet, BottomPartnerPure, XPartnerDoublet, YPartnerDoublet, YPartnerTriplet),
        ):
            raise TypeError("Invalid VLQ model instance provided to CouplingLimitEngine.")

        self.m: VLQModel = model

        self.coupling_luminosity: Optional[list[float]] = None
        self.coupling_energy: Optional[list[int]] = None
        self.which_coupling: Optional[list[str]] = None
        self.coupling_process: Optional[list[str]] = None

        self.coupling_exp_lower: Optional[list[Optional[FloatArray]]] = None
        self.coupling_obs_lower: Optional[list[Optional[FloatArray]]] = None
        self.coupling_exp_upper: Optional[list[Optional[FloatArray]]] = None
        self.coupling_obs_upper: Optional[list[Optional[FloatArray]]] = None

        self.coupling_MY_exp: Optional[list[Optional[FloatArray]]] = None
        self.coupling_MY_obs: Optional[list[Optional[FloatArray]]] = None
        self.coupling_MX_exp: Optional[list[Optional[FloatArray]]] = None
        self.coupling_MX_obs: Optional[list[Optional[FloatArray]]] = None
        self.coupling_MB_exp: Optional[list[Optional[FloatArray]]] = None
        self.coupling_MB_obs: Optional[list[Optional[FloatArray]]] = None
        self.coupling_MT_exp: Optional[list[Optional[FloatArray]]] = None
        self.coupling_MT_obs: Optional[list[Optional[FloatArray]]] = None

        self.coupling_expt: Optional[list[str]] = None
        self.coupling_label: Optional[list[str]] = None
        self.coupling_key: Optional[list[str]] = None
        self.coupling_file_name: Optional[list[str]] = None

        self.sin_l_keys: list[str] = []
        self.sin_r_keys: list[str] = []
        self.k_keys: list[str] = []
        self.mass_ratio: Optional[float] = None

        self._active_coupling_descriptors: list[CouplingLimitDescriptor] = []
        self.initialize_coupling_data()

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------
    def _active_species(self) -> str:
        if self.VLB:
            return "B"
        if self.VLX:
            return "X"
        if self.VLY:
            return "Y"
        return "T"

    def _active_scenario(self) -> str:
        return str(self.m.model())

    @classmethod
    def _registry_for(cls, species: str, scenario: str) -> tuple[CouplingLimitDescriptor, ...]:
        return COUPLING_LIMIT_REGISTRY.get((species, scenario), tuple())

    def _active_registry(self) -> tuple[CouplingLimitDescriptor, ...]:
        return self._registry_for(self._active_species(), self._active_scenario())

    # ------------------------------------------------------------------
    # Storage allocation
    # ------------------------------------------------------------------
    def initialize_coupling_lists(self, number_of_tables: int) -> None:
        self.coupling_file_name = [None] * number_of_tables
        self.coupling_key = [None] * number_of_tables
        self.coupling_label = [None] * number_of_tables
        self.coupling_expt = [None] * number_of_tables
        self.coupling_process = [None] * number_of_tables
        self.which_coupling = [None] * number_of_tables
        self.coupling_energy = [None] * number_of_tables
        self.coupling_luminosity = [None] * number_of_tables

        self.coupling_obs_upper = [None] * number_of_tables
        self.coupling_exp_upper = [None] * number_of_tables
        self.coupling_obs_lower = [None] * number_of_tables
        self.coupling_exp_lower = [None] * number_of_tables

        self.coupling_MT_obs = None
        self.coupling_MT_exp = None
        self.coupling_MB_obs = None
        self.coupling_MB_exp = None
        self.coupling_MX_obs = None
        self.coupling_MX_exp = None
        self.coupling_MY_obs = None
        self.coupling_MY_exp = None

        if isinstance(self.m, (TopPartnerSinglet, TopPartnerDoublet)):
            self.coupling_MT_obs = [None] * number_of_tables
            self.coupling_MT_exp = [None] * number_of_tables
        elif isinstance(self.m, (BottomPartnerSinglet, BottomPartnerDoublet)):
            self.coupling_MB_obs = [None] * number_of_tables
            self.coupling_MB_exp = [None] * number_of_tables
        elif isinstance(self.m, XPartnerDoublet):
            self.coupling_MX_obs = [None] * number_of_tables
            self.coupling_MX_exp = [None] * number_of_tables
        elif isinstance(self.m, (YPartnerDoublet, YPartnerTriplet)):
            self.coupling_MY_obs = [None] * number_of_tables
            self.coupling_MY_exp = [None] * number_of_tables

    def get_number_of_tables(self, vlq: str = "T") -> int:
        return len(self._registry_for(vlq.upper(), self._active_scenario()))

    def initialize_coupling_data(self) -> None:
        self.initialize_coupling_lists(len(self._active_registry()))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _mass_payload(self) -> list[Optional[list[Optional[FloatArray]]]]:
        species = self._active_species()
        if species == "T":
            return [self.coupling_MT_obs, self.coupling_MT_exp]
        if species == "B":
            return [self.coupling_MB_obs, self.coupling_MB_exp]
        if species == "X":
            return [self.coupling_MX_obs, self.coupling_MX_exp]
        if species == "Y":
            return [self.coupling_MY_obs, self.coupling_MY_exp]
        raise RuntimeError(f"Unsupported VLQ species for coupling-mass payload: {species}")

    def fill_coupling_tables(self) -> None:
        descriptors = list(self._active_registry())
        self._active_coupling_descriptors = descriptors
        if not descriptors:
            return

        if self.coupling_key is None or len(self.coupling_key) != len(descriptors):
            self.initialize_coupling_lists(len(descriptors))

        for target_index, descriptor in enumerate(descriptors):
            self.coupling_key[target_index] = descriptor.key
            self.coupling_label[target_index] = descriptor.label
            self.coupling_expt[target_index] = descriptor.experiment
            self.coupling_file_name[target_index] = descriptor.file_name
            self.coupling_process[target_index] = descriptor.process
            self.which_coupling[target_index] = descriptor.coupling_type
            self.coupling_energy[target_index] = descriptor.energy_tev
            self.coupling_luminosity[target_index] = descriptor.luminosity_fbinv

        expected_arrays = [self.coupling_exp_upper, self.coupling_exp_lower]
        observed_arrays = [self.coupling_obs_lower, self.coupling_obs_upper]
        mass_arrays = self._mass_payload()

        populate_coupling_limit_arrays(
            resource_names=self.coupling_file_name,
            file_count=len(self.coupling_key),
            mass_arrays=mass_arrays,
            expected_arrays=expected_arrays,
            observed_arrays=observed_arrays,
            experiment_labels=self.coupling_expt,
            species=VLQSpecies(self._active_species()),
        )

        if self._active_species() in {"B", "T"}:
            self.coupling_factor_to_coupling_strength(self._active_species())

        self._apply_limit_scales(descriptors)

    def _apply_limit_scales(self, descriptors: Sequence[CouplingLimitDescriptor]) -> None:
        for index, descriptor in enumerate(descriptors):
            scale = descriptor.limit_scale
            if scale == 1.0:
                continue
            for arrays in (
                self.coupling_obs_lower,
                self.coupling_obs_upper,
                self.coupling_exp_lower,
                self.coupling_exp_upper,
            ):
                if arrays is not None and arrays[index] is not None:
                    arrays[index] = np.asarray(arrays[index], dtype=np.float64) * scale

    # ------------------------------------------------------------------
    # Coupling classification
    # ------------------------------------------------------------------
    def fill_sin_and_kappa(self) -> None:
        coupling_pairs = list(zip(self.coupling_key or [], self.which_coupling or []))
        self.k_keys = [key for key, coupling in coupling_pairs if coupling and coupling.lower().startswith("k_")]
        self.sin_l_keys = [key for key, coupling in coupling_pairs if coupling in {"s_l", "s_d_l"}]
        self.sin_r_keys = [key for key, coupling in coupling_pairs if coupling in {"s_r", "s_u_r", "s_d_r"}]

    def model_coupling_calc(self, index: int) -> float:
        if index == -1 or self.which_coupling is None or self.coupling_key is None:
            return -1

        coupling_label = self.which_coupling[index]
        if coupling_label is None:
            raise RuntimeError(f"Missing coupling type for channel index {index}.")

        species = self._active_species()
        scenario = self._active_scenario()

        if species == "B":
            if scenario == "Singlet":
                if coupling_label == "k_B":
                    return self.m.get_coupling_strength()
                if coupling_label == "s_l":
                    return self.m.get_sin_left()
            elif scenario == "Doublet" and coupling_label == "k_B":
                return self.m.get_coupling_strength()

        elif species == "X":
            if coupling_label.lower() == "k_x":
                return self.m.get_coupling_strength()

        elif species == "Y":
            if scenario == "Doublet" and coupling_label.lower() == "k_y":
                return self.m.get_coupling_strength()
            if scenario == "Triplet" and coupling_label == "s_d_l":
                return self.m.get_sin_down_left()

        elif species == "T":
            if scenario == "Singlet":
                if coupling_label == "s_l":
                    return self.m.get_sin_left()
                if coupling_label == "k_T":
                    return self.m.get_coupling_strength()
            elif scenario == "Doublet" and coupling_label == "k_T":
                return self.m.get_coupling_strength()

        raise RuntimeError(
            f"Unable to evaluate the model coupling for index {index} "
            f"(species={species}, scenario={scenario}, coupling_type={coupling_label})."
        )

    # ------------------------------------------------------------------
    # Interpolation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _interpolate_limit(mass_grid: Sequence[float], value_grid: Sequence[float], target_mass: float) -> float:
        x = np.asarray(mass_grid, dtype=float)
        y = np.asarray(value_grid, dtype=float)
        return float(np.interp(target_mass, x, y))

    def _model_mass(self) -> float:
        species = self._active_species()
        if species == "T":
            return float(self.m.get_mT())
        if species == "B":
            return float(self.m.get_mB())
        if species == "X":
            return float(self.m.get_mX())
        if species == "Y":
            return float(self.m.get_mY())
        raise RuntimeError(f"Unsupported VLQ species for mass lookup: {species}")

    def _channel_mass_arrays(self) -> tuple[Optional[list[Optional[FloatArray]]], Optional[list[Optional[FloatArray]]]]:
        species = self._active_species()
        if species == "T":
            return self.coupling_MT_obs, self.coupling_MT_exp
        if species == "B":
            return self.coupling_MB_obs, self.coupling_MB_exp
        if species == "X":
            return self.coupling_MX_obs, self.coupling_MX_exp
        if species == "Y":
            return self.coupling_MY_obs, self.coupling_MY_exp
        raise RuntimeError(f"Unsupported VLQ species for channel-mass lookup: {species}")

    def get_limit_from_data(
        self,
        model_value: float,
        index: int,
        table_values: Optional[list[Optional[FloatArray]]],
        mass_values: Optional[list[Optional[FloatArray]]],
    ) -> float:
        if model_value < 0 or table_values is None or mass_values is None:
            return -1
        if index >= len(mass_values) or mass_values[index] is None or table_values[index] is None:
            return -1

        target_mass = self._model_mass()
        channel_masses = np.asarray(mass_values[index], dtype=float)
        channel_limits = np.asarray(table_values[index], dtype=float)

        if target_mass < float(np.min(channel_masses)) or target_mass > float(np.max(channel_masses)):
            return -1

        return self._interpolate_limit(channel_masses, channel_limits, target_mass)

    def identify_strong_limit(
        self,
        value_arrays: Optional[list[Optional[FloatArray]]],
        mass_arrays: Optional[list[Optional[FloatArray]]],
    ) -> int:
        strongest_ratio = float("-inf")
        strongest_position = -1

        for index, key in enumerate(self.coupling_key or []):
            if key is None:
                continue
            model_value = self.model_coupling_calc(index)
            experimental_limit = self.get_limit_from_data(model_value, index, value_arrays, mass_arrays)
            if experimental_limit in (-1, 0) or model_value == -1:
                continue
            ratio = model_value / experimental_limit
            if ratio > strongest_ratio:
                strongest_ratio = ratio
                strongest_position = index

        return strongest_position

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

    def _synchronize_public_result_state(self) -> None:
        self.result_flag = self._result
        self.dominant_channel = self._channel
        self.observed_ratio = self._obs_ratio
        self.expected_ratio = self._exp_ratio

    def result_based_on_branches(self, position: int, reference_position: int, coupling_value: float, observed_lower: float) -> None:
        if (
            self.coupling_obs_upper is None
            or self.coupling_exp_upper is None
            or self.coupling_MT_obs is None
            or self.coupling_MT_exp is None
        ):
            self.finalize(position)
            return

        observed_upper = self.get_limit_from_data(coupling_value, position, self.coupling_obs_upper, self.coupling_MT_obs)

        if position == 3:
            secondary_observed_upper = self.get_limit_from_data(coupling_value, position - 1, self.coupling_obs_upper, self.coupling_MT_obs)
            secondary_expected_upper = self.get_limit_from_data(coupling_value, reference_position - 1, self.coupling_exp_upper, self.coupling_MT_exp)
        elif position == 2:
            secondary_observed_upper = self.get_limit_from_data(coupling_value, position, self.coupling_obs_upper, self.coupling_MT_obs)
            secondary_expected_upper = self.get_limit_from_data(coupling_value, reference_position, self.coupling_exp_upper, self.coupling_MT_exp)
        else:
            secondary_observed_upper, secondary_expected_upper = None, None

        if position in [2, 3]:
            if observed_lower <= coupling_value <= observed_upper:
                self._obs_ratio = coupling_value / observed_lower
                self._result = 0
                self._channel = position
            elif observed_upper <= coupling_value <= secondary_observed_upper:
                self._obs_ratio = coupling_value / observed_upper
                self._result = 0
                self._channel = position
            elif coupling_value > secondary_observed_upper and coupling_value > observed_upper:
                channel0_lower = self.get_limit_from_data(coupling_value, 0, self.coupling_obs_lower, self.coupling_MT_obs)
                channel1_lower = self.get_limit_from_data(coupling_value, 1, self.coupling_obs_lower, self.coupling_MT_obs)
                channel4_lower = self.get_limit_from_data(coupling_value, 4, self.coupling_obs_lower, self.coupling_MT_obs)

                if min(self.coupling_MT_obs[0]) <= self.m.get_mT() <= max(self.coupling_MT_obs[0]):
                    self._obs_ratio = coupling_value / channel0_lower
                    self._result = 0
                    self._channel = 0
                elif min(self.coupling_MT_obs[1]) <= self.m.get_mT() <= max(self.coupling_MT_obs[1]):
                    self._obs_ratio = coupling_value / channel1_lower
                    self._result = 0
                    self._channel = 1
                elif min(self.coupling_MT_obs[4]) <= self.m.get_mT() <= max(self.coupling_MT_obs[4]):
                    self._obs_ratio = coupling_value / channel4_lower
                    self._result = 0
                    self._channel = 4
                else:
                    self._obs_ratio = max(secondary_observed_upper, observed_upper) / coupling_value
                    self._result = 1
                    self._channel = position
            else:
                self._result = 1
                self._channel = position

            if secondary_expected_upper is not None and coupling_value >= secondary_expected_upper:
                self._exp_ratio = secondary_expected_upper / coupling_value
            self._synchronize_public_result_state()
        else:
            self.finalize(position)

    # ------------------------------------------------------------------
    # Coupling-limit evaluation
    # ------------------------------------------------------------------
    def coupling_limit(self):
        observed_mass_arrays, expected_mass_arrays = self._channel_mass_arrays()
        position = self.identify_strong_limit(self.coupling_obs_lower, observed_mass_arrays)

        if position == -1:
            self._result, self._obs_ratio, self._exp_ratio, self._channel = -1, -1, -1, -1
            self._synchronize_public_result_state()
            return self._result, self._obs_ratio, self._exp_ratio, self._channel

        model_value = self.model_coupling_calc(position)
        observed_limit = self.get_limit_from_data(model_value, position, self.coupling_obs_lower, observed_mass_arrays)
        expected_limit = self.get_limit_from_data(model_value, position, self.coupling_exp_lower, expected_mass_arrays)

        self._obs_ratio, self._exp_ratio = calculate_observed_expected_ratios(
            model_value,
            observed_limit,
            expected_limit,
        )

        if self._active_species() == "T" and self._active_scenario() == "Singlet":
            self.result_based_on_branches(position, position, model_value, observed_limit)
        else:
            self.finalize(position)

        return self._result, self._obs_ratio, self._exp_ratio, self._channel

    def check_xs_and_coupling_limits(self) -> None:
        coupling_result, coupling_observed_ratio, coupling_expected_ratio, coupling_channel = self.coupling_limit()
        xs_result, xs_observed_ratio, xs_expected_ratio, xs_channel = self.check_channel()

        xs_invalid = xs_channel == -1 or xs_observed_ratio is None or xs_observed_ratio < 0
        coupling_invalid = coupling_channel == -1 or coupling_observed_ratio is None or coupling_observed_ratio < 0

        if xs_invalid and coupling_invalid:
            self._result, self._obs_ratio, self._exp_ratio, self._channel = -1, -1, -1, -1
            self._synchronize_public_result_state()
            return

        if coupling_invalid or (not xs_invalid and xs_observed_ratio >= coupling_observed_ratio):
            self._result = xs_result
            self._obs_ratio = xs_observed_ratio
            self._exp_ratio = xs_expected_ratio
            self._channel = xs_channel
            self._synchronize_public_result_state()
        else:
            self._result = coupling_result
            self._obs_ratio = coupling_observed_ratio
            self._exp_ratio = coupling_expected_ratio
            key_count = len(self.key or [])
            self._channel = coupling_channel + key_count
            self._synchronize_public_result_state()
            self.most_sensitive_channels.append(self._channel)

    # ------------------------------------------------------------------
    # JSON report output
    # ------------------------------------------------------------------
    @staticmethod
    def _package_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _report_output_directory(self) -> Path:
        output_dir = self._package_root() / "reports" / "manifests"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _manifest_metadata(self) -> dict[str, Any]:
        species = self._active_species()
        return {
            "generated_by": "VLQBounds",
            "schema": "vlqbounds.coupling-manifest.v1",
            "model_class": type(self.m).__name__,
            "scenario": self.m.model(),
            "species": species,
        }

    def _coupling_manifest_records(self, channel_offset: int = 0) -> list[dict[str, Any]]:
        return [
            {
                "channel": channel_offset + index,
                "dataset_key": descriptor.key,
                "label": descriptor.label,
                "experiment": descriptor.experiment,
                "process": descriptor.process,
                "energy_tev": descriptor.energy_tev,
                "luminosity_fbinv": descriptor.luminosity_fbinv,
                "file_name": descriptor.file_name,
                "coupling_type": descriptor.coupling_type,
                "limit_type": "coupling",
            }
            for index, descriptor in enumerate(self._active_coupling_descriptors)
        ]

    def _cross_section_manifest_records(self, channel_offset: int = 0) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, process in enumerate(self.process or []):
            if process is None:
                continue
            records.append(
                {
                    "channel": channel_offset + index,
                    "dataset_key": self.key[index],
                    "label": self.label[index],
                    "experiment": self.expt[index],
                    "process": process,
                    "energy_tev": self.energy[index],
                    "luminosity_fbinv": self.luminosity[index],
                    "file_name": self.file_name[index],
                    "coupling_type": None,
                    "limit_type": "cross_section",
                }
            )
        return records

    def _write_json_manifest(
        self,
        stem: str,
        records: list[dict[str, Any]],
        *,
        extra: Optional[dict[str, Any]] = None,
    ) -> Path:
        payload: dict[str, Any] = self._manifest_metadata()
        payload["record_count"] = len(records)
        payload["records"] = records
        if extra:
            payload["extra"] = dict(extra)

        output_path = self._report_output_directory() / f"{stem}.json"
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        return output_path

    def all_couplings(self) -> Path:
        return self._write_json_manifest(
            "coupling_limits",
            self._coupling_manifest_records(channel_offset=0),
        )

    def coupling_and_xs_info(self) -> Path:
        xs_records = self._cross_section_manifest_records(channel_offset=0)
        coupling_records = self._coupling_manifest_records(channel_offset=len(self.key or []))
        return self._write_json_manifest(
            "combined_limits",
            xs_records + coupling_records,
            extra={"ordering": "cross_section_limits_first_then_coupling_limits"},
        )

    def get_sensitive_limits_info(self) -> Path:
        key_count = len(self.key or [])
        xs_records = self._cross_section_manifest_records(channel_offset=0)
        coupling_records = self._coupling_manifest_records(channel_offset=key_count)

        selected_records: list[dict[str, Any]] = []
        for raw_channel in np.unique(self.most_sensitive_channels):
            channel = int(raw_channel)
            if channel == -1:
                continue
            if channel >= key_count:
                coupling_index = channel - key_count
                if 0 <= coupling_index < len(coupling_records):
                    selected_records.append(coupling_records[coupling_index])
            else:
                if 0 <= channel < len(xs_records):
                    selected_records.append(xs_records[channel])

        return self._write_json_manifest("sensitive_limits", selected_records)

    # ------------------------------------------------------------------
    # Contact-factor conversion
    # ------------------------------------------------------------------
    def coupling_factor_to_coupling_strength(self, species: str = "T") -> None:
        if self.coupling_file_name is None:
            return

        for index, file_name in enumerate(self.coupling_file_name):
            if not file_name or "cf" not in file_name:
                continue

            if species == "T" and self.coupling_MT_obs is not None and self.coupling_MT_exp is not None:
                self.coupling_obs_lower[index] = transform_contact_factor_to_kappa(
                    self.coupling_obs_lower[index],
                    self.coupling_MT_obs[index],
                    self.m.model(),
                )
                self.coupling_exp_lower[index] = transform_contact_factor_to_kappa(
                    self.coupling_exp_lower[index],
                    self.coupling_MT_exp[index],
                    self.m.model(),
                )

            elif species == "B" and self.coupling_MB_obs is not None and self.coupling_MB_exp is not None:
                self.coupling_obs_lower[index] = transform_contact_factor_to_kappa(
                    self.coupling_obs_lower[index],
                    self.coupling_MB_obs[index],
                    self.m.model(),
                )
                self.coupling_exp_lower[index] = transform_contact_factor_to_kappa(
                    self.coupling_exp_lower[index],
                    self.coupling_MB_exp[index],
                    self.m.model(),
                )


__all__ = ["CouplingLimitEngine", "CouplingLimitDescriptor", "COUPLING_LIMIT_REGISTRY"]
