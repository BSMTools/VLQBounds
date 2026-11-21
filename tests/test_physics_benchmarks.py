"""
Physics regression tests: VLQBounds must reproduce numbers quoted in the
experimental papers and in the VLQBounds/VLQ-review publications.

Package convention: kappa_T = kappa_B = sqrt(2) sin(theta_L) for singlets.
"""
from __future__ import annotations

import contextlib
import io
import math
import warnings

import numpy as np
import pytest

from vlqbounds.constraints.experimental_repository import JsonLimitCurveReader
from vlqbounds.inference import VLQBounds
from vlqbounds.inference.coupling_limits import COUPLING_LIMIT_REGISTRY
from vlqbounds.models import BottomPartnerSinglet, TopPartnerSinglet

warnings.filterwarnings("ignore", category=FutureWarning)
SQ2 = math.sqrt(2.0)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def t_singlet():
    b = _quiet(VLQBounds, TopPartnerSinglet())
    _quiet(b.prepare_analysis)
    return b


@pytest.fixture(scope="module")
def b_singlet():
    b = _quiet(VLQBounds, BottomPartnerSinglet())
    _quiet(b.prepare_analysis)
    return b


def _coupling_limit(bounds, key, species):
    i = list(bounds.coupling_key).index(key)
    masses = bounds.coupling_MT_obs if species == "T" else bounds.coupling_MB_obs
    return (np.asarray(masses[i], float), np.asarray(bounds.coupling_obs_lower[i], float),
            np.asarray(bounds.coupling_exp_lower[i], float))


def _crossings(x, y, threshold):
    d = np.asarray(y) - np.asarray(threshold)
    return [x[j] + (x[j + 1] - x[j]) * (-d[j]) / (d[j + 1] - d[j])
            for j in range(len(d) - 1) if d[j] * d[j + 1] < 0]


def _kappa_at_width(cls, setter, mass, ratio):
    model = cls()
    getattr(model, setter)(float(mass))
    model.set_width_mass_ratio(ratio)
    return model.get_coupling_strength(), model.get_sin_left()


# ----------------------------------------------------------------------------- data I/O
def test_reader_column_order():
    """Tables are stored as [mass, observed, expected, ...]."""
    curve = JsonLimitCurveReader(()).parse({"rows": [[1000.0, 1.0, 2.0], [1100.0, 3.0, 4.0]]})
    np.testing.assert_allclose(curve.observed, [1.0, 3.0])
    np.testing.assert_allclose(curve.expected, [2.0, 4.0])


def test_limit_scales():
    scales = {d.key: d.limit_scale for ds in COUPLING_LIMIT_REGISTRY.values() for d in ds}
    assert scales["01423f10a"] == pytest.approx(SQ2)
    assert scales["01423f10b"] == pytest.approx(SQ2)
    assert scales["15515f6a"] == pytest.approx(SQ2)
    assert scales["17564f4b"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- pair production
def test_t_singlet_pair_production_mass_limit(t_singlet):
    """Singlet T excluded up to 1.49 TeV by pair production (arXiv:2412.01761)."""
    masses = np.arange(1380.0, 1560.0, 5.0)
    obs, exp = [], []
    for m in masses:
        _quiet(t_singlet.singletT_params, mT=float(m), s_l=1e-3)
        _quiet(t_singlet.check_against_xs_limits)
        row = t_singlet.df.iloc[-1]
        obs.append(row["obs_ratio"]); exp.append(row["exp_ratio"])
    m_obs = masses[np.argmax(np.asarray(obs) < 1.0)]
    m_exp = masses[np.argmax(np.asarray(exp) < 1.0)]
    assert 1470.0 <= m_obs <= 1500.0
    assert m_exp < m_obs  # CMS 2209.07327: observed limit stronger than expected


# ----------------------------------------------------------------------------- CMS 2606.01423
@pytest.mark.parametrize("mass, kappa_cms", [(800, 0.36), (1200, 0.23), (1600, 0.17), (2000, 0.14)])
def test_cms_2606_01423_convention(mass, kappa_cms):
    """Table 1 of arXiv:2606.01423: kappa ~ sin(theta) at Gamma/m = 5%."""
    kappa, sin_l = _kappa_at_width(BottomPartnerSinglet, "set_mB", mass, 0.05)
    assert sin_l == pytest.approx(kappa_cms, rel=0.04)
    assert kappa == pytest.approx(SQ2 * sin_l, rel=1e-6)


def test_cms_2606_01423_mass_limit(b_singlet):
    """Sec. 8: singlet B' with Gamma/m = 5% excluded below 1.23 TeV (1.35 TeV expected)."""
    m, obs, exp = _coupling_limit(b_singlet, "01423f10a", "B")
    fine = np.linspace(m[0], m[-1], 601)
    k5 = [_kappa_at_width(BottomPartnerSinglet, "set_mB", x, 0.05)[0] for x in fine]
    x_obs = _crossings(fine, np.interp(fine, m, obs), k5)
    x_exp = _crossings(fine, np.interp(fine, m, exp), k5)
    assert len(x_obs) == 1 and x_obs[0] == pytest.approx(1230.0, abs=25.0)
    assert len(x_exp) == 1 and x_exp[0] == pytest.approx(1350.0, abs=25.0)


# ----------------------------------------------------------------------------- CMS 2604.17564
def test_cms_2604_17564_t_singlet_range(t_singlet):
    """Sec. 7: singlet T excluded for 0.82-2.15 TeV at kappa_W^T = 0.2."""
    m, obs, _ = _coupling_limit(t_singlet, "17564f4b", "T")
    fine = np.linspace(m[0], m[-1], 1701)
    x = _crossings(fine, np.interp(fine, m, obs), 0.2)
    assert len(x) == 2
    assert x[0] == pytest.approx(820.0, abs=15.0)
    assert x[1] == pytest.approx(2150.0, abs=25.0)


def test_cms_2604_17564_minimum(t_singlet):
    """Abstract: kappa_W as low as 0.086 near 1.4 TeV for B(Wb)=1, i.e. 0.086*sqrt(2) for T."""
    m, obs, _ = _coupling_limit(t_singlet, "17564f4b", "T")
    assert obs.min() == pytest.approx(0.086 * SQ2, abs=0.005)
    assert m[np.argmin(obs)] == pytest.approx(1400.0, abs=100.0)


# ----------------------------------------------------------------------------- ATLAS 2506.15515
def test_atlas_2506_15515_endpoints(t_singlet):
    """Abstract: kappa limit from 0.22 (1150 GeV) to 0.52 (2300 GeV)."""
    m, obs, _ = _coupling_limit(t_singlet, "15515f6a", "T")
    assert np.interp(2300.0, m, obs) / SQ2 == pytest.approx(0.52, abs=0.02)
    assert np.interp(1150.0, m, obs) / SQ2 == pytest.approx(0.22, abs=0.02)


# ----------------------------------------------------------------------------- engine consistency
def test_new_channel_drives_t_singlet_exclusion(t_singlet):
    """At m_T = 2 TeV and kappa_T = 0.283 the CMS Wb search is the most sensitive analysis."""
    _quiet(t_singlet.singletT_params, mT=2000.0, s_l=0.2)
    _quiet(t_singlet.check_against_xs_and_coupling_limits)
    row = t_singlet.df.iloc[-1]
    m, obs, _ = _coupling_limit(t_singlet, "17564f4b", "T")
    assert row["label"] == "arXiv:2604.17564"
    assert row["obs_ratio"] == pytest.approx(row["coupling"] / np.interp(2000.0, m, obs), rel=1e-9)
