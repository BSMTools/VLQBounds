"""
Validation of the new single-VLQ coupling limits against the published papers.

  T singlet: CMS 2604.17564 Fig. 4 lower (kappa_W^T, right axis)   limit_scale = 1
             ATLAS 2506.15515 Fig. 6a (kappa = sin theta_L)        limit_scale = sqrt(2)
  B singlet: CMS 2606.01423 Fig. 10 upper/lower-left (kappa ~ sin theta)  limit_scale = sqrt(2)

Package convention: k_T = k_B = sqrt(2) sin(theta_L) for singlets.

Checks
  A) Paper benchmarks, evaluated on the limit arrays the engine actually uses:
     - CMS 2606.01423 Sec. 8 / Table 1: singlet B' with Gamma/m = 5% excluded below
       1.23 TeV (obs), 1.35 TeV (exp). The model kappa at Gamma/m = 5% is taken from
       the package's own width calculation.
     - CMS 2604.17564 Sec. 7: singlet T excluded 0.82-2.15 TeV for kappa_W^T = 0.2;
       Y quark (kappa_W^Y = 0.2 <=> kappa_W^T = 0.2*sqrt2) excluded 0.7-2.4 TeV.
     - ATLAS 2506.15515 abstract: kappa limit 0.22 (1150 GeV) to 0.52 (2300 GeV).
  B) Engine consistency: where a new table is the limiting channel,
     obs_ratio == kappa_model / kappa_limit(m).
  C) Adding tables never shrinks the excluded region.
  D) Plot in the (m, kappa) plane.

Run from the repository root:
    python examples/validate_new_coupling_limits.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vlqbounds.inference.coupling_limits as cl
from vlqbounds.inference import VLQBounds
from vlqbounds.models import BottomPartnerSinglet, TopPartnerSinglet

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SQ2 = math.sqrt(2.0)

NEW = {
    "T": {
        "15515f6a": ("T/ATLAS_Tables/2506.15515_ATLAS_Fig6a_pp_Tbq_Wb_singlet.json", SQ2,
                     "ATLAS 2506.15515 ($Tbq\\to Wbbq$)"),
        "17564f4b": ("T/CMS_Tables/2604.17564_CMS_Fig4b_pp_VLQ_Wb_Tsinglet.json", 1.0,
                     "CMS 2604.17564 ($T\\to Wb$)"),
    },
    "B": {
        "01423f10a": ("B/CMS_Tables/2606.01423_CMS_Fig10a_pp_Bbq_wtbq_singlet.json", SQ2,
                      "CMS 2606.01423 ($Bbq\\to tW$)"),
        "01423f10b": ("B/CMS_Tables/2606.01423_CMS_Fig10b_pp_Btq_wttq_singlet.json", SQ2,
                      "CMS 2606.01423 ($Btq\\to tW$)"),
    },
}

SETUP = {
    "T": (TopPartnerSinglet, "singletT_params", "mT", "T singlet"),
    "B": (BottomPartnerSinglet, "singletB_params", "mB", "B singlet"),
}

MASSES = np.arange(800.0, 2401.0, 25.0)
MIXINGS = np.linspace(0.005, 0.45, 60)
RESULTS: list[tuple[str, bool]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def quiet_bounds(cls):
    with contextlib.redirect_stdout(io.StringIO()):
        b = VLQBounds(cls())
        b.prepare_analysis()
    return b


def engine_limit(bounds, species: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mass grid, observed and expected limits exactly as stored in the engine."""
    i = list(bounds.coupling_key).index(key)
    mobs = bounds.coupling_MT_obs if species == "T" else bounds.coupling_MB_obs
    return (np.asarray(mobs[i], float), np.asarray(bounds.coupling_obs_lower[i], float),
            np.asarray(bounds.coupling_exp_lower[i], float))


def crossings(m_grid, curve, threshold):
    d = np.asarray(curve) - np.asarray(threshold)
    out = []
    for j in range(len(d) - 1):
        if d[j] == 0 or d[j] * d[j + 1] < 0:
            out.append(m_grid[j] + (m_grid[j + 1] - m_grid[j]) * (-d[j]) / (d[j + 1] - d[j]))
    return out


# --------------------------------------------------------------------------- A
def paper_benchmarks() -> None:
    print("\n== A) Paper benchmarks")
    # CMS 2606.01423: Gamma/m = 5% singlet B', bqB' mode
    b = quiet_bounds(BottomPartnerSinglet)
    m, obs, exp = engine_limit(b, "B", "01423f10a")
    fine = np.linspace(m[0], m[-1], 1201)
    k5 = []
    for mm in fine:
        mod = BottomPartnerSinglet(); mod.set_mB(float(mm)); mod.set_width_mass_ratio(0.05)
        k5.append(mod.get_coupling_strength())
    xo = crossings(fine, np.interp(fine, m, obs), k5)
    xe = crossings(fine, np.interp(fine, m, exp), k5)
    record("2606.01423 obs mass limit (G/m=5%)", len(xo) == 1 and abs(xo[0] - 1230) < 25,
           f"code {np.round(xo, 0)} GeV vs paper 1230 GeV")
    record("2606.01423 exp mass limit (G/m=5%)", len(xe) == 1 and abs(xe[0] - 1350) < 25,
           f"code {np.round(xe, 0)} GeV vs paper 1350 GeV")

    # CMS 2604.17564
    t = quiet_bounds(TopPartnerSinglet)
    m, obs, _ = engine_limit(t, "T", "17564f4b")
    fine = np.linspace(m[0], m[-1], 1701)
    oi = np.interp(fine, m, obs)
    x = crossings(fine, oi, 0.2)
    record("2604.17564 T excluded range, kappa_W^T=0.2",
           len(x) == 2 and abs(x[0] - 820) < 15 and abs(x[1] - 2150) < 25,
           f"code {np.round(x, 0)} GeV vs paper 820-2150 GeV")
    ky = 0.2 * SQ2
    record("2604.17564 Y range, kappa_W^Y=0.2 (=0.283 in T units)",
           np.all(oi < ky), f"limit < {ky:.3f} over full table 700-2400 GeV vs paper 0.7-2.4 TeV")
    record("2604.17564 minimum (T units)", abs(oi.min() - 0.086 * SQ2) < 0.005,
           f"code {oi.min():.4f} at {fine[oi.argmin()]:.0f} GeV vs paper 0.086*sqrt2 = {0.086*SQ2:.4f} at ~1400 GeV")

    # ATLAS 2506.15515 (convert back to paper kappa)
    m, obs, _ = engine_limit(t, "T", "15515f6a")
    lo, hi = np.interp(1150, m, obs) / SQ2, np.interp(2300, m, obs) / SQ2
    record("2506.15515 kappa at 2300 GeV", abs(hi - 0.52) < 0.02, f"code {hi:.3f} vs paper 0.52")
    record("2506.15515 kappa at 1150 GeV", abs(lo - 0.22) < 0.02,
           f"code {lo:.3f} vs paper 0.22 (digitised table)")


# --------------------------------------------------------------------------- B, C
@contextlib.contextmanager
def registry_without(species: str, keys: set[str]):
    reg_key = (species, "Singlet")
    original = cl.COUPLING_LIMIT_REGISTRY[reg_key]
    kept = [d for d in original if d.key not in keys]
    cl.COUPLING_LIMIT_REGISTRY[reg_key] = tuple(
        type(d)(**{**d.to_mapping(), "index": i}) for i, d in enumerate(kept)
    )
    try:
        yield
    finally:
        cl.COUPLING_LIMIT_REGISTRY[reg_key] = original


def scan(species: str) -> dict[str, np.ndarray]:
    cls, setter, mass_kw, _ = SETUP[species]
    bounds = quiet_bounds(cls)
    shape = (len(MIXINGS), len(MASSES))
    out = {k: np.empty(shape, dtype=object if k == "label" else float)
           for k in ("kappa", "excluded", "obs_ratio", "label")}
    for i, s in enumerate(MIXINGS):
        for j, mm in enumerate(MASSES):
            with contextlib.redirect_stdout(io.StringIO()):
                getattr(bounds, setter)(**{mass_kw: float(mm), "s_l": float(s)})
                bounds.check_against_xs_and_coupling_limits()
            row = bounds.df.iloc[-1]
            out["kappa"][i, j] = row["coupling"]
            out["excluded"][i, j] = 1.0 if int(row["result"]) == 0 else 0.0
            out["obs_ratio"][i, j] = row["obs_ratio"]
            out["label"][i, j] = str(row["label"])
    return out


def load_limit(species: str, key: str):
    rel, scale, _ = NEW[species][key]
    rows = np.asarray(json.loads((DATA / rel).read_text())["rows"], dtype=float)
    return rows[:, 0], rows[:, 1] * scale, rows[:, 2] * scale


def engine_consistency(species: str, after) -> None:
    by_label = {}
    for key, (rel, _, _) in NEW[species].items():
        by_label.setdefault("arXiv:" + Path(rel).name.split("_")[0], []).append(key)
    for label, keys in by_label.items():
        mask = after["label"] == label
        if not mask.any():
            print(f"  [info] {label}: never the limiting channel")
            continue
        curves = [load_limit(species, k) for k in keys]
        worst = 0.0
        for i, j in zip(*np.nonzero(mask)):
            mm, kap, ratio = MASSES[j], after["kappa"][i, j], after["obs_ratio"][i, j]
            preds = [kap / np.interp(mm, cm, co) for cm, co, _ in curves if cm[0] <= mm <= cm[-1]]
            if preds:
                worst = max(worst, min(abs(p - ratio) / ratio for p in preds))
        record(f"{label} engine ratio", worst < 1e-3,
               f"{int(mask.sum())} limiting points, max rel. deviation {worst:.1e}")


def main() -> int:
    paper_benchmarks()

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, species in zip(axes, ("T", "B")):
        title = SETUP[species][3]
        print(f"\n== B, C) {title}")
        after = scan(species)
        with registry_without(species, set(NEW[species])):
            before = scan(species)
        engine_consistency(species, after)
        na, nb = int(after["excluded"].sum()), int(before["excluded"].sum())
        record(f"{title} monotonic exclusion", na >= nb, f"excluded points {nb} -> {na} (+{na - nb})")

        X = np.broadcast_to(MASSES, after["kappa"].shape)
        newly = (after["excluded"] > 0.5) & (before["excluded"] < 0.5)
        ax.contourf(X, after["kappa"], after["excluded"], levels=[0.5, 1.5], colors=["#c9d7ea"])
        ax.contour(X, before["kappa"], before["excluded"], levels=[0.5], colors="k",
                   linestyles="--", linewidths=1.6)
        ax.contour(X, after["kappa"], after["excluded"], levels=[0.5], colors="k", linewidths=2.0)
        ax.scatter(X[newly], after["kappa"][newly], s=9, c="#d62728", zorder=3)
        for c, key in zip(("#1f77b4", "#2ca02c"), NEW[species]):
            m, obs, exp = load_limit(species, key)
            lbl = NEW[species][key][2]
            ax.plot(m, obs, color=c, lw=2.0, label=f"{lbl} obs")
            ax.plot(m, exp, color=c, lw=1.4, ls=":", label=f"{lbl} exp")
        if species == "B":
            mm = np.linspace(800, 2000, 61)
            k5 = []
            for x in mm:
                mod = BottomPartnerSinglet(); mod.set_mB(float(x)); mod.set_width_mass_ratio(0.05)
                k5.append(mod.get_coupling_strength())
            ax.plot(mm, k5, color="#9467bd", lw=1.6, ls="-.", label=r"model $\Gamma/m=5\%$")
        else:
            ax.axhline(0.2, color="#9467bd", lw=1.4, ls="-.", label=r"$\kappa_W^T=0.2$ (CMS benchmark)")
        ax.plot([], [], "k--", label="excluded boundary, before")
        ax.plot([], [], "k-", label="excluded boundary, after")
        ax.scatter([], [], s=12, c="#d62728", label="newly excluded")
        ax.set_xlim(MASSES[0], MASSES[-1]); ax.set_ylim(0.0, 0.8)
        ax.set_xlabel(rf"$m_{species}$ [GeV]")
        ax.set_ylabel(rf"$\kappa_{species}=\sqrt{{2}}\,\sin\theta_L$")
        ax.set_title(f"{title}: new coupling limits")
        ax.grid(alpha=0.3, ls="--")
        ax.legend(fontsize=8.2, loc="upper left", framealpha=0.9)

    out_dir = ROOT / "build" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "new_coupling_limits"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=160)
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\nSaved {out.with_suffix('.pdf')} and .png")
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    for name, ok in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
