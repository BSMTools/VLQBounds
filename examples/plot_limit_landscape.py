"""
Publication-quality summary figures for singlet T and B vector-like quarks.

For each species, one two-panel figure:

  (a) Coupling-limit landscape: every single-production coupling limit in the
      registry, expressed in the package convention kappa = sqrt(2) sin(theta_L)
      (after all unit conversions applied by the engine). The 2025-2026
      analyses are highlighted; the strongest-limit envelope and the region
      excluded by pair production are shaded. Dash-dotted lines show the
      coupling at fixed Gamma/m, i.e. where the narrow-width approximation
      stops being reliable.

  (b) Exclusion map: the full VLQBounds verdict on a (m, kappa) grid, each
      excluded point coloured by the analysis that drives the exclusion,
      with the observed (solid) and expected (dashed) 95% CL boundaries.

Usage (from the repository root):
    python examples/plot_limit_landscape.py                 # T and B
    python examples/plot_limit_landscape.py --species T --out docs/figures
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from vlqbounds.inference import VLQBounds
from vlqbounds.models import BottomPartnerSinglet, TopPartnerSinglet

warnings.filterwarnings("ignore", category=FutureWarning)

SQ2 = math.sqrt(2.0)
ROOT = Path(__file__).resolve().parents[1]

# Categorical slots (validated: CVD dE >= 9.2, normal-vision dE >= 24 for all pairs).
# Fixed categorical order; slots 1-3 also validated all-pairs (area fills).
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8985"
CONTEXT = "#c9c8c3"
OTHER = "#b9b8b2"
PAIR_FILL = "#ecebe7"

NEW_ANALYSES = {"2604.17564", "2506.15515", "2606.01423"}

SPECIES = {
    "T": dict(cls=TopPartnerSinglet, setter="singletT_params", mass="mT", tex="T",
              mass_range=(800.0, 2400.0)),
    "B": dict(cls=BottomPartnerSinglet, setter="singletB_params", mass="mB", tex="B",
              mass_range=(800.0, 2400.0)),
}


# ----------------------------------------------------------------------------- style
def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "none",
        "legend.facecolor": "white",
        "legend.fontsize": 8.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def stamp(ax, species: str, extra: str = "") -> None:
    box = dict(fc="white", ec="none", alpha=0.92, pad=2.5)
    ax.text(0.03, 0.965, r"$\mathbf{VLQBounds}$" + "\n" + rf"${species}$ singlet, 95% CL" + extra,
            transform=ax.transAxes, va="top", fontsize=10, color=INK, linespacing=1.5,
            bbox=box, zorder=10)


# ----------------------------------------------------------------------------- engine
def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def make_bounds(species: str):
    b = quiet(VLQBounds, SPECIES[species]["cls"]())
    quiet(b.prepare_analysis)
    return b


def arxiv_id(label: str) -> str:
    m = re.search(r"(\d{4}\.\d{4,5}|CONF-\d{4}-\d{3})", str(label))
    return m.group(1) if m else str(label)


def coupling_curves(bounds, species: str) -> list[dict]:
    """Observed/expected limits in kappa units, exactly as the engine uses them."""
    masses = bounds.coupling_MT_obs if species == "T" else bounds.coupling_MB_obs
    curves = []
    for i, key in enumerate(bounds.coupling_key):
        m = np.asarray(masses[i], float)
        obs = np.asarray(bounds.coupling_obs_lower[i], float)
        exp = np.asarray(bounds.coupling_exp_lower[i], float)
        n = min(len(m), len(obs))
        m, obs = m[:n], obs[:n]
        exp = exp[:n] if len(exp) >= n else np.full(n, np.nan)
        to_kappa = SQ2 if str(bounds.which_coupling[i]).lower().startswith("s_") else 1.0
        ok = np.isfinite(m) & np.isfinite(obs)
        order = np.argsort(m[ok])
        curves.append(dict(
            key=key, label=bounds.coupling_label[i], expt=bounds.coupling_expt[i],
            m=m[ok][order], obs=obs[ok][order] * to_kappa, exp=exp[ok][order] * to_kappa,
        ))
    return curves


def width_line(species: str, ratio: float, masses: np.ndarray) -> np.ndarray:
    out = []
    for m in masses:
        mod = SPECIES[species]["cls"]()
        getattr(mod, f"set_m{species}")(float(m))
        mod.set_width_mass_ratio(ratio)
        out.append(mod.get_coupling_strength())
    return np.asarray(out)


def pair_mass_limit(bounds, species: str) -> tuple[float, float]:
    """Observed and expected mass limits from pair production at negligible mixing."""
    cfg = SPECIES[species]
    ms = np.arange(*cfg["mass_range"], 5.0)
    obs, exp = [], []
    for m in ms:
        quiet(getattr(bounds, cfg["setter"]), **{cfg["mass"]: float(m), "s_l": 1e-3})
        quiet(bounds.check_against_xs_limits)
        r = bounds.df.iloc[-1]
        obs.append(r["obs_ratio"]); exp.append(r["exp_ratio"])
    obs, exp = np.nan_to_num(obs), np.nan_to_num(exp)
    return float(ms[np.argmax(obs < 1)]), float(ms[np.argmax(exp < 1)])


def scan(bounds, species: str, masses: np.ndarray, mixings: np.ndarray) -> dict[str, np.ndarray]:
    cfg = SPECIES[species]
    shape = (len(mixings), len(masses))
    res = {k: np.full(shape, np.nan) for k in ("kappa", "obs", "exp")}
    res["label"] = np.empty(shape, dtype=object)
    res["expt"] = np.empty(shape, dtype=object)
    res["process"] = np.empty(shape, dtype=object)
    for i, s in enumerate(mixings):
        for j, m in enumerate(masses):
            quiet(getattr(bounds, cfg["setter"]), **{cfg["mass"]: float(m), "s_l": float(s)})
            quiet(bounds.check_against_xs_and_coupling_limits)
            r = bounds.df.iloc[-1]
            res["kappa"][i, j] = r["coupling"]
            res["obs"][i, j] = r["obs_ratio"]
            res["exp"][i, j] = r["exp_ratio"]
            res["label"][i, j] = arxiv_id(r["label"])
            res["expt"][i, j] = str(r["experiment"])
            res["process"][i, j] = str(r["process"])
    return res


def short_process(process: str) -> str:
    p = process.replace("pp -->", "").split("-->")[0].strip()
    p = p.replace("(", "").replace(")", "")
    return re.sub(r"([TBXY])([TBXY])$", r"\1\\bar{\2}", p)


# ----------------------------------------------------------------------------- panels
def label_iso_line(ax, x, y, x_at, text):
    """Inline label on a curve, rotated along it, with a white backing."""
    yi = float(np.interp(x_at, x, y))
    lo, hi = ax.get_ylim()
    if not lo < yi < hi:
        return
    p0 = ax.transData.transform((x_at - 40, np.interp(x_at - 40, x, y)))
    p1 = ax.transData.transform((x_at + 40, np.interp(x_at + 40, x, y)))
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(x_at, yi, text, fontsize=8, color=INK_2, ha="center", va="center",
            rotation=angle, rotation_mode="anchor",
            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.95), zorder=6)


def width_lines(ax, species, xlim, halo=False):
    mm = np.linspace(*xlim, 81)
    for r in (0.01, 0.05, 0.10):
        k = width_line(species, r, mm)
        if halo:
            ax.plot(mm, k, color="white", lw=2.4, zorder=2.9)
        ax.plot(mm, k, color=INK_2, lw=0.9, ls="-.", zorder=3)
        label_iso_line(ax, mm, k, 1720.0, rf"$\Gamma/m={int(r * 100)}\%$")


def panel_landscape(ax, species, curves, m_pair_obs, colors, xlim, ylim):
    tex = SPECIES[species]["tex"]
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    grid = np.linspace(*xlim, 801)

    # pair-production exclusion
    ax.axvspan(xlim[0], m_pair_obs, color=PAIR_FILL, zorder=0, lw=0)
    ax.axvline(m_pair_obs, color=INK_3, lw=0.9, zorder=1)
    ax.text(m_pair_obs - 15, ylim[0] + 0.03 * (ylim[1] - ylim[0]),
            rf"$pp\to {tex}\bar{{{tex}}}$" + f"\n$m_{tex}<{m_pair_obs/1000:.2f}$ TeV",
            ha="right", va="bottom", fontsize=8.6, color=INK_2)

    # envelope of observed single-production limits
    stack = np.full((len(curves), grid.size), np.inf)
    for k, c in enumerate(curves):
        inside = (grid >= c["m"][0]) & (grid <= c["m"][-1])
        stack[k, inside] = np.interp(grid[inside], c["m"], c["obs"])
    env = stack.min(axis=0)
    good = np.isfinite(env)
    ax.fill_between(grid[good], env[good], ylim[1], color="#dbe7f6", lw=0, zorder=0.5)
    ax.plot(grid[good], env[good], color=INK, lw=1.6, zorder=4)

    for c in curves:
        if arxiv_id(c["label"]) not in NEW_ANALYSES:
            ax.plot(c["m"], c["obs"], color=CONTEXT, lw=0.8, zorder=2)

    width_lines(ax, species, xlim)

    handles = []
    new = sorted((c for c in curves if arxiv_id(c["label"]) in NEW_ANALYSES),
                 key=lambda c: (list(colors).index(arxiv_id(c["label"])), c["key"]))
    seen = {}
    for c in new:
        aid = arxiv_id(c["label"])
        n = seen[aid] = seen.get(aid, -1) + 1
        ls = "-" if n == 0 else (0, (5, 2.2))
        ax.plot(c["m"], c["obs"], color=colors[aid], lw=2.1, ls=ls, zorder=5,
                solid_capstyle="round", dash_capstyle="round")
        tag = {"01423f10a": r" ($bqB$)", "01423f10b": r" ($tqB$)"}.get(c["key"], "")
        handles.append(Line2D([], [], color=colors[aid], lw=2.1, ls=ls,
                              label=f"{c['expt']} {aid}{tag}"))
    handles += [
        Line2D([], [], color=CONTEXT, lw=1.0, label="other single-production limits"),
        Line2D([], [], color=INK, lw=1.6, label="strongest observed limit"),
        Patch(fc=PAIR_FILL, ec="none", label=rf"excluded by $pp\to {tex}\bar{{{tex}}}$"),
    ]
    ax.legend(handles=handles, loc="upper right", borderaxespad=0.6, handlelength=2.4)
    ax.set_xlabel(rf"$m_{tex}$ [GeV]")
    ax.set_ylabel(rf"$\kappa_{tex}\;(=\sqrt{{2}}\,\sin\theta_L)$")
    stamp(ax, tex, r",  $\sqrt{s}=13$ TeV")
    ax.set_title("(a) Coupling-limit landscape (observed)", loc="left", color=INK)


def panel_map(ax, species, masses, res, top, colors, names, xlim, ylim):
    tex = SPECIES[species]["tex"]
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    excluded = np.nan_to_num(res["obs"]) >= 1.0

    idx = np.zeros(excluded.shape, int)            # 0 = allowed
    for k, lab in enumerate(top, start=1):
        idx[excluded & (res["label"] == lab)] = k
    idx[excluded & (idx == 0)] = len(top) + 1      # Other
    cmap = ListedColormap(["#ffffff"] + [colors[l] for l in top] + [OTHER])

    X = np.broadcast_to(masses, res["kappa"].shape)
    ax.pcolormesh(X, res["kappa"], idx, cmap=cmap, vmin=-0.5, vmax=len(top) + 1.5,
                  shading="nearest", alpha=0.82, zorder=0, rasterized=True)
    ax.contour(X, res["kappa"], np.nan_to_num(res["obs"]), levels=[1.0], colors=INK,
               linewidths=1.8, zorder=4)
    ax.contour(X, res["kappa"], np.nan_to_num(res["exp"]), levels=[1.0], colors=INK,
               linewidths=1.3, linestyles=[(0, (4, 2.5))], zorder=4)
    width_lines(ax, species, xlim, halo=True)

    handles = [Patch(fc=colors[lab], ec="none", alpha=0.82,
                     label=rf"{names[lab][1]} {lab}  (${short_process(names[lab][2])}$)")
               for lab in top]
    if (idx == len(top) + 1).any():
        handles.append(Patch(fc=OTHER, ec="none", alpha=0.82, label="other analyses"))
    handles += [
        Line2D([], [], color=INK, lw=1.8, label="observed 95% CL boundary"),
        Line2D([], [], color=INK, lw=1.3, ls=(0, (4, 2.5)), label="expected 95% CL boundary"),
        Line2D([], [], color=INK_2, lw=0.9, ls="-.", label=r"fixed $\Gamma/m$"),
    ]
    ax.legend(handles=handles, loc="upper right", borderaxespad=0.6, handlelength=2.2,
              title="excluded by", title_fontsize=8.6, alignment="left")
    ax.set_xlabel(rf"$m_{tex}$ [GeV]")
    ax.set_ylabel(rf"$\kappa_{tex}\;(=\sqrt{{2}}\,\sin\theta_L)$")
    stamp(ax, tex, r",  $\sqrt{s}=13$ TeV")
    ax.set_title("(b) Exclusion map: most sensitive analysis", loc="left", color=INK)


def assign_colors(res, curves) -> tuple[list[str], dict[str, str], dict]:
    """One colour per analysis across both panels (colour follows the entity).

    Order: largest-area pair-production analysis, then 2025-26 analyses, then the
    rest of the three largest map regions, then remaining highlighted curves.
    """
    excluded = np.nan_to_num(res["obs"]) >= 1.0
    names: dict[str, list] = {}
    for lab, ex, pr in zip(res["label"][excluded], res["expt"][excluded], res["process"][excluded]):
        names.setdefault(lab, [0, ex, pr])[0] += 1
    top = [lab for lab, _ in sorted(names.items(), key=lambda kv: -kv[1][0])[:3]]
    is_pair = lambda lab: re.search(r"-->\s*([TBXY])\1\b", names[lab][2]) is not None
    order = [l for l in top if is_pair(l)][:1]
    order += [l for l in top if l in NEW_ANALYSES and l not in order]
    order += [l for l in top if l not in order]
    for c in curves:
        aid = arxiv_id(c["label"])
        if aid in NEW_ANALYSES and aid not in order:
            order.append(aid)
    colors = {lab: SLOT[i] for i, lab in enumerate(order)}
    top = [l for l in order if l in top]
    return top, colors, names


# ----------------------------------------------------------------------------- main
def make_figure(species: str, out_dir: Path, n_mass: int, n_mix: int) -> list[Path]:
    cfg = SPECIES[species]
    xlim, ylim = cfg["mass_range"], (0.0, 0.8)
    bounds = make_bounds(species)
    curves = coupling_curves(bounds, species)
    m_obs, m_exp = pair_mass_limit(bounds, species)

    masses = np.linspace(*xlim, n_mass)
    mixings = np.linspace(2e-3, ylim[1] / SQ2 * 0.999, n_mix)
    res = scan(bounds, species, masses, mixings)
    top, colors, names = assign_colors(res, curves)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)
    fig.canvas.draw()
    panel_landscape(axes[0], species, curves, m_obs, colors, xlim, ylim)
    panel_map(axes[1], species, masses, res, top, colors, names, xlim, ylim)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"vlqbounds_{species}_singlet"
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    print(f"{species} singlet: pair-production mass limit obs {m_obs:.0f} GeV / exp {m_exp:.0f} GeV")
    print(f"  wrote {paths[0]} and {paths[1].name}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", nargs="+", default=["T", "B"], choices=["T", "B"])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--n-mass", type=int, default=97, help="mass grid points (default 97)")
    ap.add_argument("--n-mix", type=int, default=80, help="mixing grid points (default 80)")
    args = ap.parse_args()
    set_style()
    for sp in args.species:
        make_figure(sp, args.out, args.n_mass, args.n_mix)


if __name__ == "__main__":
    main()
