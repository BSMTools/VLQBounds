from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator


def _latex_available() -> bool:
    """
    Return True when Matplotlib can actually typeset with LaTeX.

    A `latex` executable alone is not enough: Matplotlib also needs packages
    such as type1cm and cm-super, so a tiny test render is attempted.
    """
    import shutil

    if shutil.which("latex") is None:
        return False
    try:
        from matplotlib.texmanager import TexManager

        TexManager().make_dvi(r"$\kappa$", 10)
        return True
    except Exception:
        return False


def configure_plot_style(*, use_tex: bool = True) -> None:
    """
    Configure a publication-quality plotting style.

    The function is TeX-first. If LaTeX is requested but unavailable, it
    falls back automatically to Matplotlib's mathtext while preserving a
    consistent serif look.
    """
    tex_enabled = bool(use_tex and _latex_available())

    matplotlib.rcParams.update(
        {
            "text.usetex": tex_enabled,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.linewidth": 1.8,
            "axes.labelsize": 25,
            "axes.titlesize": 25,
            "axes.titlepad": 12.0,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 19,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "hatch.linewidth": 1.2,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def add_ticks_to_tricontour(
    ax,
    contour_set,
    *,
    level_index: int = 0,
    tick_length: float = 0.01,
    tick_spacing: float = 10.0,
    tick_width: float = 1.5,
    side: str = "left",
    inclination_angle: float = 0.0,
    tick_color: str = "red",
) -> None:
    """
    Add short hatch-like ticks along a selected tricontour level.
    """
    segments = []
    level_paths = contour_set.allsegs[level_index]
    inclination_rad = np.deg2rad(inclination_angle)

    for seg in level_paths:
        if len(seg) < 2:
            continue

        x = seg[:, 0]
        y = seg[:, 1]
        distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        if not np.any(distances > 0):
            continue

        cumulative_distances = np.concatenate([[0.0], np.cumsum(distances)])
        tick_positions = np.arange(0.0, cumulative_distances[-1], tick_spacing)

        for t in tick_positions:
            idx = np.searchsorted(cumulative_distances, t) - 1
            if idx < 0 or idx >= len(x) - 1:
                continue
            if distances[idx] == 0:
                continue

            x0, y0 = x[idx], y[idx]
            x1, y1 = x[idx + 1], y[idx + 1]
            frac = (t - cumulative_distances[idx]) / distances[idx]
            xt = x0 + frac * (x1 - x0)
            yt = y0 + frac * (y1 - y0)

            dx = x1 - x0
            dy = y1 - y0
            norm = np.hypot(dx, dy)
            if norm == 0:
                continue
            dx /= norm
            dy /= norm

            if side == "left":
                nx, ny = -dy, dx
            elif side == "right":
                nx, ny = dy, -dx
            else:
                raise ValueError("side must be 'left' or 'right'")

            inclined_dx = np.cos(inclination_rad) * nx - np.sin(inclination_rad) * ny
            inclined_dy = np.sin(inclination_rad) * nx + np.cos(inclination_rad) * ny

            x_end = xt + tick_length * inclined_dx
            y_end = yt + tick_length * inclined_dy
            segments.append([[xt, yt], [x_end, y_end]])

    if segments:
        tick_collection = LineCollection(segments, color=tick_color, linewidths=tick_width)
        ax.add_collection(tick_collection)


def run_parameter_scan(
    *,
    bounds,
    parameter_callback: Callable[[object, float, float], None],
    masses: Iterable[float],
    mixings: Iterable[float],
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Run a two-dimensional scan over mass and width-ratio / mixing-like input.
    """
    rows: List[pd.Series] = []

    for mixing in mixings:
        for mass in masses:
            parameter_callback(bounds, mass, mixing)
            bounds.check_against_xs_and_coupling_limits()
            bounds.get_key()
            if not quiet:
                bounds.print_result()
            if not bounds.df.empty:
                rows.append(bounds.df.iloc[-1].copy())

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _build_title(title: str) -> str:
    """
    Return a TeX-friendly title string.
    """
    return rf"{title}"


def save_scan_outputs(
    result_df: pd.DataFrame,
    *,
    output_stem: str,
    x_label: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    use_tex: bool = True,
    colorbar_label: str = r"$\kappa$",
) -> None:
    """
    Save the scan results to CSV and PDF.

    The plot uses TeX rendering when available.
    """
    csv_path = Path(f"{output_stem}.csv")
    pdf_path = Path(f"{output_stem}.pdf")
    result_df.to_csv(csv_path, index=False)

    data = result_df[result_df["obs_ratio"].notna() & result_df["coupling"].notna()]
    if data.empty:
        print(f"No plottable points found for {output_stem}. CSV saved to {csv_path}.")
        return

    configure_plot_style(use_tex=use_tex)

    x = data["mass"].to_numpy(dtype=float)
    y = data["width_ratio"].to_numpy(dtype=float)
    z = data["obs_ratio"].to_numpy(dtype=float)
    c = data["coupling"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    filled = ax.tricontourf(x, y, c, levels=30, cmap="GnBu")
    contour_set = ax.tricontour(x, y, z, levels=[1.0], colors="red", linewidths=2.4)

    add_ticks_to_tricontour(
        ax,
        contour_set,
        level_index=0,
        tick_length=0.012,
        tick_spacing=10.0,
        tick_width=1.4,
        side="left",
        inclination_angle=0.0,
        tick_color="red",
    )

    excluded_handle = Line2D([0], [0], color="red", lw=2.4, label=r"$\mathrm{Excluded}$")
    ax.legend(handles=[excluded_handle], loc="upper left", frameon=False)

    colorbar = fig.colorbar(filled, ax=ax)
    colorbar.set_label(colorbar_label)

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\Gamma/M$")
    ax.set_title(_build_title(title))

    ax.grid(True)
    ax.tick_params(
        which="major",
        bottom=True,
        top=True,
        left=True,
        right=True,
        length=8,
        width=1.8,
        direction="in",
    )
    ax.tick_params(
        which="minor",
        bottom=True,
        top=True,
        left=True,
        right=True,
        length=4.5,
        width=1.1,
        direction="in",
    )
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved scan outputs: {csv_path} and {pdf_path}")


