"""Generate Figure 2 using the baseline case in examples/run_baseline.py.

Layout
------
(a) Left column, spanning both rows:
    radial-build poloidal cross-section.
(b) Right-top:
    electron/ion temperature and electron-density profiles on one axis.
(c) Right-bottom:
    total, bootstrap, current-drive, and inductive current densities,
    together with the q profile on one axis.

Run this script from the TokaSys project root, for example:

    pip install -e .
    python /path/to/figure2_geometry_profiles.py

Alternatively, set PYTHONPATH to the project root before running it.
"""

from __future__ import annotations

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from tokasys import default_problem, evaluate_reactor
from tokasys.diagnostics import (
    diagnostic_current_and_q_profiles,
    plot_radial_build,
)


jax.config.update("jax_enable_x64", True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a journal-style panel label."""
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        zorder=30,
    )


def make_figure(
    output_stem: str | Path = "figure2_geometry_profiles",
) -> tuple[Path, Path]:
    """Evaluate the baseline reactor and save Figure 2 as PNG and PDF."""

    # Use exactly the same baseline problem and evaluation as run_baseline.py.
    problem = default_problem()
    result = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
        problem.numerics,
    )

    design = problem.initial_design
    technology = problem.technology

    # Plasma profiles.
    rho = np.asarray(result.plasma.radial_grid, dtype=float)
    te_keV = np.asarray(
        result.plasma.electron_temperature_profile_keV,
        dtype=float,
    )
    ti_keV = np.asarray(
        result.plasma.ion_temperature_profile_keV,
        dtype=float,
    )
    ne_1e19 = (
        np.asarray(result.plasma.electron_density_profile_m3, dtype=float)
        / 1.0e19
    )

    # Current-density components are returned in MA/m^2.
    current = diagnostic_current_and_q_profiles(design, result)

    # Left panel spans both rows; right column contains two profile panels.
    #fig = plt.figure(figsize=(13.0, 7.2), constrained_layout=True)
    fig = plt.figure(figsize=(8.0, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=(1.45, 1.0),
        height_ratios=(1.0, 1.0),
    )

    ax_build = fig.add_subplot(gs[:, 0])
    ax_thermo = fig.add_subplot(gs[0, 1])
    ax_current = fig.add_subplot(gs[1, 1], sharex=ax_thermo)

    # ------------------------------------------------------------------
    # (a) Radial-build poloidal cross-section.
    # This is the same plotting helper used by run_baseline.py through
    # plot_reactor_diagnostics().
    # ------------------------------------------------------------------
    plot_radial_build(ax_build, design, technology, result)
    ax_build.set_title("Radial build")
    #add_panel_label(ax_build, "(a)")

    # ------------------------------------------------------------------
    # (b) Density and temperature profiles on the same y axis.
    # ------------------------------------------------------------------
    pedestal_start = max(1.0 - float(result.plasma.pedestal_width), 0.0)
    ax_thermo.axvspan(
        pedestal_start,
        1.0,
        color="0.90",
        alpha=0.65,
        linewidth=0.0,
        zorder=0,
    )

    ax_thermo.plot(rho, te_keV, linewidth=2.1, label=r"$T_e$")
    ax_thermo.plot(
        rho,
        ti_keV,
        linewidth=2.1,
        linestyle="--",
        label=r"$T_i$",
    )
    ax_thermo.plot(
        rho,
        ne_1e19,
        linewidth=2.1,
        linestyle="-.",
        label=r"$n_e$",
    )

    ax_thermo.set_xlim(0.0, 1.0)
    ax_thermo.set_ylim(bottom=0.0)
    ax_thermo.set_ylabel(
        r"$T_{e,i}$ [keV] or $n_e$ [$10^{19}$ m$^{-3}$]"
    )
    ax_thermo.set_title("Plasma density and temperature")
    ax_thermo.grid(alpha=0.25)
    ax_thermo.legend(frameon=False, loc="upper right", ncol=3)
    ax_thermo.tick_params(labelbottom=False)
    #add_panel_label(ax_thermo, "(b)")

    # ------------------------------------------------------------------
    # (c) Current-density components and q profile on the same y axis.
    # The baseline ranges are similar, so no secondary axis is used.
    # ------------------------------------------------------------------
    ax_current.plot(
        rho,
        current["total_current_density_ma_m2"],
        linewidth=2.3,
        label=r"$j_{\mathrm{total}}$",
    )
    ax_current.plot(
        rho,
        current["bootstrap_current_density_ma_m2"],
        linewidth=1.9,
        label=r"$j_{\mathrm{BS}}$",
    )
    ax_current.plot(
        rho,
        current["current_drive_current_density_ma_m2"],
        linewidth=1.9,
        linestyle="--",
        label=r"$j_{\mathrm{CD}}$",
    )
    ax_current.plot(
        rho,
        current["inductive_current_density_ma_m2"],
        linewidth=1.9,
        linestyle=":",
        label=r"$j_{\mathrm{ind}}$",
    )
    ax_current.plot(
        rho,
        current["q_profile"],
        linewidth=2.1,
        linestyle="-.",
        label=r"$q$",
    )

    ax_current.set_xlim(0.0, 1.0)
    ax_current.set_ylim(bottom=0.0)
    ax_current.set_xlabel(r"$\rho$")
    ax_current.set_ylabel(r"j [MA m$^{-2}$] or $q$")
    ax_current.set_title("Current density and safety factor")
    ax_current.grid(alpha=0.25)
    ax_current.legend(frameon=False, loc="upper center", ncol=3)
    #add_panel_label(ax_current, "(c)")

    output_stem = Path(output_stem)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


if __name__ == "__main__":
    png_file, pdf_file = make_figure()
    print(f"Saved PNG: {png_file.resolve()}")
    print(f"Saved PDF: {pdf_file.resolve()}")
