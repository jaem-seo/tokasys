"""Reusable diagnostic plots for reactor evaluations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="tokasys-mpl-"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, Rectangle
from matplotlib.patches import PathPatch

from tokasys.core.constants import ALPHA_FRACTION
from tokasys.plasma.fusion import dt_fusion_power_density_mw_m3
from tokasys.plasma.profiles import diagnostic_q_profile_and_shear
from tokasys.plasma.radiation import electron_ion_bremsstrahlung_mw


def _as_float(value: Any) -> float:
    return float(np.asarray(value))


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([np.asarray([0.0]), np.cumsum(increments)])


def _miller_boundary(
    major_radius_m: float,
    minor_radius_m: float,
    elongation: float,
    triangularity: float,
    num_points: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    r = major_radius_m + minor_radius_m * np.cos(
        theta + triangularity * np.sin(theta)
    )
    z = elongation * minor_radius_m * np.sin(theta)
    return r, z


def _closed_polygon(
    r: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if r.size == 0:
        return r, z
    if np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1]):
        return r, z
    return np.concatenate([r, r[:1]]), np.concatenate([z, z[:1]])


def _polygon_signed_area(r: np.ndarray, z: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    if r.size > 1 and np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1]):
        r = r[:-1]
        z = z[:-1]
    return 0.5 * float(np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z))


def _oriented_polygon(
    r: np.ndarray,
    z: np.ndarray,
    *,
    clockwise: bool,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    area = _polygon_signed_area(r, z)
    should_reverse = (clockwise and area > 0.0) or ((not clockwise) and area < 0.0)
    if should_reverse:
        return r[::-1], z[::-1]
    return r, z


def _path_vertices_and_codes(
    r: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r, z = _closed_polygon(np.asarray(r, dtype=float), np.asarray(z, dtype=float))
    vertices = np.column_stack([r, z])
    codes = np.full(r.size, MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY
    return vertices, codes


def _closed_path_patch(
    r: np.ndarray,
    z: np.ndarray,
    *,
    facecolor: str,
    edgecolor: str = "white",
    linewidth: float = 0.7,
    alpha: float = 0.90,
    zorder: int = 1,
) -> PathPatch:
    vertices, codes = _path_vertices_and_codes(r, z)
    return PathPatch(
        MplPath(vertices, codes),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        joinstyle="miter",
    )


def _closed_shell_patch(
    outer_r: np.ndarray,
    outer_z: np.ndarray,
    inner_r: np.ndarray,
    inner_z: np.ndarray,
    *,
    facecolor: str,
    edgecolor: str = "white",
    linewidth: float = 0.7,
    alpha: float = 0.90,
    zorder: int = 1,
) -> PathPatch:
    outer_r, outer_z = _oriented_polygon(outer_r, outer_z, clockwise=False)
    inner_r, inner_z = _oriented_polygon(inner_r, inner_z, clockwise=True)
    outer_vertices, outer_codes = _path_vertices_and_codes(outer_r, outer_z)
    inner_vertices, inner_codes = _path_vertices_and_codes(inner_r, inner_z)
    return PathPatch(
        MplPath(
            np.concatenate([outer_vertices, inner_vertices]),
            np.concatenate([outer_codes, inner_codes]),
        ),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )


def _asymmetric_miller_boundary(
    major_radius_m: float,
    inboard_minor_radius_m: float,
    outboard_minor_radius_m: float,
    vertical_half_height_m: float,
    triangularity: float,
    num_points: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    shaped_cos = np.cos(theta + triangularity * np.sin(theta))
    radial_extent = np.where(
        shaped_cos >= 0.0,
        outboard_minor_radius_m,
        inboard_minor_radius_m,
    )
    r = major_radius_m + radial_extent * shaped_cos
    z = vertical_half_height_m * np.sin(theta)
    return r, z


def _fill_asymmetric_miller_shell(
    ax: Any,
    major_radius_m: float,
    inner_inboard_minor_radius_m: float,
    inner_outboard_minor_radius_m: float,
    outer_inboard_minor_radius_m: float,
    outer_outboard_minor_radius_m: float,
    inner_vertical_half_height_m: float,
    outer_vertical_half_height_m: float,
    triangularity: float,
    *,
    facecolor: str,
    edgecolor: str = "white",
    alpha: float = 0.90,
    zorder: int = 1,
) -> None:
    outer_r, outer_z = _asymmetric_miller_boundary(
        major_radius_m,
        outer_inboard_minor_radius_m,
        outer_outboard_minor_radius_m,
        outer_vertical_half_height_m,
        triangularity,
    )
    inner_r, inner_z = _asymmetric_miller_boundary(
        major_radius_m,
        inner_inboard_minor_radius_m,
        inner_outboard_minor_radius_m,
        inner_vertical_half_height_m,
        triangularity,
    )
    ax.add_patch(
        _closed_shell_patch(
            outer_r,
            outer_z,
            inner_r,
            inner_z,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.7,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _fill_closed_polygon(
    ax: Any,
    r: np.ndarray,
    z: np.ndarray,
    *,
    facecolor: str,
    edgecolor: str = "white",
    linewidth: float = 0.7,
    alpha: float = 0.90,
    zorder: int = 1,
) -> None:
    ax.add_patch(
        _closed_path_patch(
            r,
            z,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _fill_closed_shell(
    ax: Any,
    outer_r: np.ndarray,
    outer_z: np.ndarray,
    inner_r: np.ndarray,
    inner_z: np.ndarray,
    *,
    facecolor: str,
    edgecolor: str = "white",
    linewidth: float = 0.7,
    alpha: float = 0.90,
    zorder: int = 1,
) -> None:
    ax.add_patch(
        _closed_shell_patch(
            outer_r,
            outer_z,
            inner_r,
            inner_z,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    )


def _fill_miller_shell(
    ax: Any,
    major_radius_m: float,
    inner_minor_radius_m: float,
    outer_minor_radius_m: float,
    elongation: float,
    triangularity: float,
    *,
    inner_elongation: float | None = None,
    outer_elongation: float | None = None,
    facecolor: str,
    edgecolor: str = "white",
    alpha: float = 0.90,
    zorder: int = 1,
) -> None:
    inner_kappa = elongation if inner_elongation is None else inner_elongation
    outer_kappa = elongation if outer_elongation is None else outer_elongation
    outer_r, outer_z = _miller_boundary(
        major_radius_m,
        outer_minor_radius_m,
        outer_kappa,
        triangularity,
    )
    inner_r, inner_z = _miller_boundary(
        major_radius_m,
        inner_minor_radius_m,
        inner_kappa,
        triangularity,
    )
    _fill_closed_shell(
        ax,
        outer_r,
        outer_z,
        inner_r,
        inner_z,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.7,
        alpha=alpha,
        zorder=zorder,
    )


def _d_shape_polygon(
    inboard_radius_m: float,
    top_radius_m: float,
    outboard_radius_m: float,
    top_half_height_m: float,
    outboard_half_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Angular six-edge D-shaped polygon for diagnostic TF/PF layout only."""
    r = np.asarray(
        [
            inboard_radius_m,
            inboard_radius_m,
            top_radius_m,
            outboard_radius_m,
            outboard_radius_m,
            top_radius_m,
        ],
        dtype=float,
    )
    z = np.asarray(
        [
            -top_half_height_m,
            top_half_height_m,
            top_half_height_m,
            outboard_half_height_m,
            -outboard_half_height_m,
            -top_half_height_m,
        ],
        dtype=float,
    )
    return r, z


def _rounded_outboard_d_shape_polygon(
    inboard_radius_m: float,
    top_radius_m: float,
    outboard_radius_m: float,
    top_half_height_m: float,
    outboard_half_height_m: float,
    *,
    arc_points: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """D-shaped TF boundary with straight inboard/top/bottom and rounded outboard."""
    del outboard_half_height_m
    radial_span = max(outboard_radius_m - top_radius_m, 1.0e-6)
    arc_center_r = top_radius_m + (
        radial_span**2 - top_half_height_m**2
    ) / (2.0 * radial_span)
    arc_radius = max(outboard_radius_m - arc_center_r, 1.0e-6)
    arc_angle_top = np.arctan2(top_half_height_m, top_radius_m - arc_center_r)
    arc_angle_bottom = np.arctan2(-top_half_height_m, top_radius_m - arc_center_r)
    arc_angles = np.linspace(arc_angle_top, arc_angle_bottom, arc_points)
    arc_r = arc_center_r + arc_radius * np.cos(arc_angles)
    arc_z = arc_radius * np.sin(arc_angles)
    r = np.concatenate(
        [
            np.asarray([inboard_radius_m, inboard_radius_m, top_radius_m]),
            arc_r,
            np.asarray([top_radius_m]),
        ],
    )
    z = np.concatenate(
        [
            np.asarray([-top_half_height_m, top_half_height_m, top_half_height_m]),
            arc_z,
            np.asarray([-top_half_height_m]),
        ],
    )
    return r, z


def _fill_tf_d_shape(
    ax: Any,
    *,
    inner_inboard_radius_m: float,
    inner_outboard_radius_m: float,
    top_radius_m: float,
    inner_top_half_height_m: float,
    tf_thickness_m: float,
    facecolor: str,
    edgecolor: str = "white",
    alpha: float = 0.90,
    zorder: int = 0,
) -> dict[str, float]:
    """Draw a diagnostic D-shaped TF coil with a rounded outboard shell."""
    thickness = max(tf_thickness_m, 1.0e-6)
    inner_outboard_half_height = 0.62 * inner_top_half_height_m
    outer_top_half_height = inner_top_half_height_m + thickness
    outer_outboard_half_height = 0.62 * outer_top_half_height
    outer_inboard_radius = max(inner_inboard_radius_m - thickness, 0.0)
    outer_outboard_radius = inner_outboard_radius_m + thickness
    outer_top_radius = top_radius_m + 0.20 * thickness

    inner_r, inner_z = _rounded_outboard_d_shape_polygon(
        inner_inboard_radius_m,
        top_radius_m,
        inner_outboard_radius_m,
        inner_top_half_height_m,
        inner_outboard_half_height,
    )
    outer_r, outer_z = _rounded_outboard_d_shape_polygon(
        outer_inboard_radius,
        outer_top_radius,
        outer_outboard_radius,
        outer_top_half_height,
        outer_outboard_half_height,
    )
    _fill_closed_shell(
        ax,
        outer_r,
        outer_z,
        inner_r,
        inner_z,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
        alpha=alpha,
        zorder=zorder,
    )
    return {
        "outer_inboard_radius_m": outer_inboard_radius,
        "inner_inboard_radius_m": inner_inboard_radius_m,
        "inner_outboard_radius_m": inner_outboard_radius_m,
        "outer_outboard_radius_m": outer_outboard_radius,
        "inner_top_half_height_m": inner_top_half_height_m,
        "outer_top_half_height_m": outer_top_half_height,
        "inner_outboard_half_height_m": inner_outboard_half_height,
        "outer_outboard_half_height_m": outer_outboard_half_height,
        "top_radius_m": top_radius_m,
        "outer_top_radius_m": outer_top_radius,
    }


def diagnostic_current_and_q_profiles(
    design: Any,
    result: Any,
) -> dict[str, np.ndarray]:
    """Return model current-density components and a diagnostic q profile."""
    rho = np.asarray(result.plasma.radial_grid, dtype=float)
    inductive_j = np.asarray(
        result.plasma.inductive_current_density_profile_a_m2,
        dtype=float,
    ) / 1.0e6
    bootstrap_j = np.asarray(
        result.plasma.bootstrap_current_density_profile_a_m2,
        dtype=float,
    ) / 1.0e6
    current_drive_j = np.asarray(
        result.plasma.current_drive_current_density_profile_a_m2,
        dtype=float,
    ) / 1.0e6
    total_j = np.asarray(
        result.plasma.total_current_density_profile_a_m2,
        dtype=float,
    ) / 1.0e6

    enclosed_integral = _cumulative_trapezoid(total_j * rho, rho)
    enclosed_fraction = enclosed_integral / np.maximum(enclosed_integral[-1], 1.0e-30)
    q_profile, _, _, _ = diagnostic_q_profile_and_shear(
        result.plasma.radial_grid,
        result.plasma.q95,
    )

    return {
        "rho": rho,
        "inductive_current_density_ma_m2": inductive_j,
        "bootstrap_current_density_ma_m2": bootstrap_j,
        "current_drive_current_density_ma_m2": current_drive_j,
        "total_current_density_ma_m2": total_j,
        "enclosed_current_fraction": enclosed_fraction,
        "q_profile": np.asarray(q_profile, dtype=float),
    }


def _density_from_weighted_shape(
    total_mw: float,
    shape: np.ndarray,
    weights: np.ndarray,
    volume_m3: float,
) -> np.ndarray:
    normalized = shape / np.maximum(np.sum(shape * weights), 1.0e-30)
    return total_mw * normalized / np.maximum(volume_m3, 1.0e-30)


def diagnostic_heating_profiles(
    design: Any,
    technology: Any,
    result: Any,
) -> dict[str, np.ndarray]:
    """Return source/loss profiles in MW/m3 for plotting."""
    rho = np.asarray(result.plasma.radial_grid, dtype=float)
    weights = np.asarray(result.plasma.volume_weights, dtype=float)
    ne = np.asarray(result.plasma.electron_density_profile_m3, dtype=float)
    ti = np.asarray(result.plasma.ion_temperature_profile_keV, dtype=float)
    te = np.asarray(result.plasma.electron_temperature_profile_keV, dtype=float)
    pressure = np.asarray(result.plasma.thermal_pressure_profile_pa, dtype=float)
    volume = _as_float(result.geometry.plasma_volume_m3)

    fusion_density = np.asarray(
        dt_fusion_power_density_mw_m3(
            ne,
            ti,
            technology.fuel_ion_fraction,
            technology.profile_fusion_enhancement,
        ),
        dtype=float,
    )
    alpha_density = ALPHA_FRACTION * fusion_density

    pressure_shape = np.maximum(pressure, 0.0)
    external_density = _density_from_weighted_shape(
        _as_float(design.auxiliary_heating_mw + design.current_drive_power_mw),
        pressure_shape,
        weights,
        volume,
    )

    ohmic_density = np.asarray(
        result.plasma.ohmic_power_density_profile_mw_m3,
        dtype=float,
    )

    brems_density = np.asarray(
        electron_ion_bremsstrahlung_mw(ne, te, technology.zeff, 1.0),
        dtype=float,
    )
    synch_density = _density_from_weighted_shape(
        _as_float(result.plasma.synchrotron_mw),
        pressure_shape,
        weights,
        volume,
    )

    return {
        "rho": rho,
        "external_mw_m3": external_density,
        "alpha_mw_m3": alpha_density,
        "ohmic_mw_m3": ohmic_density,
        "radiation_loss_mw_m3": brems_density + synch_density,
    }


def plot_radial_build(ax: Any, design: Any, technology: Any, result: Any) -> None:
    geom = result.geometry
    major_radius = _as_float(design.major_radius_m)
    minor_radius = _as_float(geom.minor_radius_m)
    elongation = _as_float(design.elongation)
    triangularity = _as_float(design.triangularity)
    sol_width = _as_float(technology.sol_width_m)
    inboard_first_wall = _as_float(design.inboard_first_wall_thickness_m)
    outboard_first_wall = _as_float(design.outboard_first_wall_thickness_m)
    inboard_blanket = _as_float(design.inboard_blanket_thickness_m)
    outboard_blanket = _as_float(design.outboard_blanket_thickness_m)
    inboard_shield = _as_float(design.inboard_shield_thickness_m)
    outboard_shield = _as_float(design.outboard_shield_thickness_m)
    vacuum_vessel = _as_float(technology.vacuum_vessel_thickness_m)
    thermal_gap = _as_float(technology.thermal_gap_m)
    shaped_layers = [
        ("SOL", sol_width, sol_width, "#fee08b"),
        ("FW", inboard_first_wall, outboard_first_wall, "#fdae61"),
        ("blanket", inboard_blanket, outboard_blanket, "#2ca25f"),
        ("shield", inboard_shield, outboard_shield, "#665191"),
        ("VV", vacuum_vessel, vacuum_vessel, "#8c8c8c"),
        ("gap", thermal_gap, thermal_gap, "#d8d8d8"),
    ]
    plasma_half_height = elongation * minor_radius
    vv_outer_half_height = plasma_half_height + (
        max(sol_width, sol_width, 0.0)
        + max(inboard_first_wall, outboard_first_wall, 0.0)
        + max(inboard_blanket, outboard_blanket, 0.0)
        + max(inboard_shield, outboard_shield, 0.0)
        + max(vacuum_vessel, vacuum_vessel, 0.0)
    )
    tf_plot_gap = max(thermal_gap, 0.0)
    tf_inner_half_height = vv_outer_half_height + tf_plot_gap
    tf_thickness = max(_as_float(design.tf_coil_thickness_m), 0.0)
    shaped_half_height = plasma_half_height + sum(
        max(in_thickness, out_thickness, 0.0)
        for _, in_thickness, out_thickness, _ in shaped_layers
    )
    drawing_half_height = max(shaped_half_height, tf_inner_half_height + tf_thickness)
    pf_radial_thickness = max(_as_float(design.pf_coil_thickness_m), 0.0)
    pf_vertical_thickness = pf_radial_thickness
    has_pf_coils = pf_radial_thickness > 0.0 and pf_vertical_thickness > 0.0
    pf_half_width = 0.5 * pf_radial_thickness
    pf_half_height = 0.5 * pf_vertical_thickness
    pf_outer_radius = _as_float(geom.machine_outer_radius_m)

    legend_items = [
        Patch(facecolor="#7a5195", edgecolor="white", label="CS"),
        Patch(facecolor="#d8d8d8", edgecolor="white", label="gap"),
        Patch(facecolor="#2f4b7c", edgecolor="white", label="TF"),
    ]
    cs_outer = _as_float(design.cs_thickness_m)
    ax.add_patch(
        Rectangle(
            (0.0, -0.92 * drawing_half_height),
            cs_outer,
            1.84 * drawing_half_height,
            facecolor="#7a5195",
            edgecolor="white",
            linewidth=0.8,
            zorder=0,
        )
    )
    ax.text(
        0.5 * cs_outer,
        0.0,
        "CS",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        zorder=5,
    )

    tf_inner_inboard_radius = _as_float(geom.inboard_tf_outer_radius_m)
    tf_inner_outboard_radius = _as_float(geom.outboard_tf_inner_radius_m)
    tf_top_radius = major_radius + 0.10 * (
        tf_inner_outboard_radius - tf_inner_inboard_radius
    )
    tf_shape = _fill_tf_d_shape(
        ax,
        inner_inboard_radius_m=tf_inner_inboard_radius,
        inner_outboard_radius_m=tf_inner_outboard_radius,
        top_radius_m=tf_top_radius,
        inner_top_half_height_m=tf_inner_half_height,
        tf_thickness_m=tf_thickness,
        facecolor="#2f4b7c",
        alpha=0.90,
        zorder=0,
    )
    inner_gap_r, inner_gap_z = _rounded_outboard_d_shape_polygon(
        tf_inner_inboard_radius,
        tf_top_radius,
        tf_inner_outboard_radius,
        tf_shape["inner_top_half_height_m"],
        tf_shape["inner_outboard_half_height_m"],
    )
    _fill_closed_polygon(
        ax,
        inner_gap_r,
        inner_gap_z,
        facecolor="#d8d8d8",
        edgecolor="none",
        linewidth=0.0,
        alpha=0.82,
        zorder=-1,
    )
    ax.text(
        0.5
        * (
            tf_shape["inner_outboard_radius_m"]
            + tf_shape["outer_outboard_radius_m"]
        ),
        0.0,
        "TF",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        zorder=5,
    )
    if has_pf_coils:
        pf_color = "#1f9d8a"
        legend_items.append(Patch(facecolor=pf_color, edgecolor="white", label="PF"))
        pf_gap = max(_as_float(getattr(technology, "pf_coil_radial_gap_m", 0.0)), 0.20)
        inboard_pf_radius = max(
            cs_outer + pf_gap + pf_half_width,
            tf_shape["inner_inboard_radius_m"] + pf_gap + pf_half_width,
        )
        sloped_pf_radius = (
            0.45 * tf_shape["outer_top_radius_m"]
            + 0.55 * tf_shape["outer_outboard_radius_m"]
            + pf_gap
            + pf_half_width
        )
        sloped_pf_z = (
            0.55 * tf_shape["outer_top_half_height_m"]
            + 0.45 * tf_shape["outer_outboard_half_height_m"]
            + pf_gap
            + pf_half_height
        )
        outer_pf_radius = (
            tf_shape["outer_outboard_radius_m"] + pf_gap + pf_half_width
        )
        pf_plot_radii = np.asarray(
            [
                inboard_pf_radius,
                inboard_pf_radius,
                sloped_pf_radius,
                sloped_pf_radius,
                outer_pf_radius,
                outer_pf_radius,
            ],
            dtype=float,
        )
        pf_plot_z = np.asarray(
            [
                tf_shape["outer_top_half_height_m"] + pf_gap + pf_half_height,
                -(tf_shape["outer_top_half_height_m"] + pf_gap + pf_half_height),
                sloped_pf_z,
                -sloped_pf_z,
                sloped_pf_z / 3,
                -sloped_pf_z / 3,
            ],
            dtype=float,
        )
        drawing_half_height = max(
            drawing_half_height,
            float(np.max(np.abs(pf_plot_z)) + pf_half_height),
        )
        pf_outer_radius = max(
            pf_outer_radius,
            float(np.max(pf_plot_radii + pf_half_width)),
        )
        for r_center, z_center in zip(pf_plot_radii, pf_plot_z, strict=True):
            ax.add_patch(
                Rectangle(
                    (r_center - pf_half_width, z_center - pf_half_height),
                    pf_radial_thickness,
                    pf_vertical_thickness,
                    facecolor=pf_color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.92,
                    zorder=2,
                )
            )
            ax.text(
                r_center,
                z_center,
                "PF",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                zorder=5,
            )

    inboard_layer_inner = minor_radius
    outboard_layer_inner = minor_radius
    layer_half_height = plasma_half_height
    legend_labels = {item.get_label() for item in legend_items}
    for label, in_thickness, out_thickness, color in shaped_layers:
        inboard_layer_thickness = max(in_thickness, 0.0)
        outboard_layer_thickness = max(out_thickness, 0.0)
        inboard_layer_outer = inboard_layer_inner + inboard_layer_thickness
        outboard_layer_outer = outboard_layer_inner + outboard_layer_thickness
        layer_outer_half_height = layer_half_height + max(
            inboard_layer_thickness,
            outboard_layer_thickness,
        )
        _fill_asymmetric_miller_shell(
            ax,
            major_radius,
            inboard_layer_inner,
            outboard_layer_inner,
            inboard_layer_outer,
            outboard_layer_outer,
            layer_half_height,
            layer_outer_half_height,
            triangularity,
            facecolor=color,
            alpha=0.88,
            zorder=1,
        )
        if label not in legend_labels:
            legend_items.append(Patch(facecolor=color, edgecolor="white", label=label))
            legend_labels.add(label)
        inboard_layer_inner = inboard_layer_outer
        outboard_layer_inner = outboard_layer_outer
        layer_half_height = layer_outer_half_height

    plasma_r, plasma_z = _miller_boundary(
        major_radius,
        minor_radius,
        elongation,
        triangularity,
    )
    _fill_closed_polygon(
        ax,
        plasma_r,
        plasma_z,
        facecolor="#e41a1c",
        edgecolor="#8b0000",
        linewidth=1.2,
        alpha=0.78,
        zorder=2,
    )
    ax.plot(plasma_r, plasma_z, color="#8b0000", linewidth=1.1, zorder=3)
    ax.text(
        major_radius,
        0.0,
        "plasma",
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        zorder=4,
    )

    ax.axvline(
        major_radius,
        color="black",
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )
    ax.text(
        major_radius,
        plasma_half_height * 0.5,
        "R0",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    plasma_in = _as_float(geom.inboard_lcfs_radius_m)
    plasma_out = _as_float(geom.outboard_lcfs_radius_m)
    ax.axvline(plasma_in, color="#8b0000", linewidth=0.8, linestyle=":", zorder=0)
    ax.axvline(plasma_out, color="#8b0000", linewidth=0.8, linestyle=":", zorder=0)
    ax.set_ylim(-1.05 * drawing_half_height, 1.05 * drawing_half_height)
    ax.set_ylabel("Z [m]")
    ax.set_xlabel("R [m]")
    ax.set_title("Radial Build Poloidal Section")
    ax.set_xlim(0.0, pf_outer_radius * 1.03)
    ax.set_aspect("equal", adjustable="box")

    legend_items.append(Patch(facecolor="#e41a1c", edgecolor="#8b0000", label="plasma"))
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=5,
        fontsize=8,
        frameon=False,
    )


def plot_reactor_diagnostics(
    design: Any,
    technology: Any,
    result: Any,
    output_path: str | Path,
    *,
    title: str = "TokaSys Reactor Diagnostics",
) -> Path:
    """Plot profiles, radial build and key scalar diagnostics."""
    output = Path(output_path)
    rho = np.asarray(result.plasma.radial_grid, dtype=float)
    te = np.asarray(result.plasma.electron_temperature_profile_keV, dtype=float)
    ti = np.asarray(result.plasma.ion_temperature_profile_keV, dtype=float)
    ne19 = np.asarray(result.plasma.electron_density_profile_m3, dtype=float) / 1.0e19
    current = diagnostic_current_and_q_profiles(design, result)
    heating = diagnostic_heating_profiles(design, technology, result)

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    ax_temp = fig.add_subplot(grid[0, 0])
    ax_heating = fig.add_subplot(grid[0, 1])
    ax_current = fig.add_subplot(grid[1, 0])
    ax_q = fig.add_subplot(grid[1, 1])
    ax_build = fig.add_subplot(grid[0, 2])
    ax_text = fig.add_subplot(grid[1, 2])

    ped_width = _as_float(result.plasma.pedestal_width)
    ped_start = max(1.0 - ped_width, 0.0)
    ax_temp.axvspan(ped_start, 1.0, color="#fdae61", alpha=0.18, label="pedestal")
    ax_temp.axhline(
        _as_float(result.plasma.pedestal_temperature_keV),
        color="0.35",
        linestyle=":",
        linewidth=1.3,
        label="Tped",
    )
    ax_temp.plot(rho, te, label="Te", color="#d73027", linewidth=2.0)
    ax_temp.plot(rho, ti, label="Ti", color="#fc8d59", linewidth=2.0, linestyle="--")
    ax_temp.plot(rho, ne19, label="ne", color="#4575b4", linewidth=2.0)
    ax_temp.set_xlabel("Normalized minor radius rho")
    ax_temp.set_ylabel("Temperature [keV], density [1e19 m^-3]")
    ax_temp.set_title("Temperature and Density Profiles")
    ax_temp.legend(frameon=False)

    ax_heating.plot(
        rho,
        heating["external_mw_m3"],
        label="external",
        color="#1b9e77",
        linewidth=1.8,
    )
    ax_heating.plot(
        rho,
        heating["alpha_mw_m3"],
        label="alpha",
        color="#d95f02",
        linewidth=1.8,
    )
    ax_heating.plot(
        rho,
        heating["ohmic_mw_m3"],
        label="ohmic",
        color="#7570b3",
        linewidth=1.8,
    )
    ax_heating.plot(
        rho,
        -heating["radiation_loss_mw_m3"],
        label="radiation loss",
        color="#e7298a",
        linewidth=1.8,
    )
    ax_heating.axhline(0.0, color="0.35", linewidth=0.8)
    ax_heating.set_xlabel("Normalized minor radius rho")
    ax_heating.set_ylabel("Power density [MW/m3]")
    ax_heating.set_title("Heating and Radiation Profiles")
    ax_heating.legend(frameon=False)

    ax_current.plot(
        rho,
        current["total_current_density_ma_m2"],
        color="black",
        linewidth=2.0,
        label="total",
    )
    ax_current.plot(
        rho,
        current["bootstrap_current_density_ma_m2"],
        color="#1b9e77",
        linewidth=1.7,
        label="bootstrap",
    )
    ax_current.plot(
        rho,
        current["current_drive_current_density_ma_m2"],
        color="#d95f02",
        linewidth=1.7,
        label="CD",
    )
    ax_current.plot(
        rho,
        current["inductive_current_density_ma_m2"],
        color="#7570b3",
        linewidth=1.5,
        linestyle=":",
        label="inductive",
    )
    ax_current_twin = ax_current.twinx()
    ax_current_twin.plot(
        rho,
        current["enclosed_current_fraction"],
        color="#666666",
        linewidth=1.5,
        linestyle="--",
        label="I(<rho)/Ip",
    )
    '''ax_current_twin.plot(
        rho, current["q_profile"], color="#e7298a", linewidth=2.0, label='q profile'
    )'''
    ax_current.set_xlabel("Normalized minor radius rho")
    ax_current.set_ylabel("Current density [MA/m2]")
    ax_current_twin.set_ylabel("Enclosed current fraction")
    #ax_current_twin.set_ylabel("Safety factor q")
    ax_current.set_title("Diagnostic Current Profile")
    lines1, labels1 = ax_current.get_legend_handles_labels()
    lines2, labels2 = ax_current_twin.get_legend_handles_labels()
    ax_current.legend(lines1+lines2, labels1+labels2, frameon=False, loc="upper center")

    ax_q.plot(rho, current["q_profile"], color="#e7298a", linewidth=2.0)
    ax_q.axhline(_as_float(result.plasma.q95), color="0.4", linestyle=":", label="q95")
    ax_q.set_xlabel("Normalized minor radius rho")
    ax_q.set_ylabel("Safety factor q")
    ax_q.set_title("Diagnostic q Profile")
    ax_q.legend(frameon=False)

    plot_radial_build(ax_build, design, technology, result)

    ax_text.axis("off")
    summary = (
        "Evaluation Summary\n"
        f"R0 / a / A        : {_as_float(design.major_radius_m):.2f} m / "
        f"{_as_float(result.geometry.minor_radius_m):.2f} m / {_as_float(design.aspect_ratio):.2f}\n"
        f"kappa / delta     : {_as_float(design.elongation):.2f} / "
        f"{_as_float(design.triangularity):.2f}\n"
        f"B0 / Ip           : {_as_float(design.toroidal_field_t):.2f} T / "
        f"{_as_float(design.plasma_current_ma):.2f} MA\n"
        f"Pext / Q          : {_as_float(design.auxiliary_heating_mw)+_as_float(design.current_drive_power_mw):.2f} / "
        f"{_as_float(result.plasma.fusion_gain):.2f} \n"
        f"Pfus / Pnet       : {_as_float(result.plasma.fusion_power_mw):.1f} / "
        f"{_as_float(result.power.net_electric_power_mw):.1f} MW\n"
        f"Gross / Recirc    : {_as_float(result.power.gross_electric_power_mw):.1f} / "
        f"{_as_float(result.power.recirculating_power_mw):.1f} MW\n"
        f"q95 / beta_N      : {_as_float(result.plasma.q95):.2f} / "
        f"{_as_float(result.plasma.beta_n):.2f}\n"
        f"s_edge / s_95     : {_as_float(result.plasma.magnetic_shear_edge):.2f} / "
        f"{_as_float(result.plasma.magnetic_shear_95):.2f}\n"
        f"ped width / Tped  : {ped_width:.3f} / "
        f"{_as_float(result.plasma.pedestal_temperature_keV):.2f} keV\n"
        f"Ibs / ICD         : {_as_float(result.plasma.bootstrap_current_ma):.2f} / "
        f"{_as_float(result.plasma.current_drive_current_ma):.2f} MA\n"
        f"TBR / NWL         : {_as_float(result.nuclear.tbr):.2f} / "
        f"{_as_float(result.nuclear.neutron_wall_loading_mw_m2):.2f} MW/m2\n"
        f"FW / Div heat     : {_as_float(result.nuclear.first_wall_surface_heat_load_mw_m2):.2f} / "
        f"{_as_float(result.exhaust.peak_divertor_heat_flux_mw_m2):.2f} MW/m2\n"
        f"Peak TF / COE     : {_as_float(result.magnets.peak_tf_field_t):.2f} T / "
        f"{_as_float(result.economics.coe_usd_mwh):.1f} USD/MWh\n"
        f"Availability      : {_as_float(result.economics.availability):.3f}\n"
        f"Pulse duty factor : {_as_float(result.power.pulse_duty_factor):.3f}"
    )
    ax_text.text(
        0.02,
        0.98,
        summary,
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f7f7f7",
            "edgecolor": "#cccccc",
        },
    )

    for ax in [ax_temp, ax_heating, ax_current, ax_q]:
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
