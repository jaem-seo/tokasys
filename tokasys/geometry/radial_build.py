"""Explicit inboard/outboard radial-build ledger.

Refs:
    PROCESS engineering model papers, Kovari et al., Fusion Eng. Des. 89,
    3054-3069 (2014), and Kovari et al., Fusion Eng. Des. 104, 9-20 (2016),
    for system-code radial-build accounting practice.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from tokasys.core.types import DesignVariables, TechnologyParameters


class RadialBuild(NamedTuple):
    inboard_lcfs_radius_m: jnp.ndarray
    outboard_lcfs_radius_m: jnp.ndarray
    inboard_sol_width_m: jnp.ndarray
    outboard_sol_width_m: jnp.ndarray
    inboard_first_wall_thickness_m: jnp.ndarray
    outboard_first_wall_thickness_m: jnp.ndarray
    inboard_blanket_thickness_m: jnp.ndarray
    outboard_blanket_thickness_m: jnp.ndarray
    inboard_shield_thickness_m: jnp.ndarray
    outboard_shield_thickness_m: jnp.ndarray
    inboard_vacuum_vessel_thickness_m: jnp.ndarray
    outboard_vacuum_vessel_thickness_m: jnp.ndarray
    inboard_thermal_gap_m: jnp.ndarray
    outboard_thermal_gap_m: jnp.ndarray
    inboard_tf_coil_thickness_m: jnp.ndarray
    outboard_tf_coil_thickness_m: jnp.ndarray
    inboard_non_tf_build_m: jnp.ndarray
    outboard_non_tf_build_m: jnp.ndarray
    inboard_total_build_m: jnp.ndarray
    outboard_total_build_m: jnp.ndarray
    inboard_plasma_edge_m: jnp.ndarray
    outboard_first_wall_radius_m: jnp.ndarray
    inboard_tf_inner_radius_m: jnp.ndarray
    inboard_tf_centroid_radius_m: jnp.ndarray
    inboard_tf_outer_radius_m: jnp.ndarray
    outboard_tf_inner_radius_m: jnp.ndarray
    outboard_tf_centroid_radius_m: jnp.ndarray
    outboard_tf_outer_radius_m: jnp.ndarray
    tf_coil_height_m: jnp.ndarray
    machine_outer_radius_m: jnp.ndarray


def radial_build_ledger(
    x: DesignVariables,
    tech: TechnologyParameters,
    minor_radius_m: jnp.ndarray,
) -> RadialBuild:
    """Construct the inboard and outboard component radii from layer thicknesses."""
    inboard_lcfs_radius = x.major_radius_m - minor_radius_m
    outboard_lcfs_radius = x.major_radius_m + minor_radius_m

    inboard_sol_width = tech.sol_width_m
    outboard_sol_width = tech.sol_width_m
    inboard_first_wall = x.inboard_first_wall_thickness_m
    outboard_first_wall = x.outboard_first_wall_thickness_m
    inboard_blanket = x.inboard_blanket_thickness_m
    outboard_blanket = x.outboard_blanket_thickness_m
    inboard_shield = x.inboard_shield_thickness_m
    outboard_shield = x.outboard_shield_thickness_m
    inboard_vacuum_vessel = tech.vacuum_vessel_thickness_m
    outboard_vacuum_vessel = tech.vacuum_vessel_thickness_m
    inboard_thermal_gap = tech.thermal_gap_m
    outboard_thermal_gap = tech.thermal_gap_m
    inboard_tf_thickness = x.tf_coil_thickness_m
    outboard_tf_thickness = x.tf_coil_thickness_m

    inboard_non_tf_build = (
        inboard_sol_width
        + inboard_first_wall
        + inboard_blanket
        + inboard_shield
        + inboard_vacuum_vessel
        + inboard_thermal_gap
    )
    outboard_non_tf_build = (
        outboard_sol_width
        + outboard_first_wall
        + outboard_blanket
        + outboard_shield
        + outboard_vacuum_vessel
        + outboard_thermal_gap
    )
    inboard_total_build = inboard_non_tf_build + inboard_tf_thickness
    outboard_total_build = outboard_non_tf_build + outboard_tf_thickness

    inboard_first_wall_radius = inboard_lcfs_radius - inboard_sol_width
    outboard_first_wall_radius = outboard_lcfs_radius + outboard_sol_width
    inboard_tf_outer_radius = inboard_lcfs_radius - inboard_non_tf_build
    inboard_tf_centroid = inboard_tf_outer_radius - 0.5 * inboard_tf_thickness
    inboard_tf_inner_radius = inboard_tf_outer_radius - inboard_tf_thickness

    outboard_tf_inner_radius = outboard_lcfs_radius + outboard_non_tf_build
    outboard_tf_centroid = outboard_tf_inner_radius + 0.5 * outboard_tf_thickness
    outboard_tf_outer_radius = outboard_tf_inner_radius + outboard_tf_thickness
    machine_outer_radius = outboard_lcfs_radius + outboard_total_build
    tf_height = 2.0 * (x.elongation * minor_radius_m + outboard_total_build)

    return RadialBuild(
        inboard_lcfs_radius_m=inboard_lcfs_radius,
        outboard_lcfs_radius_m=outboard_lcfs_radius,
        inboard_sol_width_m=inboard_sol_width,
        outboard_sol_width_m=outboard_sol_width,
        inboard_first_wall_thickness_m=inboard_first_wall,
        outboard_first_wall_thickness_m=outboard_first_wall,
        inboard_blanket_thickness_m=inboard_blanket,
        outboard_blanket_thickness_m=outboard_blanket,
        inboard_shield_thickness_m=inboard_shield,
        outboard_shield_thickness_m=outboard_shield,
        inboard_vacuum_vessel_thickness_m=inboard_vacuum_vessel,
        outboard_vacuum_vessel_thickness_m=outboard_vacuum_vessel,
        inboard_thermal_gap_m=inboard_thermal_gap,
        outboard_thermal_gap_m=outboard_thermal_gap,
        inboard_tf_coil_thickness_m=inboard_tf_thickness,
        outboard_tf_coil_thickness_m=outboard_tf_thickness,
        inboard_non_tf_build_m=inboard_non_tf_build,
        outboard_non_tf_build_m=outboard_non_tf_build,
        inboard_total_build_m=inboard_total_build,
        outboard_total_build_m=outboard_total_build,
        inboard_plasma_edge_m=inboard_first_wall_radius,
        outboard_first_wall_radius_m=outboard_first_wall_radius,
        inboard_tf_inner_radius_m=inboard_tf_inner_radius,
        inboard_tf_centroid_radius_m=inboard_tf_centroid,
        inboard_tf_outer_radius_m=inboard_tf_outer_radius,
        outboard_tf_inner_radius_m=outboard_tf_inner_radius,
        outboard_tf_centroid_radius_m=outboard_tf_centroid,
        outboard_tf_outer_radius_m=outboard_tf_outer_radius,
        tf_coil_height_m=tf_height,
        machine_outer_radius_m=machine_outer_radius,
    )
