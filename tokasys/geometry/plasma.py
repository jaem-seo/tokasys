"""Top-level analytic plasma geometry evaluator."""

from __future__ import annotations

from tokasys.core.types import (
    DesignVariables,
    GeometryState,
    NumericalOptions,
    TechnologyParameters,
)
from tokasys.geometry.radial_build import radial_build_ledger
from tokasys.geometry.volumes import profile_volume_elements, shaped_plasma_volumes


def evaluate_geometry(
    x: DesignVariables,
    tech: TechnologyParameters,
    numerics: NumericalOptions,
) -> GeometryState:
    r0 = x.major_radius_m
    a = r0 / x.aspect_ratio
    eps = a / r0

    volume_geometry = shaped_plasma_volumes(
        major_radius_m=r0,
        minor_radius_m=a,
        elongation=x.elongation,
        triangularity=x.triangularity,
    )
    build = radial_build_ledger(x, tech, a)
    rho, volume_prime, shell_volumes, volume_weights = profile_volume_elements(
        major_radius_m=r0,
        minor_radius_m=a,
        elongation=x.elongation,
        num_points=numerics.profile_num_points,
    )

    return GeometryState(
        minor_radius_m=a,
        inverse_aspect_ratio=eps,
        plasma_cross_section_m2=volume_geometry.cross_section_m2,
        plasma_volume_m3=volume_geometry.volume_m3,
        plasma_surface_area_m2=volume_geometry.surface_area_m2,
        first_wall_area_m2=volume_geometry.first_wall_area_m2,
        inboard_first_wall_area_m2=volume_geometry.inboard_first_wall_area_m2,
        outboard_first_wall_area_m2=volume_geometry.outboard_first_wall_area_m2,
        inboard_lcfs_radius_m=build.inboard_lcfs_radius_m,
        outboard_lcfs_radius_m=build.outboard_lcfs_radius_m,
        inboard_sol_width_m=build.inboard_sol_width_m,
        outboard_sol_width_m=build.outboard_sol_width_m,
        inboard_first_wall_thickness_m=build.inboard_first_wall_thickness_m,
        outboard_first_wall_thickness_m=build.outboard_first_wall_thickness_m,
        inboard_blanket_thickness_m=build.inboard_blanket_thickness_m,
        outboard_blanket_thickness_m=build.outboard_blanket_thickness_m,
        inboard_shield_thickness_m=build.inboard_shield_thickness_m,
        outboard_shield_thickness_m=build.outboard_shield_thickness_m,
        inboard_vacuum_vessel_thickness_m=build.inboard_vacuum_vessel_thickness_m,
        outboard_vacuum_vessel_thickness_m=build.outboard_vacuum_vessel_thickness_m,
        inboard_thermal_gap_m=build.inboard_thermal_gap_m,
        outboard_thermal_gap_m=build.outboard_thermal_gap_m,
        inboard_tf_coil_thickness_m=build.inboard_tf_coil_thickness_m,
        outboard_tf_coil_thickness_m=build.outboard_tf_coil_thickness_m,
        inboard_non_tf_build_m=build.inboard_non_tf_build_m,
        outboard_non_tf_build_m=build.outboard_non_tf_build_m,
        inboard_total_build_m=build.inboard_total_build_m,
        outboard_total_build_m=build.outboard_total_build_m,
        inboard_plasma_edge_m=build.inboard_plasma_edge_m,
        outboard_first_wall_radius_m=build.outboard_first_wall_radius_m,
        inboard_tf_inner_radius_m=build.inboard_tf_inner_radius_m,
        inboard_tf_centroid_radius_m=build.inboard_tf_centroid_radius_m,
        inboard_tf_outer_radius_m=build.inboard_tf_outer_radius_m,
        outboard_tf_inner_radius_m=build.outboard_tf_inner_radius_m,
        outboard_tf_centroid_radius_m=build.outboard_tf_centroid_radius_m,
        outboard_tf_outer_radius_m=build.outboard_tf_outer_radius_m,
        tf_coil_height_m=build.tf_coil_height_m,
        machine_outer_radius_m=build.machine_outer_radius_m,
        radial_grid=rho,
        volume_prime_m3=volume_prime,
        shell_volumes_m3=shell_volumes,
        volume_weights=volume_weights,
    )


__all__ = ["evaluate_geometry", "profile_volume_elements"]
