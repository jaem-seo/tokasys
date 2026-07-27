"""Poloidal-field equilibrium-current requirements.

Refs:
    Freidberg, Ideal MHD / tokamak equilibrium treatments, and PROCESS-style
    PF engineering constraints. Coil-by-coil placement is a differentiable
    reduced layout, not a free-boundary equilibrium/control-coil optimization.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.constants import MU0
from tokasys.core.types import (
    DesignVariables,
    GeometryState,
    PlasmaState,
    ReactorConfig,
    TechnologyParameters,
)
from tokasys.magnets.superconductors import winding_pack_current_density_limit


def vertical_field_requirement_t(
    x: DesignVariables,
    geom: GeometryState,
    plasma: PlasmaState,
    tech: TechnologyParameters,
) -> jnp.ndarray:
    """Reduced vertical-field requirement for radial force balance."""
    return (
        MU0
        * x.plasma_current_ma
        * 1.0e6
        / (4.0 * jnp.pi * x.major_radius_m)
        * jnp.maximum(
            jnp.log(8.0 * x.major_radius_m / jnp.maximum(geom.minor_radius_m, 1.0e-9))
            + plasma.beta_p
            + 0.5 * tech.plasma_internal_inductance
            - 1.5,
            0.05,
        )
    )


def required_equilibrium_ampere_turns_a(
    vertical_field_t: jnp.ndarray,
    geom: GeometryState,
    tech: TechnologyParameters,
) -> jnp.ndarray:
    """PF ampere-turn proxy needed to supply the required vertical field."""
    return (
        tech.pf_vertical_field_coil_factor
        * vertical_field_t
        * geom.machine_outer_radius_m
        / MU0
    )


def pf_coil_placement_and_loads(
    x: DesignVariables,
    geom: GeometryState,
    vertical_field_t: jnp.ndarray,
    required_ampere_turns_a: jnp.ndarray,
    tech: TechnologyParameters,
    config: ReactorConfig,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Place one inboard and two outboard symmetric PF coil pairs.

    This reduced placement model keeps a differentiable fixed-size PF set while
    exposing the engineering quantities optimizers need: per-coil current
    density, peak field and magnetic-pressure stress.
    """
    pair_weights = jnp.asarray([0.30, 0.45, 0.25], dtype=float)

    half_height = 0.5 * geom.tf_coil_height_m
    plasma_half_height = x.elongation * geom.minor_radius_m
    radial_thickness = jnp.maximum(x.pf_coil_thickness_m, 1.0e-6)
    vertical_thickness = radial_thickness
    radial_gap = jnp.maximum(tech.pf_coil_radial_gap_m, 0.0)
    half_radial_thickness = 0.5 * radial_thickness
    half_vertical_thickness = 0.5 * vertical_thickness
    coil_area = radial_thickness * vertical_thickness * jnp.maximum(
        tech.pf_winding_pack_fraction,
        1.0e-6,
    )

    inboard_lower_radius = (
        geom.inboard_tf_outer_radius_m
        + radial_gap
        + half_radial_thickness
    )
    inboard_upper_radius = (
        geom.inboard_lcfs_radius_m
        - radial_gap
        - half_radial_thickness
    )
    inboard_radius = jnp.where(
        inboard_upper_radius > inboard_lower_radius,
        0.5 * (inboard_lower_radius + inboard_upper_radius),
        inboard_lower_radius,
    )
    outboard_radii = (
        geom.outboard_tf_outer_radius_m
        + radial_gap
        + half_radial_thickness
        + tech.pf_coil_pair_radial_step_m * jnp.arange(2.0)
    )

    inboard_z_abs = jnp.minimum(
        plasma_half_height + radial_gap + half_vertical_thickness,
        0.92 * half_height,
    )
    outboard_z_abs = jnp.asarray(
        [
            0.55 * plasma_half_height,
            jnp.minimum(
                plasma_half_height + radial_gap + half_vertical_thickness,
                0.92 * half_height,
            ),
        ],
        dtype=float,
    )

    coil_radii = jnp.concatenate(
        (
            jnp.repeat(inboard_radius, 2),
            jnp.repeat(outboard_radii, 2),
        )
    )
    coil_z = jnp.concatenate(
        (
            jnp.asarray([inboard_z_abs, -inboard_z_abs], dtype=float),
            jnp.ravel(jnp.stack((outboard_z_abs, -outboard_z_abs), axis=1)),
        )
    )
    coil_pair_weights = jnp.repeat(pair_weights, 2)
    coil_ampere_turns = 0.5 * coil_pair_weights * required_ampere_turns_a

    current_density = jnp.abs(coil_ampere_turns) / coil_area
    self_field = MU0 * jnp.abs(coil_ampere_turns) / (
        2.0 * jnp.maximum(radial_thickness + vertical_thickness, 1.0e-6)
    )
    toroidal_field = x.toroidal_field_t * x.major_radius_m / jnp.maximum(coil_radii, 1.0e-6)
    peak_field = jnp.sqrt(
        toroidal_field**2 + self_field**2 + vertical_field_t**2
    )
    (
        material_allowable_j,
        critical_j,
        superconductor_temperature,
        superconductor_strain,
        field_derating,
        temperature_derating,
        strain_derating,
    ) = winding_pack_current_density_limit(
        peak_field,
        tech,
        config,
        superconductor_id=config.pf_superconductor_id,
    )
    current_density_limit = material_allowable_j
    current_density_utilization = jnp.max(
        current_density / jnp.maximum(current_density_limit, 1.0e-9)
    )
    magnetic_pressure = peak_field**2 / (2.0 * MU0)
    stress = (
        magnetic_pressure
        * coil_radii
        / jnp.maximum(tech.pf_structural_fraction * radial_thickness, 1.0e-6)
    )

    return (
        coil_radii,
        coil_z,
        coil_ampere_turns,
        current_density,
        peak_field,
        stress,
        jnp.max(current_density),
        jnp.min(current_density_limit),
        jnp.min(critical_j),
        superconductor_temperature,
        superconductor_strain,
        jnp.min(field_derating),
        jnp.min(temperature_derating),
        jnp.min(strain_derating),
        current_density_utilization,
        jnp.max(peak_field),
        jnp.max(stress),
    )
