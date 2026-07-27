"""Central-solenoid flux-swing and stress requirements.

Refs:
    PROCESS engineering practice and tokamak startup volt-second accounting as
    summarized in Kovari et al., Fusion Eng. Des. 104, 9-20 (2016).

Notes:
    Flux coupling, winding area, peak field and stress are reduced constraints.
    The CS volt-second capacity is modeled as a linked poloidal-flux swing,
    proportional to the solenoid bore/enclosed area rather than to coil height.
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
from tokasys.magnets.pf import (
    required_equilibrium_ampere_turns_a,
    vertical_field_requirement_t,
)
from tokasys.magnets.superconductors import winding_pack_current_density_limit


def cs_pf_requirements(
    x: DesignVariables,
    geom: GeometryState,
    plasma: PlasmaState,
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
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Reduced PF equilibrium and CS volt-second/startup requirements."""
    r0 = x.major_radius_m
    a = geom.minor_radius_m
    ip_a = x.plasma_current_ma * 1.0e6
    li = tech.plasma_internal_inductance

    vertical_field = vertical_field_requirement_t(x, geom, plasma, tech)
    pf_required_ampere_turns = required_equilibrium_ampere_turns_a(
        vertical_field,
        geom,
        tech,
    )

    plasma_inductance = MU0 * r0 * (
        jnp.log(8.0 * r0 / jnp.maximum(a, 1.0e-9)) - 2.0 + 0.5 * li
    )
    current_ramp_flux = plasma_inductance * ip_a
    loop_voltage = plasma.ohmic_power_mw * 1.0e6 / jnp.maximum(ip_a, 1.0e-9)
    startup_flux = current_ramp_flux + 0.5 * loop_voltage * tech.cs_startup_duration_s
    flat_top_flux = jnp.where(
        config.steady_state,
        0.0,
        loop_voltage * tech.cs_pulse_flat_top_duration_s,
    )
    required_flux = tech.cs_flux_margin * (startup_flux + flat_top_flux)

    cs_outer_radius = jnp.maximum(x.cs_thickness_m, 0.25)
    cs_mean_radius = 0.5 * cs_outer_radius
    cs_height = jnp.maximum(geom.tf_coil_height_m, 1.0e-6)
    cs_winding_area = x.cs_thickness_m * cs_height
    cs_linked_flux_area = jnp.pi * cs_outer_radius**2
    coupling = jnp.maximum(tech.cs_flux_swing_coupling, 1.0e-6)

    cs_peak_field = required_flux / jnp.maximum(
        2.0 * coupling * cs_linked_flux_area,
        1.0e-9,
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
        cs_peak_field,
        tech,
        config,
        superconductor_id=config.cs_superconductor_id,
    )
    cs_current_density_limit = material_allowable_j
    cs_required_ampere_turns = cs_peak_field * cs_height / MU0
    cs_available_ampere_turns = cs_current_density_limit * cs_winding_area
    cs_peak_from_current_limit = MU0 * cs_available_ampere_turns / cs_height
    cs_capacity_peak_field = jnp.minimum(
        tech.cs_peak_field_limit_t,
        cs_peak_from_current_limit,
    )
    cs_flux_capacity = (
        2.0 * coupling * cs_capacity_peak_field * cs_linked_flux_area
    )
    cs_current_density = cs_required_ampere_turns / jnp.maximum(cs_winding_area, 1.0e-9)
    cs_current_density_utilization = cs_current_density / jnp.maximum(
        cs_current_density_limit,
        1.0e-9,
    )
    cs_structural_fraction = jnp.clip(tech.cs_structural_fraction, 0.05, 0.95)
    cs_stress = (
        cs_peak_field**2
        / (2.0 * MU0)
        * cs_mean_radius
        / jnp.maximum(cs_structural_fraction * x.cs_thickness_m, 1.0e-6)
    )

    return (
        vertical_field,
        pf_required_ampere_turns,
        cs_required_ampere_turns,
        cs_available_ampere_turns,
        startup_flux,
        flat_top_flux,
        required_flux,
        cs_flux_capacity,
        cs_peak_field,
        cs_current_density,
        cs_current_density_limit,
        critical_j,
        superconductor_temperature,
        superconductor_strain,
        field_derating,
        temperature_derating,
        strain_derating,
        cs_current_density_utilization,
        cs_stress,
        loop_voltage,
    )
