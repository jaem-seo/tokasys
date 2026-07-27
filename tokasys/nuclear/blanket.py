"""0D blanket, shielding and tritium breeding models.

Refs:
    PROCESS engineering-system-code practice: Kovari et al., Fusion Eng. Des.
    104, 9-20 (2016). Blanket concepts, TBR, attenuation, TF fluence/dpa and
    tritium doubling time are reduced surrogates intended for differentiable
    design-space exploration, not substitutes for neutronics inventory codes.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.constants import (
    DT_FUSION_ENERGY_J,
    DT_NEUTRON_ENERGY_J,
    SECONDS_PER_YEAR,
    TRITIUM_ATOMIC_MASS_KG,
)
from tokasys.core.types import (
    DesignVariables,
    GeometryState,
    NuclearState,
    PlasmaState,
    ReactorConfig,
    TechnologyParameters,
)


def blanket_concept_parameters(
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
]:
    """Return reduced blanket-concept coefficients.

    Concept ids are intentionally compact and JAX-friendly:
    0 LiPb/WCLL-like, 1 FLiBe molten-salt, 2 HCPB ceramic-breeder. The default
    concept uses the technology parameters directly for backwards continuity;
    other concepts apply reduced relative scalings around that baseline.
    """
    concept = config.blanket_concept_id
    is_flibe = concept == 1
    is_hcpb = concept == 2

    coverage = jnp.where(
        is_flibe,
        0.88,
        jnp.where(is_hcpb, 0.82, tech.blanket_coverage_fraction),
    )
    energy_multiplication = jnp.where(
        is_flibe,
        1.22,
        jnp.where(is_hcpb, 1.10, tech.blanket_energy_multiplication),
    )
    blanket_attenuation = jnp.where(
        is_flibe,
        4.2,
        jnp.where(is_hcpb, 6.2, tech.blanket_attenuation_per_m),
    )
    shield_attenuation = jnp.where(
        is_flibe,
        7.5,
        jnp.where(is_hcpb, 8.5, tech.shield_attenuation_per_m),
    )
    tbr_asymptote = jnp.where(
        is_flibe,
        1.75,
        jnp.where(is_hcpb, 1.65, tech.tbr_asymptote),
    )
    tbr_scale_length = jnp.where(
        is_flibe,
        0.45,
        jnp.where(is_hcpb, 0.30, tech.tbr_scale_length_m),
    )
    shield_neutron_heat_fraction = jnp.where(
        is_flibe,
        0.20,
        jnp.where(is_hcpb, 0.30, 0.25),
    )
    pumping_fraction = jnp.where(
        is_flibe,
        0.018,
        jnp.where(is_hcpb, 0.035, tech.blanket_pumping_fraction),
    )
    tritium_inventory_multiplier = jnp.where(
        is_flibe,
        1.25,
        jnp.where(is_hcpb, 0.85, 1.0),
    )
    return (
        coverage,
        energy_multiplication,
        blanket_attenuation,
        shield_attenuation,
        tbr_asymptote,
        tbr_scale_length,
        shield_neutron_heat_fraction,
        pumping_fraction,
        tritium_inventory_multiplier,
    )


def evaluate_nuclear_island(
    x: DesignVariables,
    geom: GeometryState,
    plasma: PlasmaState,
    tech: TechnologyParameters,
    config: ReactorConfig,
) -> NuclearState:
    (
        coverage,
        energy_multiplication,
        blanket_attenuation_per_m,
        shield_attenuation_per_m,
        tbr_asymptote,
        tbr_scale_length,
        shield_neutron_heat_fraction,
        pumping_fraction,
        tritium_inventory_multiplier,
    ) = blanket_concept_parameters(tech, config)

    inboard_area = geom.inboard_first_wall_area_m2
    outboard_area = geom.outboard_first_wall_area_m2
    safe_first_wall_area = jnp.maximum(geom.first_wall_area_m2, 1.0e-9)
    blanket_volume = (
        inboard_area * x.inboard_blanket_thickness_m
        + outboard_area * x.outboard_blanket_thickness_m
    )
    shield_volume = (
        inboard_area * x.inboard_shield_thickness_m
        + outboard_area * x.outboard_shield_thickness_m
    )
    effective_blanket_thickness = blanket_volume / safe_first_wall_area
    blanket_attenuation = jnp.exp(
        -blanket_attenuation_per_m * x.inboard_blanket_thickness_m
    )
    shield_attenuation = jnp.exp(
        -shield_attenuation_per_m * x.inboard_shield_thickness_m
    )
    attenuation = blanket_attenuation * shield_attenuation

    blanket_thermal = energy_multiplication * plasma.neutron_power_mw
    first_wall_neutron_heat = (
        tech.first_wall_neutron_heat_fraction * plasma.neutron_power_mw
    )
    first_wall_radiation_heat = (
        tech.first_wall_radiation_heat_fraction * plasma.radiation_power_mw
    )
    first_wall_alpha_heat = tech.first_wall_alpha_loss_fraction * plasma.alpha_power_mw
    first_wall_thermal = (
        first_wall_neutron_heat
        + first_wall_radiation_heat
        + first_wall_alpha_heat
    )
    shield_thermal = (
        (1.0 - coverage) * shield_neutron_heat_fraction * plasma.neutron_power_mw
    )

    tbr = (
        coverage
        * tbr_asymptote
        * (1.0 - jnp.exp(-effective_blanket_thickness / tbr_scale_length))
    )

    first_wall_heat_load = first_wall_thermal / safe_first_wall_area
    neutron_wall_loading = plasma.neutron_power_mw / safe_first_wall_area
    tf_neutron_wall_loading = (
        tech.tf_neutron_geometric_view_factor
        * neutron_wall_loading
        * attenuation
    )
    tf_fluence = (
        tf_neutron_wall_loading
        * 1.0e6
        * SECONDS_PER_YEAR
        * tech.tf_fluence_lifetime_fpy
        / DT_NEUTRON_ENERGY_J
    )
    tf_dpa = tech.tf_dpa_per_1e25_neutrons_m2 * tf_fluence / 1.0e25

    fusion_reaction_rate = plasma.fusion_power_mw * 1.0e6 / DT_FUSION_ENERGY_J
    tritium_burn_rate = (
        fusion_reaction_rate * TRITIUM_ATOMIC_MASS_KG * SECONDS_PER_YEAR
    )
    tritium_breeding_rate = tbr * tritium_burn_rate
    tritium_surplus = (tbr - 1.0) * tritium_burn_rate
    tritium_inventory = tritium_inventory_multiplier * (
        tech.tritium_startup_inventory_kg
        + tritium_burn_rate
        * tech.tritium_processing_residence_time_days
        / 365.25
    )
    tritium_doubling_time = tritium_inventory / jnp.maximum(tritium_surplus, 1.0e-9)

    blanket_loop_thermal = blanket_thermal + first_wall_thermal + shield_thermal
    blanket_pumping = pumping_fraction * blanket_loop_thermal

    return NuclearState(
        blanket_concept_id=jnp.asarray(config.blanket_concept_id, dtype=float),
        neutron_attenuation=attenuation,
        blanket_neutron_attenuation=blanket_attenuation,
        shield_neutron_attenuation=shield_attenuation,
        blanket_thermal_power_mw=blanket_thermal,
        first_wall_neutron_heat_mw=first_wall_neutron_heat,
        first_wall_radiation_heat_mw=first_wall_radiation_heat,
        first_wall_alpha_heat_mw=first_wall_alpha_heat,
        first_wall_thermal_power_mw=first_wall_thermal,
        shield_thermal_power_mw=shield_thermal,
        first_wall_surface_heat_load_mw_m2=first_wall_heat_load,
        neutron_wall_loading_mw_m2=neutron_wall_loading,
        tf_neutron_wall_loading_mw_m2=tf_neutron_wall_loading,
        tf_fluence_n_m2=tf_fluence,
        tf_dpa_proxy=tf_dpa,
        tbr=tbr,
        tbr_asymptote=tbr_asymptote,
        tbr_scale_length_m=tbr_scale_length,
        tritium_burn_rate_kg_per_year=tritium_burn_rate,
        tritium_breeding_rate_kg_per_year=tritium_breeding_rate,
        tritium_surplus_kg_per_year=tritium_surplus,
        tritium_inventory_kg=tritium_inventory,
        tritium_doubling_time_years=tritium_doubling_time,
        blanket_pumping_power_mw=blanket_pumping,
        blanket_pumping_fraction=pumping_fraction,
        blanket_energy_multiplication=energy_multiplication,
        blanket_coverage_fraction=coverage,
        blanket_attenuation_per_m=blanket_attenuation_per_m,
        shield_attenuation_per_m=shield_attenuation_per_m,
        blanket_volume_m3=blanket_volume,
        shield_volume_m3=shield_volume,
    )
