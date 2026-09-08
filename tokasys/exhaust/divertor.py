"""Reduced power-exhaust and divertor model.

Refs:
    Eich et al., Nucl. Fusion 53, 093031 (2013), for the H-mode SOL power-width
    scaling with poloidal magnetic field. Divertor target temperature and
    lifetime are reduced engineering surrogates for conceptual design studies.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import (
    DesignVariables,
    ExhaustState,
    GeometryState,
    PlasmaState,
    ReactorConfig,
    TechnologyParameters,
)
from tokasys.core.constants import MU0


def divertor_lambda_scaling(
    x: DesignVariables,
    geom: GeometryState,
    tech: TechnologyParameters,
    config: ReactorConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return upstream lambda_q, target integral lambda and field diagnostics."""
    b_pol = MU0 * x.plasma_current_ma * 1.0e6 / (
        2.0 * jnp.pi * geom.minor_radius_m
    )
    outboard_major_radius = x.major_radius_m + geom.minor_radius_m
    b_t_outboard = x.toroidal_field_t * x.major_radius_m / outboard_major_radius
    b_parallel = jnp.sqrt(b_t_outboard**2 + b_pol**2)

    eich_lambda = tech.divertor_lambda_q_coefficient_m * jnp.maximum(
        b_pol,
        1.0e-6,
    ) ** tech.divertor_lambda_q_bpol_exponent
    lambda_q = jnp.where(
        config.divertor_heat_load_model_id == 0,
        tech.divertor_lambda_integral_m,
        eich_lambda,
    )
    lambda_integral = jnp.where(
        config.divertor_heat_load_model_id == 0,
        tech.divertor_lambda_integral_m,
        jnp.sqrt(lambda_q**2 + tech.divertor_spreading_factor_m**2),
    )
    return lambda_q, lambda_integral, b_pol, b_parallel


def evaluate_exhaust(
    x: DesignVariables,
    geom: GeometryState,
    plasma: PlasmaState,
    tech: TechnologyParameters,
    config: ReactorConfig,
) -> ExhaustState:
    """Evaluate separatrix power deposition, divertor heat flux, and lifetime."""
    # The transport loss is the net power crossing the separatrix in this 0D closure.
    p_sep = plasma.transport_loss_mw
    radiated_power = tech.divertor_radiated_fraction * p_sep
    target_power = p_sep - radiated_power
    nuclear_heat = tech.divertor_neutron_heat_fraction * plasma.neutron_power_mw
    radiation_heat = tech.divertor_radiation_heat_fraction * plasma.radiation_power_mw
    heat_deposited = p_sep + nuclear_heat + radiation_heat
    lambda_q, lambda_integral, b_pol, b_parallel = divertor_lambda_scaling(
        x=x,
        geom=geom,
        tech=tech,
        config=config,
    )
    parallel_heat_flux = target_power / jnp.maximum(
        tech.divertor_n_targets
        * 2.0
        * jnp.pi
        * x.major_radius_m
        * lambda_q,
        1.0e-9,
    )
    target_area = (
        tech.divertor_n_targets
        * 2.0
        * jnp.pi
        * x.major_radius_m
        * lambda_integral
        * tech.divertor_flux_expansion
    )
    integral_flux = target_power / jnp.maximum(target_area, 1.0e-9)
    peak_flux = tech.divertor_peak_to_integral_factor * integral_flux
    heat_flux_utilization = peak_flux / jnp.maximum(
        tech.divertor_heat_flux_limit_mw_m2,
        1.0e-9,
    )
    target_temperature = (
        tech.divertor_coolant_temperature_c
        + tech.divertor_thermal_resistance_c_per_mw_m2
        * jnp.maximum(peak_flux, 0.0) ** tech.divertor_temperature_heat_flux_exponent
    )
    temperature_utilization = target_temperature / jnp.maximum(
        tech.divertor_target_temperature_limit_c,
        1.0e-9,
    )
    lifetime_utilization = jnp.maximum(
        jnp.sqrt(0.70 * heat_flux_utilization**2 + 0.30 * temperature_utilization**2),
        0.05,
    )
    divertor_lifetime = tech.divertor_reference_lifetime_fpy * (
        lifetime_utilization ** (-tech.divertor_lifetime_heat_flux_exponent)
    )

    return ExhaustState(
        separatrix_power_mw=p_sep,
        divertor_radiated_power_mw=radiated_power,
        divertor_target_power_mw=target_power,
        divertor_nuclear_heat_mw=nuclear_heat,
        divertor_radiation_heat_mw=radiation_heat,
        divertor_heat_deposited_mw=heat_deposited,
        divertor_poloidal_field_t=b_pol,
        divertor_parallel_field_t=b_parallel,
        divertor_upstream_lambda_q_m=lambda_q,
        divertor_integral_lambda_m=lambda_integral,
        divertor_parallel_heat_flux_mw_m2=parallel_heat_flux,
        divertor_flux_expansion=tech.divertor_flux_expansion,
        divertor_peak_to_integral_factor=tech.divertor_peak_to_integral_factor,
        divertor_effective_area_m2=target_area,
        divertor_integral_heat_flux_mw_m2=integral_flux,
        peak_divertor_heat_flux_mw_m2=peak_flux,
        divertor_target_temperature_c=target_temperature,
        divertor_lifetime_fpy=divertor_lifetime,
    )
