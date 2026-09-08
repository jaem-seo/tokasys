"""Top-level differentiable coupled reactor evaluator.

Refs:
    PROCESS systems-code formulation: Kovari et al., Fusion Eng. Des. 89,
    3054-3069 (2014), and 104, 9-20 (2016). TokaSys differs by keeping the
    coupled 0D/profile model JAX-differentiable and exposing both full-space
    and self-consistent energy-closure modes.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import (
    ClosureVariables,
    DesignVariables,
    MarginState,
    NumericalOptions,
    ReactorConfig,
    ReactorResult,
    ReactorVariables,
    ResidualState,
    TechnologyParameters,
)
from tokasys.exhaust.divertor import evaluate_exhaust
from tokasys.geometry.plasma import evaluate_geometry
from tokasys.magnets.tf import evaluate_magnets
from tokasys.nuclear.blanket import evaluate_nuclear_island
from tokasys.plant.economics import evaluate_economics
from tokasys.plant.power import evaluate_power_plant
from tokasys.plasma.physics import evaluate_plasma


def evaluate_reactor(
    x: DesignVariables | ReactorVariables,
    tech: TechnologyParameters,
    config: ReactorConfig,
    numerics: NumericalOptions | None = None,
) -> ReactorResult:
    """Evaluate the coupled reactor and construct optimizer residuals and margins.

    The function accepts either design variables alone for a forward evaluation
    or design plus closure variables for a full-space optimization evaluation.
    It returns all geometry, plasma, magnet, nuclear, exhaust, power, economics,
    equality-residual, and inequality-margin states in one immutable result.
    """
    if numerics is None:
        numerics = NumericalOptions()

    closure = None
    if isinstance(x, ReactorVariables):
        closure = x.closure
        x = x.design

    geometry = evaluate_geometry(x, tech, numerics)
    plasma = evaluate_plasma(x, geometry, tech, config, closure)
    nuclear = evaluate_nuclear_island(x, geometry, plasma, tech, config)
    magnets = evaluate_magnets(x, geometry, plasma, nuclear, tech, config)
    exhaust = evaluate_exhaust(x, geometry, plasma, tech, config)
    power = evaluate_power_plant(
        x,
        geometry,
        plasma,
        magnets,
        nuclear,
        exhaust,
        tech,
        config,
    )
    economics = evaluate_economics(
        x,
        geometry,
        plasma,
        magnets,
        nuclear,
        exhaust,
        power,
        tech,
        config,
    )

    # Equality residuals are normalized and should be driven to zero.
    plasma_power_input = (
        plasma.alpha_power_mw
        + x.auxiliary_heating_mw
        + x.current_drive_power_mw
        + plasma.ohmic_power_mw
    )
    plasma_power_output = plasma.radiation_power_mw + plasma.transport_loss_mw
    power_scale = max(tech.net_electric_target_mw, 100.0)
    power_balance_residual = (plasma_power_input - plasma_power_output) / power_scale

    current_balance_residual = (
        x.plasma_current_ma
        - plasma.bootstrap_current_ma
        - plasma.current_drive_current_ma
    ) / x.plasma_current_ma

    target_net_power_residual = (
        power.net_electric_power_mw - tech.net_electric_target_mw
    ) / tech.net_electric_target_mw

    model_closure = ClosureVariables(
        stored_energy_mj=plasma.stored_energy_mj,
        transport_loss_mw=plasma.ipb98_transport_loss_mw,
        bootstrap_current_ma=plasma.bootstrap_current_ma,
        current_drive_current_ma=plasma.current_drive_current_ma,
        net_electric_power_mw=power.net_electric_power_mw,
        tf_peak_field_t=magnets.peak_tf_field_t,
        tbr=nuclear.tbr,
        divertor_heat_flux_mw_m2=exhaust.peak_divertor_heat_flux_mw_m2,
    )
    if closure is None:
        closure = model_closure

    stored_energy_closure = (
        closure.stored_energy_mj - model_closure.stored_energy_mj
    ) / jnp.maximum(model_closure.stored_energy_mj, 1.0e-9)
    transport_loss_closure = (
        closure.transport_loss_mw - model_closure.transport_loss_mw
    ) / jnp.maximum(model_closure.transport_loss_mw, 1.0e-9)
    bootstrap_current_closure = (
        closure.bootstrap_current_ma - model_closure.bootstrap_current_ma
    ) / jnp.maximum(x.plasma_current_ma, 1.0e-9)
    current_drive_current_closure = (
        closure.current_drive_current_ma - model_closure.current_drive_current_ma
    ) / jnp.maximum(x.plasma_current_ma, 1.0e-9)
    net_electric_power_closure = (
        closure.net_electric_power_mw - model_closure.net_electric_power_mw
    ) / jnp.maximum(tech.net_electric_target_mw, 100.0)
    tf_peak_field_closure = (
        closure.tf_peak_field_t - model_closure.tf_peak_field_t
    ) / jnp.maximum(model_closure.tf_peak_field_t, 1.0e-9)
    tbr_closure = (closure.tbr - model_closure.tbr) / jnp.maximum(
        model_closure.tbr,
        1.0e-9,
    )
    divertor_heat_flux_closure = (
        closure.divertor_heat_flux_mw_m2 - model_closure.divertor_heat_flux_mw_m2
    ) / jnp.maximum(model_closure.divertor_heat_flux_mw_m2, 1.0e-9)

    residuals = ResidualState(
        plasma_power_balance=power_balance_residual,
        plasma_current_balance=current_balance_residual,
        target_net_power=target_net_power_residual,
        stored_energy_closure=stored_energy_closure,
        transport_loss_closure=transport_loss_closure,
        bootstrap_current_closure=bootstrap_current_closure,
        current_drive_current_closure=current_drive_current_closure,
        net_electric_power_closure=net_electric_power_closure,
        tf_peak_field_closure=tf_peak_field_closure,
        tbr_closure=tbr_closure,
        divertor_heat_flux_closure=divertor_heat_flux_closure,
    )

    # Every margin is dimensionless and positive when feasible.
    margins = MarginState(
        q95=(plasma.q95 - tech.q95_min) / tech.q95_min,
        beta_n=(tech.beta_n_limit - plasma.beta_n) / tech.beta_n_limit,
        greenwald=(tech.greenwald_fraction_max - x.greenwald_fraction)
        / tech.greenwald_fraction_max,
        fusion_gain=(plasma.fusion_gain - tech.fusion_gain_min) / tech.fusion_gain_min,
        target_net_power=(
            power.net_electric_power_mw - tech.net_electric_target_mw
        )
        / tech.net_electric_target_mw,
        inboard_space=(geometry.inboard_tf_inner_radius_m - x.cs_thickness_m)
        / x.major_radius_m,
        tf_peak_field=(tech.tf_peak_field_limit_t - magnets.peak_tf_field_t)
        / tech.tf_peak_field_limit_t,
        tf_current_density=(
            magnets.tf_allowable_engineering_current_density_a_m2
            - magnets.tf_engineering_current_density_a_m2
        )
        / magnets.tf_allowable_engineering_current_density_a_m2,
        tf_stress=(tech.tf_allowable_stress_pa - magnets.tf_stress_pa)
        / tech.tf_allowable_stress_pa,
        tf_nuclear_heat=(tech.tf_nuclear_heat_limit_mw - magnets.tf_nuclear_heat_mw)
        / tech.tf_nuclear_heat_limit_mw,
        pf_equilibrium_ampere_turns=(
            tech.pf_max_equilibrium_ampere_turns_a
            - magnets.pf_required_equilibrium_ampere_turns_a
        )
        / tech.pf_max_equilibrium_ampere_turns_a,
        pf_current_density=1.0 - magnets.pf_current_density_utilization,
        pf_peak_field=(tech.pf_peak_field_limit_t - magnets.pf_max_peak_field_t)
        / tech.pf_peak_field_limit_t,
        pf_stress=(tech.pf_allowable_stress_pa - magnets.pf_max_stress_pa)
        / tech.pf_allowable_stress_pa,
        cs_flux_swing=(
            magnets.cs_flux_capacity_wb - magnets.cs_flux_required_wb
        )
        / magnets.cs_flux_required_wb,
        cs_peak_field=(tech.cs_peak_field_limit_t - magnets.cs_peak_field_t)
        / tech.cs_peak_field_limit_t,
        cs_current_density=1.0 - magnets.cs_current_density_utilization,
        cs_stress=(tech.cs_allowable_stress_pa - magnets.cs_stress_pa)
        / tech.cs_allowable_stress_pa,
        tbr=(nuclear.tbr - tech.tbr_min) / tech.tbr_min,
        tf_fluence=(tech.tf_fluence_limit_n_m2 - nuclear.tf_fluence_n_m2)
        / tech.tf_fluence_limit_n_m2,
        tf_dpa=(tech.tf_dpa_limit - nuclear.tf_dpa_proxy) / tech.tf_dpa_limit,
        tritium_doubling_time=(
            tech.tritium_doubling_time_max_years
            - nuclear.tritium_doubling_time_years
        )
        / tech.tritium_doubling_time_max_years,
        first_wall_heat_load=(
            tech.first_wall_surface_heat_load_limit_mw_m2
            - nuclear.first_wall_surface_heat_load_mw_m2
        )
        / tech.first_wall_surface_heat_load_limit_mw_m2,
        divertor_heat_flux=(
            tech.divertor_heat_flux_limit_mw_m2
            - exhaust.peak_divertor_heat_flux_mw_m2
        )
        / tech.divertor_heat_flux_limit_mw_m2,
        divertor_target_temperature=(
            tech.divertor_target_temperature_limit_c
            - exhaust.divertor_target_temperature_c
        )
        / tech.divertor_target_temperature_limit_c,
        divertor_lifetime=(exhaust.divertor_lifetime_fpy - tech.divertor_lifetime_min_fpy)
        / tech.divertor_lifetime_min_fpy,
        recirculating_fraction=(
            tech.recirculating_fraction_max - power.recirculating_fraction
        )
        / tech.recirculating_fraction_max,
    )

    return ReactorResult(
        geometry=geometry,
        plasma=plasma,
        magnets=magnets,
        nuclear=nuclear,
        exhaust=exhaust,
        power=power,
        economics=economics,
        residuals=residuals,
        margins=margins,
    )
