"""Parametric plant-cost, availability and cost-of-electricity model.

Refs:
    PROCESS system-code economics/engineering practice: Kovari et al., Fusion
    Eng. Des. 104, 9-20 (2016). The availability model follows reduced
    component-lifetime/RAMI logic for conceptual studies.

Notes:
    Costs and availability are calibrated surrogates, not bottom-up estimates.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.constants import HOURS_PER_YEAR
from tokasys.core.types import (
    DesignVariables,
    EconomicsState,
    ExhaustState,
    GeometryState,
    MagnetState,
    NuclearState,
    PlasmaState,
    PowerPlantState,
    ReactorConfig,
    TechnologyParameters,
)


def evaluate_availability(
    magnets: MagnetState,
    nuclear: NuclearState,
    exhaust: ExhaustState,
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
    """Reduced component-lifetime availability model.

    The model annualizes scheduled maintenance: each high-heat/high-fluence
    component gets a smooth lifetime scaling around a reference lifetime, and
    replacement or repair days are converted to an availability factor. It is a
    reduced RAMI surrogate, not a detailed failure-rate model.
    """
    blanket_neutron_utilization = nuclear.neutron_wall_loading_mw_m2 / jnp.maximum(
        tech.blanket_replacement_reference_neutron_wall_loading_mw_m2,
        1.0e-9,
    )
    first_wall_heat_utilization = (
        nuclear.first_wall_surface_heat_load_mw_m2
        / jnp.maximum(tech.first_wall_surface_heat_load_limit_mw_m2, 1.0e-9)
    )
    blanket_utilization = jnp.sqrt(
        0.70 * blanket_neutron_utilization**2
        + 0.30 * first_wall_heat_utilization**2
    )
    tf_damage_utilization = jnp.maximum(
        nuclear.tf_dpa_proxy / jnp.maximum(tech.tf_dpa_limit, 1.0e-12),
        nuclear.tf_fluence_n_m2 / jnp.maximum(tech.tf_fluence_limit_n_m2, 1.0e-12),
    )
    tf_stress_utilization = magnets.tf_stress_pa / jnp.maximum(
        tech.tf_allowable_stress_pa,
        1.0e-9,
    )
    tf_j_utilization = magnets.tf_engineering_current_density_a_m2 / jnp.maximum(
        magnets.tf_allowable_engineering_current_density_a_m2,
        1.0e-9,
    )
    cs_field_utilization = magnets.cs_peak_field_t / jnp.maximum(
        tech.cs_peak_field_limit_t,
        1.0e-9,
    )
    cs_stress_utilization = magnets.cs_stress_pa / jnp.maximum(
        tech.cs_allowable_stress_pa,
        1.0e-9,
    )
    pf_field_utilization = magnets.pf_max_peak_field_t / jnp.maximum(
        tech.pf_peak_field_limit_t,
        1.0e-9,
    )
    pf_stress_utilization = magnets.pf_max_stress_pa / jnp.maximum(
        tech.pf_allowable_stress_pa,
        1.0e-9,
    )
    magnet_utilization = jnp.sqrt(
        0.30 * tf_damage_utilization**2
        + 0.25 * tf_stress_utilization**2
        + 0.15 * tf_j_utilization**2
        + 0.10 * cs_field_utilization**2
        + 0.10 * cs_stress_utilization**2
        + 0.05 * pf_field_utilization**2
        + 0.05 * pf_stress_utilization**2
    )

    safe_blanket_utilization = jnp.maximum(blanket_utilization, 0.05)
    safe_magnet_utilization = jnp.maximum(magnet_utilization, 0.05)
    divertor_lifetime = exhaust.divertor_lifetime_fpy
    blanket_lifetime = tech.blanket_reference_lifetime_fpy * (
        safe_blanket_utilization ** (-tech.blanket_lifetime_neutron_exponent)
    )
    magnet_lifetime = tech.magnet_reference_lifetime_fpy * (
        safe_magnet_utilization ** (-tech.magnet_lifetime_utilization_exponent)
    )
    divertor_downtime = tech.divertor_replacement_duration_days / jnp.maximum(
        divertor_lifetime,
        1.0e-6,
    )
    blanket_downtime = tech.blanket_replacement_duration_days / jnp.maximum(
        blanket_lifetime,
        1.0e-6,
    )
    magnet_downtime = tech.magnet_repair_duration_days / jnp.maximum(
        magnet_lifetime,
        1.0e-6,
    )
    divertor_factor = jnp.exp(
        -(1.0 + tech.divertor_availability_penalty) * divertor_downtime / 365.25
    )
    neutron_factor = jnp.exp(
        -(1.0 + tech.neutron_availability_penalty) * blanket_downtime / 365.25
    )
    magnet_factor = jnp.exp(
        -(1.0 + tech.magnet_availability_penalty) * magnet_downtime / 365.25
    )
    pulsed_factor = jnp.where(
        config.steady_state,
        1.0,
        1.0 - tech.pulsed_availability_penalty,
    )
    scheduled_downtime = divertor_downtime + blanket_downtime + magnet_downtime
    availability = (
        tech.availability
        * divertor_factor
        * neutron_factor
        * magnet_factor
        * pulsed_factor
    )
    availability = jnp.clip(availability, tech.availability_min, tech.availability)
    return (
        availability,
        divertor_factor,
        neutron_factor,
        magnet_factor,
        pulsed_factor,
        blanket_lifetime,
        divertor_lifetime,
        magnet_lifetime,
        scheduled_downtime,
    )


def evaluate_economics(
    x: DesignVariables,
    geom: GeometryState,
    plasma: PlasmaState,
    magnets: MagnetState,
    nuclear: NuclearState,
    exhaust: ExhaustState,
    power: PowerPlantState,
    tech: TechnologyParameters,
    config: ReactorConfig,
) -> EconomicsState:
    del plasma
    tf_material_multiplier = jnp.where(
        config.tf_superconductor_id == 1,
        tech.nb3sn_magnet_cost_multiplier,
        tech.rebco_magnet_cost_multiplier,
    )
    tf_stress_utilization = magnets.tf_stress_pa / jnp.maximum(
        tech.tf_allowable_stress_pa,
        1.0e-9,
    )
    tf_j_utilization = magnets.tf_engineering_current_density_a_m2 / jnp.maximum(
        magnets.tf_allowable_engineering_current_density_a_m2,
        1.0e-9,
    )
    tf_coil_scale = (
        (geom.machine_outer_radius_m / 10.0) ** 1.5
        * (magnets.tf_stored_energy_gj / 50.0 + 0.2) ** 0.35
        * (1.0 + 0.08 * tf_stress_utilization + 0.04 * tf_j_utilization)
    )
    tf_coil_cost = (
        tech.base_magnet_cost_busd * tf_material_multiplier * tf_coil_scale
    )

    # Reduced PF assembly-volume surrogate.  The six entries in
    # pf_coil_radii_m are the individual coils in the three symmetric pairs,
    # so summing circumference times cross-section counts the whole PF set.
    pf_coil_cross_section_m2 = jnp.maximum(x.pf_coil_thickness_m, 1.0e-6) ** 2
    pf_coil_volume_m3 = jnp.sum(
        2.0
        * jnp.pi
        * jnp.maximum(magnets.pf_coil_radii_m, 1.0e-6)
        * pf_coil_cross_section_m2
    )
    # The available cost coefficients distinguish REBCO from conventional
    # low-temperature superconductors; Nb3Sn is also used as the NbTi proxy.
    pf_material_multiplier = jnp.where(
        config.pf_superconductor_id == 0,
        tech.rebco_magnet_cost_multiplier,
        tech.nb3sn_magnet_cost_multiplier,
    )
    pf_field_utilization = magnets.pf_max_peak_field_t / jnp.maximum(
        tech.pf_peak_field_limit_t,
        1.0e-9,
    )
    pf_coil_scale = (
        (jnp.maximum(pf_coil_volume_m3, 1.0) / 250.0) ** 0.70
        * (
            1.0
            + 0.08 * pf_field_utilization
            + 0.04 * magnets.pf_current_density_utilization
        )
    )
    pf_coil_cost = (
        tech.base_pf_coil_cost_busd * pf_material_multiplier * pf_coil_scale
    )
    magnet_cost = tf_coil_cost + pf_coil_cost

    nuclear_volume = nuclear.blanket_volume_m3 + nuclear.shield_volume_m3
    blanket_concept_multiplier = jnp.where(
        config.blanket_concept_id == 1,
        tech.flibe_blanket_cost_multiplier,
        jnp.where(config.blanket_concept_id == 2, tech.hcpb_blanket_cost_multiplier, 1.0),
    )
    nuclear_cost = tech.base_nuclear_island_cost_busd * (
        jnp.maximum(nuclear_volume, 1.0) / 2500.0
    ) ** 0.65 * blanket_concept_multiplier

    bop_cost = tech.base_balance_plant_cost_busd * (
        jnp.maximum(power.gross_electric_power_mw, 1.0) / 1000.0
    ) ** 0.70 * (1.0 + 0.12 * power.recirculating_fraction)

    hcd_injected = x.auxiliary_heating_mw + x.current_drive_power_mw
    hcd_cost = tech.base_hcd_cost_busd * (jnp.maximum(hcd_injected, 1.0) / 150.0) ** 0.75

    direct_capital = magnet_cost + nuclear_cost + bop_cost + hcd_cost
    indirect_cost = tech.indirect_cost_fraction * direct_capital
    contingency_cost = tech.contingency_fraction * direct_capital
    total_capital = direct_capital + indirect_cost + contingency_cost
    (
        availability,
        divertor_availability_factor,
        neutron_availability_factor,
        magnet_availability_factor,
        pulsed_availability_factor,
        blanket_lifetime,
        divertor_lifetime,
        magnet_lifetime,
        scheduled_downtime,
    ) = evaluate_availability(magnets, nuclear, exhaust, tech, config)

    blanket_neutron_loading_ratio = nuclear.neutron_wall_loading_mw_m2 / jnp.maximum(
        tech.blanket_replacement_reference_neutron_wall_loading_mw_m2,
        1.0e-9,
    )
    blanket_replacement_cost = tech.blanket_replacement_cost_busd * jnp.maximum(
        blanket_neutron_loading_ratio,
        0.05,
    ) ** tech.blanket_replacement_neutron_wall_loading_exponent
    divertor_replacement_cost = tech.divertor_replacement_cost_busd * (
        exhaust.peak_divertor_heat_flux_mw_m2
        / jnp.maximum(tech.divertor_heat_flux_limit_mw_m2, 1.0e-9)
    ) ** 2
    replacement_cost_busd = blanket_replacement_cost + divertor_replacement_cost
    # COE is reported on the same flat-top net-power and base-availability
    # basis used by PROCESS-style reference comparisons. The reduced component
    # availability remains a diagnostic/output for optimization constraints.
    annual_energy = (
        jnp.maximum(power.flat_top_net_electric_power_mw, 1.0e-3)
        * HOURS_PER_YEAR
        * tech.availability
    )
    capacity_factor = availability * power.pulse_duty_factor
    annualized_capital_musd = tech.fixed_charge_rate * total_capital * 1.0e3
    annual_om_musd = tech.annual_om_fraction * direct_capital * 1.0e3
    annual_maintenance_musd = tech.maintenance_cost_fraction * replacement_cost_busd * 1.0e3
    capital_coe = annualized_capital_musd * 1.0e6 / annual_energy
    om_coe = annual_om_musd * 1.0e6 / annual_energy
    maintenance_coe = annual_maintenance_musd * 1.0e6 / annual_energy
    coe = capital_coe + om_coe + maintenance_coe

    return EconomicsState(
        tf_coil_cost_busd=tf_coil_cost,
        pf_coil_cost_busd=pf_coil_cost,
        magnet_cost_busd=magnet_cost,
        nuclear_island_cost_busd=nuclear_cost,
        balance_of_plant_cost_busd=bop_cost,
        hcd_cost_busd=hcd_cost,
        direct_capital_cost_busd=direct_capital,
        indirect_cost_busd=indirect_cost,
        contingency_cost_busd=contingency_cost,
        total_capital_cost_busd=total_capital,
        availability=availability,
        divertor_availability_factor=divertor_availability_factor,
        neutron_availability_factor=neutron_availability_factor,
        magnet_availability_factor=magnet_availability_factor,
        pulsed_availability_factor=pulsed_availability_factor,
        blanket_lifetime_fpy=blanket_lifetime,
        divertor_lifetime_fpy=divertor_lifetime,
        magnet_lifetime_fpy=magnet_lifetime,
        scheduled_downtime_days_per_year=scheduled_downtime,
        capacity_factor=capacity_factor,
        annualized_capital_cost_musd=annualized_capital_musd,
        annual_om_cost_musd=annual_om_musd,
        annual_maintenance_cost_musd=annual_maintenance_musd,
        annual_electricity_mwh=annual_energy,
        capital_coe_usd_mwh=capital_coe,
        om_coe_usd_mwh=om_coe,
        maintenance_coe_usd_mwh=maintenance_coe,
        coe_usd_mwh=coe,
    )
