"""Selectable plant-cost, availability and cost-of-electricity models.

Refs:
    PROCESS system-code economics/engineering practice: Kovari et al., Fusion
    Eng. Des. 104, 9-20 (2016). The availability model follows reduced
    component-lifetime/RAMI logic for conceptual studies.
    Jo et al., "Cost Assessment of a Tokamak Fusion Reactor with an Inventive
    Method for Optimum Build Determination", Energies 14, 6817 (2021),
    https://doi.org/10.3390/en14206817, Section 2.3 and Tables 2-3.

Notes:
    ``cost_model_id=0`` preserves the calibrated legacy surrogate.
    ``cost_model_id=1`` applies the standard C/C_ref=(P/P_ref)^0.6 capacity law
    to the existing reference component costs.
    ``cost_model_id=2`` implements Jo et al. Equations (8)-(14). Quantities
    that Jo et al. obtained from their coupled systems analysis are reconstructed
    from the corresponding TokaSys geometry and magnet states below.
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

# Jo et al. (2021), Table 2, all in constant 2010 USD.
_JO_FIRST_WALL_USD_M2 = 6.0e4
_JO_DIVERTOR_USD_M2 = 6.0e5
_JO_PBLI_USD_KG = 20.0
_JO_FMS_USD_KG = 90.0
_JO_SIC_USD_KG = 100.0
_JO_SHIELD_USD_KG = 40.0
_JO_TF_COIL_USD_M = 2.0e3
_JO_PF_COIL_USD_A_M_T = 0.02
_JO_CS_USD_M = 1.0e3
_JO_COIL_CASE_USD_KG = 50.0
_JO_NBI_USD_W = 5.3

# Jo et al. (2021), Table 3 and Equations (8)-(14).
_JO_PROCESS_CONTINGENCY = 0.15
_JO_INDIRECT_CHARGE = 0.375
_JO_CAPITALIZATION_FACTOR = 1.075
_JO_FIXED_CHARGE_RATE = 0.10
_JO_REACTOR_LIFETIME_YEARS = 40.0
_JO_BLANKET_FLUENCE_LIMIT_MW_YEAR_M2 = 20.0
_JO_DIVERTOR_FLUENCE_LIMIT_MW_YEAR_M2 = 15.0
_JO_DECOMMISSIONING_COE_USD_MWH = 1.0
_JO_WASTE_COE_USD_MWH = 0.5

# TokaSys adapters for component masses/lengths that the paper obtained from a
# separate coupled systems analysis. They are deliberately visible here rather
# than hidden in fitted cost multipliers.
_JO_COIL_OPERATING_CURRENT_A = 5.0e4
_JO_PBLI_DENSITY_KG_M3 = 9.3e3
_JO_FMS_DENSITY_KG_M3 = 7.8e3
_JO_SIC_DENSITY_KG_M3 = 3.2e3
_JO_SHIELD_EFFECTIVE_DENSITY_KG_M3 = 1.0e4
_JO_COIL_CASE_DENSITY_KG_M3 = 8.0e3
_JO_BLANKET_PBLI_VOLUME_FRACTION = 0.90
_JO_BLANKET_FMS_VOLUME_FRACTION = 0.09
_JO_BLANKET_SIC_VOLUME_FRACTION = 0.01
_CAPACITY_REFERENCE_POWER_MW = 1000.0
_CAPACITY_SCALING_EXPONENT = 0.60


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


def _evaluate_economics_legacy(
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
    """Evaluate the original calibrated surrogate (``cost_model_id=0``)."""
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


def _select_cost_model(
    model_id: int,
    legacy_value: jnp.ndarray,
    capacity_value: jnp.ndarray,
    jo_value: jnp.ndarray,
) -> jnp.ndarray:
    """Select a cost-model result without introducing a JAX control-flow break."""
    return jnp.where(
        model_id == 0,
        legacy_value,
        jnp.where(model_id == 1, capacity_value, jo_value),
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
    """Evaluate the selected cost model.

    Model 0 is the original TokaSys cost surrogate. Model 1 is a 0.6-power
    capacity scaling anchored to the existing component base costs at
    1000 MW gross electric capacity. Model 2 follows Jo et al. (2021),
    Section 2.3, with system-analysis inputs reconstructed from TokaSys states.
    """
    legacy = _evaluate_economics_legacy(
        x,
        geom,
        plasma,
        magnets,
        nuclear,
        exhaust,
        power,
        tech,
        config,
    )

    tf_material_multiplier = jnp.where(
        config.tf_superconductor_id == 1,
        tech.nb3sn_magnet_cost_multiplier,
        tech.rebco_magnet_cost_multiplier,
    )
    pf_material_multiplier = jnp.where(
        config.pf_superconductor_id == 0,
        tech.rebco_magnet_cost_multiplier,
        tech.nb3sn_magnet_cost_multiplier,
    )
    blanket_concept_multiplier = jnp.where(
        config.blanket_concept_id == 1,
        tech.flibe_blanket_cost_multiplier,
        jnp.where(
            config.blanket_concept_id == 2,
            tech.hcpb_blanket_cost_multiplier,
            1.0,
        ),
    )

    # Model 1: the standard capacity-scaling law C/C_ref=(P/P_ref)^n with
    # n=0.6. Gross electric power is used as plant nameplate capacity so that
    # a larger recirculating load does not spuriously make the plant cheaper.
    capacity_scale = (
        jnp.maximum(power.gross_electric_power_mw, 1.0)
        / _CAPACITY_REFERENCE_POWER_MW
    ) ** _CAPACITY_SCALING_EXPONENT
    tf_coil_cost_1 = (
        tech.base_magnet_cost_busd * tf_material_multiplier * capacity_scale
    )
    pf_coil_cost_1 = (
        tech.base_pf_coil_cost_busd * pf_material_multiplier * capacity_scale
    )
    magnet_cost_1 = tf_coil_cost_1 + pf_coil_cost_1
    nuclear_cost_1 = (
        tech.base_nuclear_island_cost_busd
        * blanket_concept_multiplier
        * capacity_scale
    )
    bop_cost_1 = tech.base_balance_plant_cost_busd * capacity_scale
    hcd_cost_1 = tech.base_hcd_cost_busd * capacity_scale
    direct_capital_1 = (
        magnet_cost_1 + nuclear_cost_1 + bop_cost_1 + hcd_cost_1
    )
    indirect_cost_1 = tech.indirect_cost_fraction * direct_capital_1
    contingency_cost_1 = tech.contingency_fraction * direct_capital_1
    total_capital_1 = (
        direct_capital_1 + indirect_cost_1 + contingency_cost_1
    )
    annualized_capital_1 = tech.fixed_charge_rate * total_capital_1 * 1.0e3
    annual_om_1 = tech.annual_om_fraction * direct_capital_1 * 1.0e3
    annual_maintenance_1 = legacy.annual_maintenance_cost_musd
    capital_coe_1 = (
        annualized_capital_1 * 1.0e6 / legacy.annual_electricity_mwh
    )
    om_coe_1 = annual_om_1 * 1.0e6 / legacy.annual_electricity_mwh
    maintenance_coe_1 = (
        annual_maintenance_1 * 1.0e6 / legacy.annual_electricity_mwh
    )
    coe_1 = capital_coe_1 + om_coe_1 + maintenance_coe_1

    # Model 2 component costs. Jo et al. obtained component length, area,
    # volume and mass from a separate coupled systems analysis. The following
    # conversions provide those quantities from the TokaSys state:
    #   - a D-coil perimeter and 50 kA conductor current for TF/CS length;
    #   - circumference * ampere-turns * peak field for PF cost;
    #   - structural volumes times effective densities for coil cases;
    #   - the TokaSys first-wall/divertor areas and blanket/shield volumes.
    tf_radial_span_m = jnp.maximum(
        geom.outboard_tf_centroid_radius_m
        - geom.inboard_tf_centroid_radius_m,
        1.0e-6,
    )
    tf_turn_length_m = (
        2.0 * geom.tf_coil_height_m + jnp.pi * tf_radial_span_m
    )
    tf_conductor_length_m = (
        jnp.abs(magnets.tf_required_ampere_turns_a)
        / _JO_COIL_OPERATING_CURRENT_A
        * tf_turn_length_m
    )
    cs_outer_radius_m = jnp.maximum(x.cs_thickness_m, 0.25)
    cs_mean_radius_m = 0.5 * cs_outer_radius_m
    cs_turn_length_m = 2.0 * jnp.pi * cs_mean_radius_m
    cs_conductor_length_m = (
        jnp.abs(magnets.cs_required_ampere_turns_a)
        / _JO_COIL_OPERATING_CURRENT_A
        * cs_turn_length_m
    )
    pf_ampere_metre_tesla = jnp.sum(
        jnp.abs(magnets.pf_coil_ampere_turns_a)
        * 2.0
        * jnp.pi
        * jnp.maximum(magnets.pf_coil_radii_m, 1.0e-6)
        * magnets.pf_coil_peak_field_t
    )

    tf_case_volume_m3 = (
        magnets.tf_structural_area_m2
        * tf_turn_length_m
        * tech.tf_n_coils
    )
    pf_gross_volume_m3 = jnp.sum(
        2.0
        * jnp.pi
        * jnp.maximum(magnets.pf_coil_radii_m, 1.0e-6)
        * jnp.maximum(x.pf_coil_thickness_m, 1.0e-6) ** 2
    )
    pf_case_volume_m3 = tech.pf_structural_fraction * pf_gross_volume_m3
    cs_case_volume_m3 = (
        2.0
        * jnp.pi
        * cs_mean_radius_m
        * x.cs_thickness_m
        * geom.tf_coil_height_m
        * tech.cs_structural_fraction
    )
    tf_cs_case_cost_musd = (
        _JO_COIL_CASE_USD_KG
        * _JO_COIL_CASE_DENSITY_KG_M3
        * (tf_case_volume_m3 + cs_case_volume_m3)
        / 1.0e6
    )
    pf_case_cost_musd = (
        _JO_COIL_CASE_USD_KG
        * _JO_COIL_CASE_DENSITY_KG_M3
        * pf_case_volume_m3
        / 1.0e6
    )
    tf_cs_cost_musd = (
        _JO_TF_COIL_USD_M * tf_conductor_length_m
        + _JO_CS_USD_M * cs_conductor_length_m
    ) / 1.0e6 + tf_cs_case_cost_musd
    pf_cost_musd = (
        _JO_PF_COIL_USD_A_M_T * pf_ampere_metre_tesla / 1.0e6
        + pf_case_cost_musd
    )

    first_wall_cost_musd = (
        _JO_FIRST_WALL_USD_M2 * geom.first_wall_area_m2 / 1.0e6
    )
    shield_cost_musd = (
        _JO_SHIELD_USD_KG
        * _JO_SHIELD_EFFECTIVE_DENSITY_KG_M3
        * nuclear.shield_volume_m3
        / 1.0e6
    )
    reactor_cost_musd = first_wall_cost_musd + shield_cost_musd
    blanket_cost_musd = nuclear.blanket_volume_m3 * (
        _JO_BLANKET_PBLI_VOLUME_FRACTION
        * _JO_PBLI_DENSITY_KG_M3
        * _JO_PBLI_USD_KG
        + _JO_BLANKET_FMS_VOLUME_FRACTION
        * _JO_FMS_DENSITY_KG_M3
        * _JO_FMS_USD_KG
        + _JO_BLANKET_SIC_VOLUME_FRACTION
        * _JO_SIC_DENSITY_KG_M3
        * _JO_SIC_USD_KG
    ) / 1.0e6
    divertor_cost_musd = (
        _JO_DIVERTOR_USD_M2 * exhaust.divertor_effective_area_m2 / 1.0e6
    )
    hcd_injected_mw = x.auxiliary_heating_mw + x.current_drive_power_mw
    auxiliary_cost_musd = _JO_NBI_USD_W * hcd_injected_mw

    pe_mw = jnp.maximum(power.net_electric_power_mw, 1.0e-3)
    pth_mw = jnp.maximum(power.recoverable_thermal_power_mw, 1.0e-3)
    fusion_island_volume_m3 = (
        jnp.pi * geom.machine_outer_radius_m**2 * geom.tf_coil_height_m
    )
    bop_equipment_cost_musd = (
        900.0 + 900.0 * pe_mw / 1200.0
    ) * (pth_mw / 4150.0) ** 0.60
    building_cost_musd = 839.0 * (
        fusion_island_volume_m3 / 5100.0
    ) ** 0.67
    steam_system_cost_musd = 221.0 * (pth_mw / 4150.0) ** 0.60

    # Jo et al. Equation (11): blanket and divertor costs are excluded from
    # initial capital and enter the annual fuel-cycle cost below.
    tf_coil_cost_2 = 1.5 * tf_cs_cost_musd / 1.0e3
    pf_coil_cost_2 = 1.5 * pf_cost_musd / 1.0e3
    magnet_cost_2 = tf_coil_cost_2 + pf_coil_cost_2
    nuclear_cost_2 = 1.1 * reactor_cost_musd / 1.0e3
    bop_cost_2 = (
        bop_equipment_cost_musd
        + building_cost_musd
        + steam_system_cost_musd
    ) / 1.0e3
    hcd_cost_2 = 1.1 * auxiliary_cost_musd / 1.0e3
    pre_contingency_capital_2 = (
        magnet_cost_2 + nuclear_cost_2 + bop_cost_2 + hcd_cost_2
    )
    contingency_cost_2 = (
        _JO_PROCESS_CONTINGENCY * pre_contingency_capital_2
    )
    jo_direct_capital_2 = (
        pre_contingency_capital_2 + contingency_cost_2
    )
    # Equation (9) writes f_IND as a multiplier, while Table 3 reports an
    # "indirect charge" of 0.375. Treating that charge as 1 + 0.375 avoids the
    # unphysical result that adding indirect costs reduces total capital and
    # is consistent with the paper's reported 4-6 BUSD direct-cost cases.
    total_capital_2 = (
        jo_direct_capital_2
        * _JO_CAPITALIZATION_FACTOR
        * (1.0 + _JO_INDIRECT_CHARGE)
    )
    indirect_cost_2 = total_capital_2 - jo_direct_capital_2
    direct_capital_2 = pre_contingency_capital_2

    availability_2 = tech.availability
    annual_electricity_2 = (
        pe_mw * HOURS_PER_YEAR * availability_2
    )
    annualized_capital_2 = (
        _JO_FIXED_CHARGE_RATE * total_capital_2 * 1.0e3
    )
    annual_blanket_cost_musd = 1.1 * (
        1.1 * blanket_cost_musd * _JO_FIXED_CHARGE_RATE
        + (
            availability_2
            * _JO_REACTOR_LIFETIME_YEARS
            * nuclear.neutron_wall_loading_mw_m2
            / _JO_BLANKET_FLUENCE_LIMIT_MW_YEAR_M2
            - 1.0
        )
        * blanket_cost_musd
        / _JO_REACTOR_LIFETIME_YEARS
    )
    annual_divertor_cost_musd = 1.2 * (
        1.1 * divertor_cost_musd * _JO_FIXED_CHARGE_RATE
        + (
            availability_2
            * _JO_REACTOR_LIFETIME_YEARS
            * exhaust.peak_divertor_heat_flux_mw_m2
            / _JO_DIVERTOR_FLUENCE_LIMIT_MW_YEAR_M2
            - 1.0
        )
        * divertor_cost_musd
        / _JO_REACTOR_LIFETIME_YEARS
    )
    annual_fuel_cycle_cost_musd = (
        annual_blanket_cost_musd
        + annual_divertor_cost_musd
        + 0.1 * auxiliary_cost_musd
        + 7.5
    )
    annual_om_2 = 108.0 * (pe_mw / 1200.0) ** 0.5
    fixed_backend_coe_2 = (
        _JO_DECOMMISSIONING_COE_USD_MWH
        + _JO_WASTE_COE_USD_MWH
    )
    annual_maintenance_2 = (
        annual_fuel_cycle_cost_musd
        + fixed_backend_coe_2 * annual_electricity_2 / 1.0e6
    )
    capital_coe_2 = (
        annualized_capital_2 * 1.0e6 / annual_electricity_2
    )
    om_coe_2 = annual_om_2 * 1.0e6 / annual_electricity_2
    maintenance_coe_2 = (
        annual_maintenance_2 * 1.0e6 / annual_electricity_2
    )
    coe_2 = capital_coe_2 + om_coe_2 + maintenance_coe_2

    model_id = config.cost_model_id

    def select(
        legacy_value: jnp.ndarray,
        capacity_value: jnp.ndarray,
        jo_value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Select one component result from the three evaluated cost models."""
        return _select_cost_model(
            model_id,
            legacy_value,
            capacity_value,
            jo_value,
        )

    return EconomicsState(
        tf_coil_cost_busd=select(
            legacy.tf_coil_cost_busd,
            tf_coil_cost_1,
            tf_coil_cost_2,
        ),
        pf_coil_cost_busd=select(
            legacy.pf_coil_cost_busd,
            pf_coil_cost_1,
            pf_coil_cost_2,
        ),
        magnet_cost_busd=select(
            legacy.magnet_cost_busd,
            magnet_cost_1,
            magnet_cost_2,
        ),
        nuclear_island_cost_busd=select(
            legacy.nuclear_island_cost_busd,
            nuclear_cost_1,
            nuclear_cost_2,
        ),
        balance_of_plant_cost_busd=select(
            legacy.balance_of_plant_cost_busd,
            bop_cost_1,
            bop_cost_2,
        ),
        hcd_cost_busd=select(
            legacy.hcd_cost_busd,
            hcd_cost_1,
            hcd_cost_2,
        ),
        direct_capital_cost_busd=select(
            legacy.direct_capital_cost_busd,
            direct_capital_1,
            direct_capital_2,
        ),
        indirect_cost_busd=select(
            legacy.indirect_cost_busd,
            indirect_cost_1,
            indirect_cost_2,
        ),
        contingency_cost_busd=select(
            legacy.contingency_cost_busd,
            contingency_cost_1,
            contingency_cost_2,
        ),
        total_capital_cost_busd=select(
            legacy.total_capital_cost_busd,
            total_capital_1,
            total_capital_2,
        ),
        availability=legacy.availability,
        divertor_availability_factor=legacy.divertor_availability_factor,
        neutron_availability_factor=legacy.neutron_availability_factor,
        magnet_availability_factor=legacy.magnet_availability_factor,
        pulsed_availability_factor=legacy.pulsed_availability_factor,
        blanket_lifetime_fpy=legacy.blanket_lifetime_fpy,
        divertor_lifetime_fpy=legacy.divertor_lifetime_fpy,
        magnet_lifetime_fpy=legacy.magnet_lifetime_fpy,
        scheduled_downtime_days_per_year=(
            legacy.scheduled_downtime_days_per_year
        ),
        capacity_factor=legacy.capacity_factor,
        annualized_capital_cost_musd=select(
            legacy.annualized_capital_cost_musd,
            annualized_capital_1,
            annualized_capital_2,
        ),
        annual_om_cost_musd=select(
            legacy.annual_om_cost_musd,
            annual_om_1,
            annual_om_2,
        ),
        annual_maintenance_cost_musd=select(
            legacy.annual_maintenance_cost_musd,
            annual_maintenance_1,
            annual_maintenance_2,
        ),
        annual_electricity_mwh=select(
            legacy.annual_electricity_mwh,
            legacy.annual_electricity_mwh,
            annual_electricity_2,
        ),
        capital_coe_usd_mwh=select(
            legacy.capital_coe_usd_mwh,
            capital_coe_1,
            capital_coe_2,
        ),
        om_coe_usd_mwh=select(
            legacy.om_coe_usd_mwh,
            om_coe_1,
            om_coe_2,
        ),
        maintenance_coe_usd_mwh=select(
            legacy.maintenance_coe_usd_mwh,
            maintenance_coe_1,
            maintenance_coe_2,
        ),
        coe_usd_mwh=select(legacy.coe_usd_mwh, coe_1, coe_2),
    )
