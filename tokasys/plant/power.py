"""Plant thermal and electric power balance.

Refs:
    PROCESS systems-code model papers: Kovari et al., Fusion Eng. Des. 89,
    3054-3069 (2014), and Kovari et al., Fusion Eng. Des. 104, 9-20 (2016).

Notes:
    Auxiliary loads, coolant pumping, cryogenic loads, tritium plant loads and
    duty-factor corrections are reduced plant-balance surrogates.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import (
    DesignVariables,
    ExhaustState,
    GeometryState,
    MagnetState,
    NuclearState,
    PlasmaState,
    PowerPlantState,
    ReactorConfig,
    TechnologyParameters,
)


def evaluate_power_plant(
    x: DesignVariables,
    geom: GeometryState,
    plasma: PlasmaState,
    magnets: MagnetState,
    nuclear: NuclearState,
    exhaust: ExhaustState,
    tech: TechnologyParameters,
    config: ReactorConfig,
) -> PowerPlantState:
    """Convert recoverable heat to gross and net electric power.

    The balance includes pulsed duty factor, heating and current-drive wall-plug
    demand, coolant pumping, cryogenics, vacuum and tritium systems, power
    supplies, and fixed house loads.
    """
    pulse_cycle_s = (
        tech.cs_startup_duration_s
        + tech.cs_pulse_flat_top_duration_s
        + tech.pulse_ramp_down_duration_s
        + tech.pulse_dwell_time_s
    )
    pulsed_duty = tech.cs_pulse_flat_top_duration_s / jnp.maximum(pulse_cycle_s, 1.0e-9)
    duty_factor = jnp.where(config.steady_state, 1.0, pulsed_duty)

    flat_top_heating_wallplug = x.auxiliary_heating_mw / tech.heating_wallplug_efficiency
    flat_top_cd_wallplug = x.current_drive_power_mw / tech.current_drive_wallplug_efficiency
    flat_top_blanket_pumping = nuclear.blanket_pumping_power_mw
    flat_top_first_wall_pumping = (
        tech.first_wall_pumping_fraction * nuclear.first_wall_thermal_power_mw
    )
    flat_top_divertor_pumping = (
        tech.divertor_pumping_fraction * exhaust.divertor_target_power_mw
    )
    flat_top_coolant_pumping = (
        flat_top_blanket_pumping
        + flat_top_first_wall_pumping
        + flat_top_divertor_pumping
    )
    pump_electric_efficiency = jnp.maximum(
        tech.coolant_pump_electric_efficiency,
        1.0e-3,
    )
    flat_top_blanket_pumping_electric = (
        flat_top_blanket_pumping / pump_electric_efficiency
    )
    flat_top_first_wall_pumping_electric = (
        flat_top_first_wall_pumping / pump_electric_efficiency
    )
    flat_top_divertor_pumping_electric = (
        flat_top_divertor_pumping / pump_electric_efficiency
    )
    flat_top_coolant_pumping_electric = (
        flat_top_blanket_pumping_electric
        + flat_top_first_wall_pumping_electric
        + flat_top_divertor_pumping_electric
    )
    recoverable_thermal = (
        nuclear.blanket_thermal_power_mw
        + nuclear.first_wall_thermal_power_mw
        + nuclear.shield_thermal_power_mw
        + exhaust.divertor_heat_deposited_mw
        + flat_top_coolant_pumping
    )
    gross_electric = tech.thermal_efficiency * recoverable_thermal
    time_averaged_gross = duty_factor * gross_electric

    heating_wallplug = duty_factor * flat_top_heating_wallplug
    cd_wallplug = duty_factor * flat_top_cd_wallplug
    blanket_pumping = duty_factor * flat_top_blanket_pumping_electric
    first_wall_pumping = duty_factor * flat_top_first_wall_pumping_electric
    divertor_pumping = duty_factor * flat_top_divertor_pumping_electric
    pumping = blanket_pumping + first_wall_pumping + divertor_pumping
    mechanical_pumping = duty_factor * flat_top_coolant_pumping

    vacuum_pumping = (
        tech.vacuum_pumping_base_mw
        + tech.vacuum_pumping_mw_per_1000_m3 * geom.plasma_volume_m3 / 1000.0
    )
    tritium_throughput_kg_day = nuclear.tritium_breeding_rate_kg_per_year / 365.25
    tritium_plant = (
        tech.tritium_plant_base_mw
        + tech.tritium_plant_mw_per_kg_day * tritium_throughput_kg_day
    )
    cryogenic = magnets.cryogenic_electric_power_mw
    plant_base_auxiliary = tech.balance_of_plant_auxiliary_fraction * gross_electric
    tf_electric_supply = jnp.asarray(tech.tf_electric_supply_base_mw, dtype=float)
    pulse_cycle_mw = (
        jnp.abs(x.plasma_current_ma * 1.0e6 * magnets.cs_flux_required_wb)
        / jnp.maximum(pulse_cycle_s, 1.0e-9)
        / 1.0e6
    )
    pf_cs_electric_supply = jnp.where(
        config.steady_state,
        0.0,
        tech.pf_cs_electric_supply_base_mw
        + tech.pf_cs_power_supply_loss_multiplier * pulse_cycle_mw,
    )
    continuous_auxiliaries = (
        vacuum_pumping
        + tritium_plant
        + cryogenic
        + plant_base_auxiliary
        + tf_electric_supply
        + pf_cs_electric_supply
        + tech.fixed_house_load_mw
    )

    recirc = (
        heating_wallplug
        + cd_wallplug
        + pumping
        + continuous_auxiliaries
    )
    flat_top_recirc = (
        flat_top_heating_wallplug
        + flat_top_cd_wallplug
        + flat_top_coolant_pumping_electric
        + continuous_auxiliaries
    )
    flat_top_net = gross_electric - flat_top_recirc
    time_averaged_net = time_averaged_gross - recirc
    recirc_fraction = flat_top_recirc / jnp.maximum(gross_electric, 1.0e-6)

    return PowerPlantState(
        recoverable_thermal_power_mw=recoverable_thermal,
        gross_electric_power_mw=gross_electric,
        time_averaged_gross_electric_power_mw=time_averaged_gross,
        heating_wallplug_power_mw=heating_wallplug,
        current_drive_wallplug_power_mw=cd_wallplug,
        blanket_pumping_power_mw=blanket_pumping,
        first_wall_pumping_power_mw=first_wall_pumping,
        divertor_pumping_power_mw=divertor_pumping,
        coolant_pumping_power_mw=pumping,
        coolant_mechanical_pumping_power_mw=mechanical_pumping,
        vacuum_pumping_power_mw=vacuum_pumping,
        tritium_plant_power_mw=tritium_plant,
        cryogenic_static_power_mw=magnets.cryogenic_static_electric_power_mw,
        cryogenic_nuclear_power_mw=magnets.cryogenic_nuclear_electric_power_mw,
        cryogenic_current_lead_joint_power_mw=(
            magnets.cryogenic_current_lead_joint_electric_power_mw
        ),
        cryogenic_electric_power_mw=cryogenic,
        plant_base_auxiliary_power_mw=plant_base_auxiliary,
        tf_electric_supply_power_mw=tf_electric_supply,
        pf_cs_electric_supply_power_mw=pf_cs_electric_supply,
        fixed_house_load_mw=jnp.asarray(tech.fixed_house_load_mw, dtype=float),
        pulse_duty_factor=duty_factor,
        recirculating_power_mw=recirc,
        recirculating_fraction=recirc_fraction,
        flat_top_net_electric_power_mw=flat_top_net,
        time_averaged_net_electric_power_mw=time_averaged_net,
        net_electric_power_mw=time_averaged_net,
    )
