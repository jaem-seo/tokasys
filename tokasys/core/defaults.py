"""Default reactor problem used by examples and tests."""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import (
    ClosureVariables,
    DesignVariables,
    NumericalOptions,
    OptimizationConfig,
    ReactorConfig,
    ReactorProblem,
    ReactorVariableBounds,
    ReactorVariables,
    TechnologyParameters,
    VariableBounds,
)


def _dv(*values: float) -> DesignVariables:
    """Build JAX-backed design variables from scalar values in field order."""
    return DesignVariables(*[jnp.asarray(v, dtype=float) for v in values])


def _cv(*values: float) -> ClosureVariables:
    """Build JAX-backed closure variables from scalar values in field order."""
    return ClosureVariables(*[jnp.asarray(v, dtype=float) for v in values])


def default_config() -> ReactorConfig:
    """Return the baseline discrete reactor-model configuration."""
    return ReactorConfig(steady_state=True, objective_id=0, cost_model_id=0)


def default_numerics() -> NumericalOptions:
    """Return the baseline numerical discretization options."""
    return NumericalOptions(profile_num_points=96)


def default_optimization() -> OptimizationConfig:
    """Return the baseline objective and constraint-selection settings."""
    return OptimizationConfig()


def default_technology() -> TechnologyParameters:
    """Return the baseline physics, engineering, and economic assumptions."""
    return TechnologyParameters(
        effective_ion_mass_amu=2.5,
        ion_to_electron_temperature_ratio=1.0,
        zeff=1.8,
        fuel_ion_fraction=0.85,
        profile_fusion_enhancement=1.0,
        bootstrap_coefficient=0.55,
        current_drive_efficiency_20=0.45,
        heating_wallplug_efficiency=0.40,
        current_drive_wallplug_efficiency=0.42,
        beta_n_limit=3.0,
        q95_min=3.0,
        greenwald_fraction_max=0.95,
        fusion_gain_min=5.0,
        net_electric_target_mw=500.0,
        vacuum_vessel_thickness_m=0.18,
        thermal_gap_m=0.12,
        sol_width_m=0.08,
        tf_peak_field_limit_t=20.0,
        tf_allowable_stress_pa=700.0e6,
        tf_engineering_current_density_limit_a_m2=70.0e6,
        tf_structural_factor=0.60,
        tf_n_coils=18.0,
        tf_coil_toroidal_pack_fraction=0.65,
        tf_winding_pack_superconductor_fraction=0.25,
        tf_superconductor_operating_fraction=0.60,
        rebco_reference_current_density_a_m2=170.0e6,
        rebco_reference_temperature_k=20.0,
        rebco_operating_temperature_k=20.0,
        rebco_critical_temperature_k=92.0,
        rebco_upper_critical_field_t=40.0,
        rebco_reference_strain=0.0,
        rebco_operating_strain=0.0,
        rebco_strain_scale=0.006,
        nb3sn_reference_current_density_a_m2=95.0e6,
        nb3sn_reference_temperature_k=4.2,
        nb3sn_operating_temperature_k=4.5,
        nb3sn_critical_temperature_k=18.3,
        nb3sn_upper_critical_field_t=25.0,
        nb3sn_reference_strain=0.0,
        nb3sn_operating_strain=-0.0025,
        nb3sn_strain_scale=0.004,
        nbti_reference_current_density_a_m2=70.0e6,
        nbti_reference_temperature_k=4.2,
        nbti_operating_temperature_k=4.5,
        nbti_critical_temperature_k=9.2,
        nbti_upper_critical_field_t=14.5,
        nbti_reference_strain=0.0,
        nbti_operating_strain=-0.0010,
        nbti_strain_scale=0.006,
        cryogenic_cop=1.0 / 250.0,
        static_cryo_load_mw=0.020,
        nuclear_cryo_multiplier=1.0,
        current_lead_cryo_load_mw_per_ma_turn=1.0e-4,
        joint_cryo_load_mw_per_gj=1.0e-4,
        tf_nuclear_heat_limit_mw=0.050,
        plasma_internal_inductance=0.80,
        pf_vertical_field_coil_factor=1.30,
        pf_max_equilibrium_ampere_turns_a=80.0e6,
        pf_coil_radial_gap_m=0.35,
        pf_coil_pair_radial_step_m=0.15,
        pf_winding_pack_fraction=0.60,
        pf_structural_fraction=0.50,
        pf_peak_field_limit_t=12.0,
        pf_allowable_stress_pa=500.0e6,
        cs_allowable_stress_pa=700.0e6,
        cs_structural_fraction=0.60,
        cs_peak_field_limit_t=16.0,
        cs_flux_swing_coupling=0.90,
        cs_startup_duration_s=100.0,
        cs_pulse_flat_top_duration_s=7200.0,
        cs_flux_margin=1.15,
        blanket_energy_multiplication=1.35,
        blanket_coverage_fraction=0.85,
        blanket_attenuation_per_m=5.5,
        shield_attenuation_per_m=8.0,
        tbr_asymptote=1.55,
        tbr_scale_length_m=0.35,
        tbr_min=1.10,
        blanket_pumping_fraction=0.025,
        first_wall_neutron_heat_fraction=0.0,
        first_wall_radiation_heat_fraction=0.70,
        first_wall_alpha_loss_fraction=0.05,
        divertor_neutron_heat_fraction=0.0,
        divertor_radiation_heat_fraction=0.30,
        tf_neutron_geometric_view_factor=0.005,
        tf_fluence_lifetime_fpy=5.0,
        tf_fluence_limit_n_m2=1.0e22,
        tf_dpa_per_1e25_neutrons_m2=1.0,
        tf_dpa_limit=1.0e-3,
        first_wall_surface_heat_load_limit_mw_m2=1.0,
        tritium_processing_residence_time_days=2.0,
        tritium_startup_inventory_kg=2.0,
        tritium_doubling_time_max_years=5.0,
        divertor_radiated_fraction=0.55,
        divertor_lambda_integral_m=0.055,
        divertor_lambda_q_coefficient_m=0.00063,
        divertor_lambda_q_bpol_exponent=-1.19,
        divertor_spreading_factor_m=0.050,
        divertor_peak_to_integral_factor=1.20,
        divertor_flux_expansion=8.0,
        divertor_n_targets=2.0,
        divertor_heat_flux_limit_mw_m2=15.0,
        divertor_coolant_temperature_c=150.0,
        divertor_thermal_resistance_c_per_mw_m2=35.0,
        divertor_temperature_heat_flux_exponent=1.15,
        divertor_target_temperature_limit_c=1200.0,
        divertor_lifetime_min_fpy=1.0,
        thermal_efficiency=0.42,
        coolant_pumping_fraction=0.025,
        first_wall_pumping_fraction=0.015,
        divertor_pumping_fraction=0.040,
        coolant_pump_electric_efficiency=0.45,
        balance_of_plant_auxiliary_fraction=0.050,
        tf_electric_supply_base_mw=0.0,
        pf_cs_electric_supply_base_mw=12.0,
        pf_cs_power_supply_loss_multiplier=4.0,
        vacuum_pumping_base_mw=8.0,
        vacuum_pumping_mw_per_1000_m3=2.0,
        tritium_plant_base_mw=5.0,
        tritium_plant_mw_per_kg_day=2.5,
        pulse_ramp_down_duration_s=120.0,
        pulse_dwell_time_s=600.0,
        fixed_house_load_mw=45.0,
        recirculating_fraction_max=0.45,
        availability=0.75,
        availability_min=0.35,
        divertor_availability_penalty=0.035,
        neutron_availability_penalty=0.025,
        magnet_availability_penalty=0.020,
        pulsed_availability_penalty=0.050,
        blanket_reference_lifetime_fpy=3.0,
        blanket_replacement_duration_days=60.0,
        blanket_lifetime_neutron_exponent=1.0,
        divertor_reference_lifetime_fpy=2.0,
        divertor_replacement_duration_days=30.0,
        divertor_lifetime_heat_flux_exponent=1.5,
        magnet_reference_lifetime_fpy=20.0,
        magnet_repair_duration_days=20.0,
        magnet_lifetime_utilization_exponent=2.0,
        fixed_charge_rate=0.08,
        annual_om_fraction=0.025,
        indirect_cost_fraction=0.15,
        contingency_fraction=0.10,
        maintenance_cost_fraction=0.25,
        blanket_replacement_cost_busd=0.80,
        blanket_replacement_reference_neutron_wall_loading_mw_m2=2.0,
        blanket_replacement_neutron_wall_loading_exponent=0.60,
        divertor_replacement_cost_busd=0.25,
        rebco_magnet_cost_multiplier=1.18,
        nb3sn_magnet_cost_multiplier=1.00,
        flibe_blanket_cost_multiplier=1.12,
        hcpb_blanket_cost_multiplier=1.05,
        base_magnet_cost_busd=2.1,
        # Calibrated near the 0.52--0.64 BUSD PF-assembly costs in the
        # bundled PROCESS reference runs, before material/utilization factors.
        base_pf_coil_cost_busd=0.55,
        base_nuclear_island_cost_busd=3.0,
        base_balance_plant_cost_busd=2.7,
        base_hcd_cost_busd=0.9,
    )


def default_problem() -> ReactorProblem:
    """Assemble the baseline design point, closure state, bounds, and settings."""
    initial = _dv(
        6.2,   # R [m]
        3.1,   # aspect ratio
        1.8,   # elongation
        0.35,  # triangularity
        5.3,   # B0 [T]
        15.0,  # Ip [MA]
        0.82,  # f_GW
        8.00,  # average T [keV]
        1.00,  # H98
        0.00,  # auxiliary heating [MW]
        50.0,  # current-drive power [MW]
        0.05,  # inboard FW [m]
        0.05,  # outboard FW [m]
        0.85,  # inboard blanket [m]
        0.85,  # outboard blanket [m]
        0.75,  # inboard shield [m]
        0.75,  # outboard shield [m]
        0.85,  # TF thickness [m]
        0.85,  # PF radial/vertical thickness [m]
        0.80,  # CS thickness [m]
    )

    lower = _dv(
        4.0, 2.0, 1.4, 0.05, 3.0, 7.0, 0.40, 6.0, 0.70,
        0.0, 0.0, 0.02, 0.02, 0.40, 0.40, 0.30, 0.30, 0.30, 0.30, 0.30,
    )
    upper = _dv(
        12.0, 4.5, 2.2, 0.65, 10.0, 30.0, 1.05, 30.0, 1.80,
        250.0, 350.0, 0.20, 0.20, 1.50, 1.50, 1.50, 1.50, 2.00, 2.00, 2.00,
    )
    closure = _cv(
        650.0,  # stored energy [MJ]
        280.0,  # transport loss [MW]
        2.1,    # bootstrap current [MA]
        6.2,    # current-drive current [MA]
        540.0,  # net electric power [MW]
        17.4,   # peak TF field [T]
        1.20,   # TBR
        4.25,   # divertor heat flux [MW/m2]
    )
    closure_lower = _cv(
        10.0, 1.0, 0.0, 0.0, -500.0, 1.0, 0.1, 0.01,
    )
    closure_upper = _cv(
        3000.0, 3000.0, 30.0, 30.0, 3000.0, 35.0, 2.5, 50.0,
    )

    technology = default_technology()

    return ReactorProblem(
        initial_design=initial,
        initial_variables=ReactorVariables(design=initial, closure=closure),
        bounds=VariableBounds(lower=lower, upper=upper),
        variable_bounds=ReactorVariableBounds(
            lower=ReactorVariables(design=lower, closure=closure_lower),
            upper=ReactorVariables(design=upper, closure=closure_upper),
        ),
        technology=technology,
        config=default_config(),
        optimization=default_optimization(),
        numerics=default_numerics(),
    )
