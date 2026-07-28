"""JAX-compatible immutable data containers.

NamedTuple is used because JAX treats it as a pytree without extra registration.
All numeric fields are scalar JAX arrays or Python floats that become arrays
inside traced functions.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class DesignVariables(NamedTuple):
    major_radius_m: Array
    aspect_ratio: Array
    elongation: Array
    triangularity: Array
    toroidal_field_t: Array
    plasma_current_ma: Array
    greenwald_fraction: Array
    target_temperature_keV: Array
    confinement_h98: Array
    auxiliary_heating_mw: Array
    current_drive_power_mw: Array
    inboard_first_wall_thickness_m: Array
    outboard_first_wall_thickness_m: Array
    inboard_blanket_thickness_m: Array
    outboard_blanket_thickness_m: Array
    inboard_shield_thickness_m: Array
    outboard_shield_thickness_m: Array
    tf_coil_thickness_m: Array
    pf_coil_thickness_m: Array
    cs_thickness_m: Array


class ClosureVariables(NamedTuple):
    stored_energy_mj: Array
    transport_loss_mw: Array
    bootstrap_current_ma: Array
    current_drive_current_ma: Array
    net_electric_power_mw: Array
    tf_peak_field_t: Array
    tbr: Array
    divertor_heat_flux_mw_m2: Array


class ReactorVariables(NamedTuple):
    design: DesignVariables
    closure: ClosureVariables


class TechnologyParameters(NamedTuple):
    # Plasma and operation
    effective_ion_mass_amu: float
    ion_to_electron_temperature_ratio: float
    zeff: float
    fuel_ion_fraction: float
    profile_fusion_enhancement: float
    bootstrap_coefficient: float
    current_drive_efficiency_20: float
    heating_wallplug_efficiency: float
    current_drive_wallplug_efficiency: float
    beta_n_limit: float
    q95_min: float
    greenwald_fraction_max: float
    fusion_gain_min: float
    net_electric_target_mw: float
    # Build and magnet
    vacuum_vessel_thickness_m: float
    thermal_gap_m: float
    sol_width_m: float
    tf_peak_field_limit_t: float
    tf_allowable_stress_pa: float
    tf_engineering_current_density_limit_a_m2: float
    tf_structural_factor: float
    tf_n_coils: float
    tf_coil_toroidal_pack_fraction: float
    tf_winding_pack_superconductor_fraction: float
    tf_superconductor_operating_fraction: float
    rebco_reference_current_density_a_m2: float
    rebco_reference_temperature_k: float
    rebco_operating_temperature_k: float
    rebco_critical_temperature_k: float
    rebco_upper_critical_field_t: float
    rebco_reference_strain: float
    rebco_operating_strain: float
    rebco_strain_scale: float
    nb3sn_reference_current_density_a_m2: float
    nb3sn_reference_temperature_k: float
    nb3sn_operating_temperature_k: float
    nb3sn_critical_temperature_k: float
    nb3sn_upper_critical_field_t: float
    nb3sn_reference_strain: float
    nb3sn_operating_strain: float
    nb3sn_strain_scale: float
    nbti_reference_current_density_a_m2: float
    nbti_reference_temperature_k: float
    nbti_operating_temperature_k: float
    nbti_critical_temperature_k: float
    nbti_upper_critical_field_t: float
    nbti_reference_strain: float
    nbti_operating_strain: float
    nbti_strain_scale: float
    cryogenic_cop: float
    static_cryo_load_mw: float
    nuclear_cryo_multiplier: float
    current_lead_cryo_load_mw_per_ma_turn: float
    joint_cryo_load_mw_per_gj: float
    tf_nuclear_heat_limit_mw: float
    plasma_internal_inductance: float
    pf_vertical_field_coil_factor: float
    pf_max_equilibrium_ampere_turns_a: float
    pf_coil_radial_gap_m: float
    pf_coil_pair_radial_step_m: float
    pf_winding_pack_fraction: float
    pf_structural_fraction: float
    pf_peak_field_limit_t: float
    pf_allowable_stress_pa: float
    cs_allowable_stress_pa: float
    cs_structural_fraction: float
    cs_peak_field_limit_t: float
    cs_flux_swing_coupling: float
    cs_startup_duration_s: float
    cs_pulse_flat_top_duration_s: float
    cs_flux_margin: float
    # Nuclear island
    blanket_energy_multiplication: float
    blanket_coverage_fraction: float
    blanket_attenuation_per_m: float
    shield_attenuation_per_m: float
    tbr_asymptote: float
    tbr_scale_length_m: float
    tbr_min: float
    blanket_pumping_fraction: float
    first_wall_neutron_heat_fraction: float
    first_wall_radiation_heat_fraction: float
    first_wall_alpha_loss_fraction: float
    divertor_neutron_heat_fraction: float
    divertor_radiation_heat_fraction: float
    tf_neutron_geometric_view_factor: float
    tf_fluence_lifetime_fpy: float
    tf_fluence_limit_n_m2: float
    tf_dpa_per_1e25_neutrons_m2: float
    tf_dpa_limit: float
    first_wall_surface_heat_load_limit_mw_m2: float
    tritium_processing_residence_time_days: float
    tritium_startup_inventory_kg: float
    tritium_doubling_time_max_years: float
    # Exhaust
    divertor_radiated_fraction: float
    divertor_lambda_integral_m: float
    divertor_lambda_q_coefficient_m: float
    divertor_lambda_q_bpol_exponent: float
    divertor_spreading_factor_m: float
    divertor_peak_to_integral_factor: float
    divertor_flux_expansion: float
    divertor_n_targets: float
    divertor_heat_flux_limit_mw_m2: float
    divertor_coolant_temperature_c: float
    divertor_thermal_resistance_c_per_mw_m2: float
    divertor_temperature_heat_flux_exponent: float
    divertor_target_temperature_limit_c: float
    divertor_lifetime_min_fpy: float
    # Plant
    thermal_efficiency: float
    coolant_pumping_fraction: float
    first_wall_pumping_fraction: float
    divertor_pumping_fraction: float
    coolant_pump_electric_efficiency: float
    balance_of_plant_auxiliary_fraction: float
    tf_electric_supply_base_mw: float
    pf_cs_electric_supply_base_mw: float
    pf_cs_power_supply_loss_multiplier: float
    vacuum_pumping_base_mw: float
    vacuum_pumping_mw_per_1000_m3: float
    tritium_plant_base_mw: float
    tritium_plant_mw_per_kg_day: float
    pulse_ramp_down_duration_s: float
    pulse_dwell_time_s: float
    fixed_house_load_mw: float
    recirculating_fraction_max: float
    availability: float
    availability_min: float
    divertor_availability_penalty: float
    neutron_availability_penalty: float
    magnet_availability_penalty: float
    pulsed_availability_penalty: float
    blanket_reference_lifetime_fpy: float
    blanket_replacement_duration_days: float
    blanket_lifetime_neutron_exponent: float
    divertor_reference_lifetime_fpy: float
    divertor_replacement_duration_days: float
    divertor_lifetime_heat_flux_exponent: float
    magnet_reference_lifetime_fpy: float
    magnet_repair_duration_days: float
    magnet_lifetime_utilization_exponent: float
    fixed_charge_rate: float
    annual_om_fraction: float
    indirect_cost_fraction: float
    contingency_fraction: float
    maintenance_cost_fraction: float
    blanket_replacement_cost_busd: float
    blanket_replacement_reference_neutron_wall_loading_mw_m2: float
    blanket_replacement_neutron_wall_loading_exponent: float
    divertor_replacement_cost_busd: float
    rebco_magnet_cost_multiplier: float
    nb3sn_magnet_cost_multiplier: float
    flibe_blanket_cost_multiplier: float
    hcpb_blanket_cost_multiplier: float
    # Cost coefficients in billion USD
    base_magnet_cost_busd: float
    base_pf_coil_cost_busd: float
    base_nuclear_island_cost_busd: float
    base_balance_plant_cost_busd: float
    base_hcd_cost_busd: float


class ReactorConfig(NamedTuple):
    """Static model/scenario choices encoded for JIT friendliness."""

    steady_state: bool = True
    objective_id: int = 0  # 0 radius, 1 COE, 2 negative net power
    cost_model_id: int = 0  # 0 legacy proxy, 1 capacity scaling, 2 Jo et al. (2021)
    energy_closure_mode: str = "self_consistent"  # full_space, self_consistent, fixed_point
    energy_closure_iterations: int = 5
    plasma_profile_model: str = "tokasys"  # tokasys, process_parabolic, process_pedestal
    temperature_profile_exponent: float = 1.5
    pedestal_fixed_width: float = 0.04
    pedestal_width_coefficient: float = 0.076
    alpha_critical_shear: float = 1.0  # multiplier on q-profile diagnostic shear
    separatrix_temperature_keV: float = 0.10
    density_edge_fraction: float = 0.35
    density_tanh_width: float = 0.5  # tanh scale as a fraction of pedestal width
    density_core_peaking_fraction: float = 0.5
    density_profile_exponent: float = 1.5
    process_profile_alphan: float = 0.9
    process_profile_alphat: float = 1.4
    process_profile_tbeta: float = 2.0
    process_density_pedestal_radius: float = 0.95
    process_temperature_pedestal_radius: float = 0.95
    process_density_pedestal_greenwald_fraction: float = 0.8
    process_density_separatrix_greenwald_fraction: float = 0.1
    process_temperature_pedestal_keV: float = 4.5
    process_temperature_separatrix_keV: float = 0.10
    auxiliary_fast_ion_energy_mev: float = 1.0
    fast_ion_coulomb_log: float = 17.0
    synchrotron_wall_reflectivity: float = 0.80
    synchrotron_profile_beta: float = 1.5
    tf_superconductor_id: int = 0  # 0 REBCO, 1 Nb3Sn
    pf_superconductor_id: int = 0  # 0 REBCO, 1 Nb3Sn, 2 NbTi
    cs_superconductor_id: int = 1  # 0 REBCO, 1 Nb3Sn, 2 NbTi
    blanket_concept_id: int = 0  # 0 LiPb/WCLL, 1 FLiBe, 2 HCPB
    divertor_heat_load_model_id: int = 1  # 0 fixed lambda, 1 Eich-like Bp scaling


class NumericalOptions(NamedTuple):
    """Numerical discretization and algorithmic options."""

    profile_num_points: int = 96


class GeometryState(NamedTuple):
    minor_radius_m: Array
    inverse_aspect_ratio: Array
    plasma_cross_section_m2: Array
    plasma_volume_m3: Array
    plasma_surface_area_m2: Array
    first_wall_area_m2: Array
    inboard_first_wall_area_m2: Array
    outboard_first_wall_area_m2: Array
    inboard_lcfs_radius_m: Array
    outboard_lcfs_radius_m: Array
    inboard_sol_width_m: Array
    outboard_sol_width_m: Array
    inboard_first_wall_thickness_m: Array
    outboard_first_wall_thickness_m: Array
    inboard_blanket_thickness_m: Array
    outboard_blanket_thickness_m: Array
    inboard_shield_thickness_m: Array
    outboard_shield_thickness_m: Array
    inboard_vacuum_vessel_thickness_m: Array
    outboard_vacuum_vessel_thickness_m: Array
    inboard_thermal_gap_m: Array
    outboard_thermal_gap_m: Array
    inboard_tf_coil_thickness_m: Array
    outboard_tf_coil_thickness_m: Array
    inboard_non_tf_build_m: Array
    outboard_non_tf_build_m: Array
    inboard_total_build_m: Array
    outboard_total_build_m: Array
    inboard_plasma_edge_m: Array
    outboard_first_wall_radius_m: Array
    inboard_tf_inner_radius_m: Array
    inboard_tf_centroid_radius_m: Array
    inboard_tf_outer_radius_m: Array
    outboard_tf_inner_radius_m: Array
    outboard_tf_centroid_radius_m: Array
    outboard_tf_outer_radius_m: Array
    tf_coil_height_m: Array
    machine_outer_radius_m: Array
    radial_grid: Array
    volume_prime_m3: Array
    shell_volumes_m3: Array
    volume_weights: Array


class PlasmaState(NamedTuple):
    electron_density_m3: Array
    greenwald_density_m3: Array
    ion_density_m3: Array
    thermal_pressure_pa: Array
    fast_ion_pressure_pa: Array
    beta_toroidal: Array
    thermal_beta_toroidal: Array
    fast_ion_beta_toroidal: Array
    alpha_fast_ion_beta_toroidal: Array
    auxiliary_fast_ion_beta_toroidal: Array
    beta_n: Array
    beta_p: Array
    q95: Array
    stored_energy_mj: Array
    confinement_time_s: Array
    transport_loss_mw: Array
    ipb98_transport_loss_mw: Array
    power_balance_transport_loss_mw: Array
    fusion_power_mw: Array
    alpha_power_mw: Array
    neutron_power_mw: Array
    bremsstrahlung_mw: Array
    synchrotron_mw: Array
    radiation_power_mw: Array
    ohmic_power_mw: Array
    bootstrap_fraction: Array
    bootstrap_current_ma: Array
    current_drive_current_ma: Array
    fusion_gain: Array
    radial_grid: Array
    volume_weights: Array
    electron_temperature_profile_keV: Array
    ion_temperature_profile_keV: Array
    electron_density_profile_m3: Array
    thermal_pressure_profile_pa: Array
    total_current_density_profile_a_m2: Array
    inductive_current_density_profile_a_m2: Array
    bootstrap_current_density_profile_a_m2: Array
    current_drive_current_density_profile_a_m2: Array
    resistivity_profile_ohm_m: Array
    ohmic_power_density_profile_mw_m3: Array
    pedestal_width: Array
    pedestal_pressure_pa: Array
    pedestal_temperature_keV: Array
    core_temperature_scale_keV: Array
    synchrotron_no_reflect_mw: Array
    synchrotron_optical_depth: Array
    synchrotron_reflectivity_factor: Array
    synchrotron_profile_beta: Array
    alpha_slowing_down_time_s: Array
    auxiliary_slowing_down_time_s: Array
    magnetic_shear_edge: Array
    magnetic_shear_95: Array
    alpha_critical_shear: Array
    bootstrap_collisionality: Array
    bootstrap_pressure_gradient_factor: Array
    current_drive_efficiency_20: Array


class MagnetState(NamedTuple):
    peak_tf_field_t: Array
    tf_required_ampere_turns_a: Array
    tf_engineering_current_density_a_m2: Array
    tf_allowable_engineering_current_density_a_m2: Array
    tf_winding_pack_area_m2: Array
    tf_structural_area_m2: Array
    tf_superconductor_area_m2: Array
    tf_superconductor_current_density_a_m2: Array
    tf_superconductor_critical_current_density_a_m2: Array
    tf_superconductor_temperature_k: Array
    tf_superconductor_strain: Array
    tf_field_derating: Array
    tf_temperature_derating: Array
    tf_strain_derating: Array
    tf_structural_fraction: Array
    tf_winding_pack_fraction: Array
    tf_magnetic_pressure_pa: Array
    tf_centering_force_per_coil_mn: Array
    tf_stress_pa: Array
    tf_stored_energy_gj: Array
    tf_nuclear_heat_mw: Array
    cryogenic_static_heat_mw: Array
    cryogenic_nuclear_heat_mw: Array
    cryogenic_current_lead_joint_heat_mw: Array
    cryogenic_static_electric_power_mw: Array
    cryogenic_nuclear_electric_power_mw: Array
    cryogenic_current_lead_joint_electric_power_mw: Array
    cryogenic_electric_power_mw: Array
    pf_vertical_field_t: Array
    pf_required_equilibrium_ampere_turns_a: Array
    pf_coil_radii_m: Array
    pf_coil_z_m: Array
    pf_coil_ampere_turns_a: Array
    pf_coil_current_density_a_m2: Array
    pf_coil_peak_field_t: Array
    pf_coil_stress_pa: Array
    pf_max_current_density_a_m2: Array
    pf_current_density_limit_a_m2: Array
    pf_superconductor_critical_current_density_a_m2: Array
    pf_superconductor_temperature_k: Array
    pf_superconductor_strain: Array
    pf_field_derating: Array
    pf_temperature_derating: Array
    pf_strain_derating: Array
    pf_current_density_utilization: Array
    pf_max_peak_field_t: Array
    pf_max_stress_pa: Array
    cs_required_ampere_turns_a: Array
    cs_available_ampere_turns_a: Array
    cs_startup_flux_required_wb: Array
    cs_flat_top_flux_required_wb: Array
    cs_flux_required_wb: Array
    cs_flux_capacity_wb: Array
    cs_peak_field_t: Array
    cs_current_density_a_m2: Array
    cs_current_density_limit_a_m2: Array
    cs_superconductor_critical_current_density_a_m2: Array
    cs_superconductor_temperature_k: Array
    cs_superconductor_strain: Array
    cs_field_derating: Array
    cs_temperature_derating: Array
    cs_strain_derating: Array
    cs_current_density_utilization: Array
    cs_stress_pa: Array


class NuclearState(NamedTuple):
    blanket_concept_id: Array
    neutron_attenuation: Array
    blanket_neutron_attenuation: Array
    shield_neutron_attenuation: Array
    blanket_thermal_power_mw: Array
    first_wall_neutron_heat_mw: Array
    first_wall_radiation_heat_mw: Array
    first_wall_alpha_heat_mw: Array
    first_wall_thermal_power_mw: Array
    shield_thermal_power_mw: Array
    first_wall_surface_heat_load_mw_m2: Array
    neutron_wall_loading_mw_m2: Array
    tf_neutron_wall_loading_mw_m2: Array
    tf_fluence_n_m2: Array
    tf_dpa_proxy: Array
    tbr: Array
    tbr_asymptote: Array
    tbr_scale_length_m: Array
    tritium_burn_rate_kg_per_year: Array
    tritium_breeding_rate_kg_per_year: Array
    tritium_surplus_kg_per_year: Array
    tritium_inventory_kg: Array
    tritium_doubling_time_years: Array
    blanket_pumping_power_mw: Array
    blanket_pumping_fraction: Array
    blanket_energy_multiplication: Array
    blanket_coverage_fraction: Array
    blanket_attenuation_per_m: Array
    shield_attenuation_per_m: Array
    blanket_volume_m3: Array
    shield_volume_m3: Array


class ExhaustState(NamedTuple):
    separatrix_power_mw: Array
    divertor_radiated_power_mw: Array
    divertor_target_power_mw: Array
    divertor_nuclear_heat_mw: Array
    divertor_radiation_heat_mw: Array
    divertor_heat_deposited_mw: Array
    divertor_poloidal_field_t: Array
    divertor_parallel_field_t: Array
    divertor_upstream_lambda_q_m: Array
    divertor_integral_lambda_m: Array
    divertor_parallel_heat_flux_mw_m2: Array
    divertor_flux_expansion: Array
    divertor_peak_to_integral_factor: Array
    divertor_effective_area_m2: Array
    divertor_integral_heat_flux_mw_m2: Array
    peak_divertor_heat_flux_mw_m2: Array
    divertor_target_temperature_c: Array
    divertor_lifetime_fpy: Array


class PowerPlantState(NamedTuple):
    recoverable_thermal_power_mw: Array
    gross_electric_power_mw: Array
    time_averaged_gross_electric_power_mw: Array
    heating_wallplug_power_mw: Array
    current_drive_wallplug_power_mw: Array
    blanket_pumping_power_mw: Array
    first_wall_pumping_power_mw: Array
    divertor_pumping_power_mw: Array
    coolant_pumping_power_mw: Array
    coolant_mechanical_pumping_power_mw: Array
    vacuum_pumping_power_mw: Array
    tritium_plant_power_mw: Array
    cryogenic_static_power_mw: Array
    cryogenic_nuclear_power_mw: Array
    cryogenic_current_lead_joint_power_mw: Array
    cryogenic_electric_power_mw: Array
    plant_base_auxiliary_power_mw: Array
    tf_electric_supply_power_mw: Array
    pf_cs_electric_supply_power_mw: Array
    fixed_house_load_mw: Array
    pulse_duty_factor: Array
    recirculating_power_mw: Array
    recirculating_fraction: Array
    flat_top_net_electric_power_mw: Array
    time_averaged_net_electric_power_mw: Array
    net_electric_power_mw: Array


class EconomicsState(NamedTuple):
    tf_coil_cost_busd: Array
    pf_coil_cost_busd: Array
    magnet_cost_busd: Array
    nuclear_island_cost_busd: Array
    balance_of_plant_cost_busd: Array
    hcd_cost_busd: Array
    direct_capital_cost_busd: Array
    indirect_cost_busd: Array
    contingency_cost_busd: Array
    total_capital_cost_busd: Array
    availability: Array
    divertor_availability_factor: Array
    neutron_availability_factor: Array
    magnet_availability_factor: Array
    pulsed_availability_factor: Array
    blanket_lifetime_fpy: Array
    divertor_lifetime_fpy: Array
    magnet_lifetime_fpy: Array
    scheduled_downtime_days_per_year: Array
    capacity_factor: Array
    annualized_capital_cost_musd: Array
    annual_om_cost_musd: Array
    annual_maintenance_cost_musd: Array
    annual_electricity_mwh: Array
    capital_coe_usd_mwh: Array
    om_coe_usd_mwh: Array
    maintenance_coe_usd_mwh: Array
    coe_usd_mwh: Array


class ResidualState(NamedTuple):
    plasma_power_balance: Array
    plasma_current_balance: Array
    target_net_power: Array
    stored_energy_closure: Array
    transport_loss_closure: Array
    bootstrap_current_closure: Array
    current_drive_current_closure: Array
    net_electric_power_closure: Array
    tf_peak_field_closure: Array
    tbr_closure: Array
    divertor_heat_flux_closure: Array


class MarginState(NamedTuple):
    q95: Array
    beta_n: Array
    greenwald: Array
    fusion_gain: Array
    target_net_power: Array
    inboard_space: Array
    tf_peak_field: Array
    tf_current_density: Array
    tf_stress: Array
    tf_nuclear_heat: Array
    pf_equilibrium_ampere_turns: Array
    pf_current_density: Array
    pf_peak_field: Array
    pf_stress: Array
    cs_flux_swing: Array
    cs_peak_field: Array
    cs_current_density: Array
    cs_stress: Array
    tbr: Array
    tf_fluence: Array
    tf_dpa: Array
    tritium_doubling_time: Array
    first_wall_heat_load: Array
    divertor_heat_flux: Array
    divertor_target_temperature: Array
    divertor_lifetime: Array
    recirculating_fraction: Array


class ReactorResult(NamedTuple):
    geometry: GeometryState
    plasma: PlasmaState
    magnets: MagnetState
    nuclear: NuclearState
    exhaust: ExhaustState
    power: PowerPlantState
    economics: EconomicsState
    residuals: ResidualState
    margins: MarginState


class VariableBounds(NamedTuple):
    lower: DesignVariables
    upper: DesignVariables


class ReactorVariableBounds(NamedTuple):
    lower: ReactorVariables
    upper: ReactorVariables


class OptimizationConfig(NamedTuple):
    objective_terms: tuple[str, ...] = ("major_radius",)
    objective_weights: tuple[float, ...] = ()
    enabled_equalities: tuple[str, ...] = ()
    disabled_equalities: tuple[str, ...] = ()
    enabled_inequalities: tuple[str, ...] = ()
    disabled_inequalities: tuple[str, ...] = ()
    equality_tolerance: float = 1.0e-3
    inequality_tolerance: float = 1.0e-8


class ReactorProblem(NamedTuple):
    initial_design: DesignVariables
    initial_variables: ReactorVariables
    bounds: VariableBounds
    variable_bounds: ReactorVariableBounds
    technology: TechnologyParameters
    config: ReactorConfig
    optimization: OptimizationConfig
    numerics: NumericalOptions
