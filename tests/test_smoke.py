import jax
import jax.numpy as jnp
import pytest

from tokasys import (
    analyze_constraints,
    default_problem,
    evaluate_reactor,
    format_constraint_report,
)
from tokasys.core.constants import HOURS_PER_YEAR, KEV_TO_J
from tokasys.core.variables import normalize_reactor_variables
from tokasys.models.constraints import equality_vector, inequality_vector
from tokasys.plasma.profiles import density_at_normalized_radius_m3
from tokasys.solvers.config import apply_optimization_config
from tokasys.solvers.interfaces import make_vector_functions


def test_reactor_evaluation_is_finite():
    problem = default_problem()
    result = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
        problem.numerics,
    )
    leaves = jax.tree.leaves(result)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert equality_vector(result).shape == (11,)
    assert inequality_vector(result).shape == (27,)
    assert result.geometry.radial_grid.shape == (problem.numerics.profile_num_points,)
    assert result.geometry.volume_weights.shape == (problem.numerics.profile_num_points,)
    assert jnp.isclose(
        jnp.sum(result.geometry.shell_volumes_m3),
        result.geometry.plasma_volume_m3,
    )
    assert jnp.isclose(
        result.geometry.inboard_first_wall_area_m2
        + result.geometry.outboard_first_wall_area_m2,
        result.geometry.first_wall_area_m2,
    )
    assert jnp.allclose(result.geometry.volume_weights, result.plasma.volume_weights)
    assert result.plasma.radial_grid.shape == (problem.numerics.profile_num_points,)
    assert result.plasma.electron_temperature_profile_keV.shape == (
        problem.numerics.profile_num_points,
    )
    assert result.plasma.electron_density_profile_m3.shape == (
        problem.numerics.profile_num_points,
    )
    coarse = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
        problem.numerics._replace(profile_num_points=32),
    )
    assert coarse.geometry.radial_grid.shape == (32,)
    density_average = jnp.sum(
        result.plasma.electron_density_profile_m3 * result.plasma.volume_weights
    )
    assert jnp.isclose(density_average, result.plasma.electron_density_m3)
    pedestal_top_index = jnp.argmin(
        jnp.abs(result.plasma.radial_grid - (1.0 - result.plasma.pedestal_width))
    )
    assert result.plasma.electron_density_profile_m3[pedestal_top_index] > (
        result.plasma.electron_density_profile_m3[-1]
    )
    pedestal_top_density = density_at_normalized_radius_m3(
        1.0 - result.plasma.pedestal_width,
        result.plasma.radial_grid,
        result.plasma.volume_weights,
        result.plasma.electron_density_m3,
        result.plasma.pedestal_width,
        problem.config,
    )
    central_to_pedestal_density = (
        result.plasma.electron_density_profile_m3[0] / pedestal_top_density
    )
    assert central_to_pedestal_density > 1.45
    assert central_to_pedestal_density < 1.55
    assert result.plasma.bootstrap_fraction >= 0.0
    assert result.plasma.bootstrap_fraction <= 0.95
    assert result.plasma.bootstrap_collisionality >= 0.0
    assert result.plasma.current_drive_efficiency_20 > 0.0
    assert result.plasma.synchrotron_profile_beta > 0.0
    assert result.plasma.thermal_beta_toroidal > 0.0
    assert result.plasma.fast_ion_beta_toroidal > 0.0
    assert jnp.isclose(
        result.plasma.beta_toroidal,
        result.plasma.thermal_beta_toroidal
        + result.plasma.fast_ion_beta_toroidal,
    )
    assert jnp.isclose(
        result.plasma.fast_ion_beta_toroidal,
        result.plasma.alpha_fast_ion_beta_toroidal
        + result.plasma.auxiliary_fast_ion_beta_toroidal,
    )
    assert result.plasma.alpha_slowing_down_time_s > 0.0
    assert result.plasma.auxiliary_slowing_down_time_s > 0.0
    assert result.plasma.total_current_density_profile_a_m2.shape == (
        problem.numerics.profile_num_points,
    )
    assert result.plasma.inductive_current_density_profile_a_m2.shape == (
        problem.numerics.profile_num_points,
    )
    assert result.plasma.ohmic_power_density_profile_mw_m3.shape == (
        problem.numerics.profile_num_points,
    )
    assert jnp.isclose(
        result.plasma.ohmic_power_mw,
        result.geometry.plasma_volume_m3
        * jnp.sum(
            result.plasma.ohmic_power_density_profile_mw_m3
            * result.plasma.volume_weights
        ),
    )
    expected_power_balance = (
        result.plasma.alpha_power_mw
        + problem.initial_design.auxiliary_heating_mw
        + problem.initial_design.current_drive_power_mw
        + result.plasma.ohmic_power_mw
        - result.plasma.radiation_power_mw
        - result.plasma.transport_loss_mw
    ) / max(problem.technology.net_electric_target_mw, 100.0)
    assert jnp.isclose(result.residuals.plasma_power_balance, expected_power_balance)
    assert problem.config.pedestal_width_coefficient == 0.076
    assert problem.config.pedestal_fixed_width == 0.04
    assert problem.config.energy_closure_mode in {
        "self_consistent",
        "fixed_point",
        "full_space",
    }
    assert result.plasma.magnetic_shear_edge > 0.0
    assert result.plasma.magnetic_shear_95 > 0.0
    assert result.plasma.alpha_critical_shear == jnp.maximum(
        result.plasma.magnetic_shear_edge,
        result.plasma.magnetic_shear_95,
    )
    assert jnp.isclose(
        result.margins.inboard_space,
        (
            result.geometry.inboard_tf_inner_radius_m
            - problem.initial_design.cs_thickness_m
        )
        / problem.initial_design.major_radius_m,
    )
    assert result.plasma.pedestal_width > 0.01
    assert jnp.isclose(result.plasma.pedestal_width, problem.config.pedestal_fixed_width)
    assert result.plasma.pedestal_temperature_keV > 0.0
    assert result.magnets.tf_winding_pack_area_m2 > 0.0
    assert result.magnets.tf_structural_area_m2 > 0.0
    assert result.magnets.tf_allowable_engineering_current_density_a_m2 > 0.0
    assert result.magnets.tf_superconductor_critical_current_density_a_m2 > 0.0
    assert result.magnets.tf_field_derating > 0.0
    assert result.magnets.tf_temperature_derating > 0.0
    assert result.magnets.tf_strain_derating > 0.0
    assert result.magnets.tf_centering_force_per_coil_mn > 0.0
    assert result.magnets.pf_vertical_field_t > 0.0
    assert result.magnets.pf_required_equilibrium_ampere_turns_a > 0.0
    assert result.magnets.pf_current_density_limit_a_m2 > 0.0
    assert result.magnets.pf_superconductor_critical_current_density_a_m2 > 0.0
    assert result.magnets.cs_required_ampere_turns_a > 0.0
    assert result.magnets.cs_available_ampere_turns_a > 0.0
    assert result.magnets.cs_flux_required_wb > 0.0
    assert result.magnets.cs_flux_capacity_wb > 0.0
    assert result.magnets.cs_peak_field_t > 0.0
    assert result.magnets.cs_current_density_a_m2 > 0.0
    assert result.magnets.cs_current_density_limit_a_m2 > 0.0
    assert result.magnets.cs_superconductor_critical_current_density_a_m2 > 0.0
    assert result.magnets.cs_stress_pa > 0.0
    assert result.magnets.cryogenic_static_electric_power_mw > 0.0
    assert result.magnets.cryogenic_nuclear_electric_power_mw > 0.0
    assert result.magnets.cryogenic_current_lead_joint_electric_power_mw > 0.0
    assert jnp.isclose(
        result.magnets.cryogenic_electric_power_mw,
        result.magnets.cryogenic_static_electric_power_mw
        + result.magnets.cryogenic_nuclear_electric_power_mw
        + result.magnets.cryogenic_current_lead_joint_electric_power_mw,
    )
    assert result.nuclear.blanket_neutron_attenuation > 0.0
    assert result.nuclear.shield_neutron_attenuation > 0.0
    assert result.nuclear.blanket_energy_multiplication > 0.0
    assert result.nuclear.blanket_coverage_fraction > 0.0
    assert result.nuclear.tbr_asymptote > 0.0
    assert result.nuclear.tbr_scale_length_m > 0.0
    assert result.nuclear.first_wall_surface_heat_load_mw_m2 > 0.0
    assert result.nuclear.neutron_wall_loading_mw_m2 > 0.0
    assert result.nuclear.tf_neutron_wall_loading_mw_m2 > 0.0
    assert result.nuclear.tf_fluence_n_m2 > 0.0
    assert result.nuclear.tf_dpa_proxy > 0.0
    assert result.nuclear.tritium_burn_rate_kg_per_year > 0.0
    assert result.nuclear.tritium_breeding_rate_kg_per_year > (
        result.nuclear.tritium_burn_rate_kg_per_year
    )
    assert result.nuclear.tritium_surplus_kg_per_year > 0.0
    assert result.nuclear.tritium_inventory_kg > 0.0
    assert result.nuclear.tritium_doubling_time_years > 0.0
    assert result.nuclear.blanket_pumping_power_mw > 0.0
    assert jnp.isclose(
        result.nuclear.first_wall_surface_heat_load_mw_m2,
        result.nuclear.first_wall_thermal_power_mw / result.geometry.first_wall_area_m2,
    )
    assert jnp.isclose(
        result.nuclear.neutron_wall_loading_mw_m2,
        result.plasma.neutron_power_mw / result.geometry.first_wall_area_m2,
    )
    assert result.exhaust.divertor_radiated_power_mw > 0.0
    assert result.exhaust.divertor_poloidal_field_t > 0.0
    assert result.exhaust.divertor_parallel_field_t > 0.0
    assert result.exhaust.divertor_upstream_lambda_q_m > 0.0
    assert result.exhaust.divertor_integral_lambda_m > 0.0
    assert result.exhaust.divertor_parallel_heat_flux_mw_m2 > 0.0
    assert result.exhaust.divertor_integral_heat_flux_mw_m2 > 0.0
    assert result.exhaust.peak_divertor_heat_flux_mw_m2 >= (
        result.exhaust.divertor_integral_heat_flux_mw_m2
    )
    assert result.exhaust.divertor_target_temperature_c > (
        problem.technology.divertor_coolant_temperature_c
    )
    assert result.exhaust.divertor_lifetime_fpy > 0.0
    assert result.power.blanket_pumping_power_mw > result.nuclear.blanket_pumping_power_mw
    assert result.power.first_wall_pumping_power_mw > 0.0
    assert result.power.divertor_pumping_power_mw > 0.0
    assert jnp.isclose(
        result.power.coolant_pumping_power_mw,
        result.power.blanket_pumping_power_mw
        + result.power.first_wall_pumping_power_mw
        + result.power.divertor_pumping_power_mw,
    )
    assert result.power.coolant_pumping_power_mw > (
        result.power.coolant_mechanical_pumping_power_mw
    )
    assert result.power.vacuum_pumping_power_mw > 0.0
    assert result.power.tritium_plant_power_mw > 0.0
    assert jnp.isclose(
        result.power.cryogenic_electric_power_mw,
        result.power.cryogenic_static_power_mw
        + result.power.cryogenic_nuclear_power_mw
        + result.power.cryogenic_current_lead_joint_power_mw,
    )
    assert result.power.plant_base_auxiliary_power_mw > 0.0
    assert result.power.tf_electric_supply_power_mw == 0.0
    assert result.power.pf_cs_electric_supply_power_mw == 0.0
    assert result.power.pulse_duty_factor == 1.0
    assert jnp.isclose(
        result.power.net_electric_power_mw,
        result.power.time_averaged_net_electric_power_mw,
    )
    assert result.economics.availability <= problem.technology.availability
    assert result.economics.availability >= problem.technology.availability_min
    assert jnp.isclose(result.economics.capacity_factor, result.economics.availability)
    assert result.economics.direct_capital_cost_busd > 0.0
    assert result.economics.indirect_cost_busd > 0.0
    assert result.economics.contingency_cost_busd > 0.0
    assert result.economics.annualized_capital_cost_musd > 0.0
    assert result.economics.annual_om_cost_musd > 0.0
    assert result.economics.annual_maintenance_cost_musd > 0.0
    assert jnp.isclose(
        result.economics.annual_electricity_mwh,
        jnp.maximum(result.power.flat_top_net_electric_power_mw, 1.0e-3)
        * HOURS_PER_YEAR
        * problem.technology.availability,
    )
    assert result.economics.blanket_lifetime_fpy > 0.0
    assert result.economics.divertor_lifetime_fpy > 0.0
    assert result.economics.magnet_lifetime_fpy > 0.0
    assert result.economics.scheduled_downtime_days_per_year > 0.0
    assert jnp.isclose(
        result.economics.total_capital_cost_busd,
        result.economics.direct_capital_cost_busd
        + result.economics.indirect_cost_busd
        + result.economics.contingency_cost_busd,
    )
    assert jnp.isclose(
        result.economics.coe_usd_mwh,
        result.economics.capital_coe_usd_mwh
        + result.economics.om_coe_usd_mwh
        + result.economics.maintenance_coe_usd_mwh,
    )
    assert result.geometry.inboard_non_tf_build_m > 0.0
    assert result.geometry.outboard_non_tf_build_m > 0.0
    assert result.geometry.inboard_total_build_m > result.geometry.inboard_non_tf_build_m
    assert result.geometry.outboard_total_build_m > result.geometry.outboard_non_tf_build_m
    assert jnp.isclose(
        result.geometry.inboard_total_build_m,
        result.geometry.inboard_non_tf_build_m
        + result.geometry.inboard_tf_coil_thickness_m,
    )
    assert jnp.isclose(
        result.geometry.outboard_total_build_m,
        result.geometry.outboard_non_tf_build_m
        + result.geometry.outboard_tf_coil_thickness_m,
    )
    assert jnp.isclose(
        result.geometry.machine_outer_radius_m,
        result.geometry.outboard_lcfs_radius_m + result.geometry.outboard_total_build_m,
    )
    assert jnp.isclose(
        result.geometry.inboard_tf_centroid_radius_m,
        0.5
        * (
            result.geometry.inboard_tf_inner_radius_m
            + result.geometry.inboard_tf_outer_radius_m
        ),
    )
    assert jnp.isclose(
        result.geometry.outboard_tf_centroid_radius_m,
        0.5
        * (
            result.geometry.outboard_tf_inner_radius_m
            + result.geometry.outboard_tf_outer_radius_m
        ),
    )


def test_reduced_availability_tracks_component_utilization():
    problem = default_problem()
    base = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
    ).economics
    high_power = evaluate_reactor(
        problem.initial_design._replace(target_temperature_keV=14.0),
        problem.technology,
        problem.config,
    ).economics

    assert high_power.availability < base.availability
    assert high_power.blanket_lifetime_fpy < base.blanket_lifetime_fpy
    assert high_power.divertor_lifetime_fpy < base.divertor_lifetime_fpy
    assert high_power.scheduled_downtime_days_per_year > (
        base.scheduled_downtime_days_per_year
    )


def test_process_parabolic_profile_option_matches_scalar_targets():
    problem = default_problem()
    config = problem.config._replace(
        energy_closure_mode="full_space",
        plasma_profile_model="process_parabolic",
        process_profile_alphan=0.9,
        process_profile_alphat=1.4,
    )
    result = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        config,
        problem.numerics,
    )
    density_average = jnp.sum(
        result.plasma.electron_density_profile_m3 * result.plasma.volume_weights
    )
    expected_target_energy_mj = (
        1.5
        * (
            result.plasma.electron_density_m3
            * problem.initial_design.target_temperature_keV
            + result.plasma.ion_density_m3
            * problem.technology.ion_to_electron_temperature_ratio
            * problem.initial_design.target_temperature_keV
        )
        * KEV_TO_J
        * result.geometry.plasma_volume_m3
        / 1.0e6
    )

    assert jnp.isclose(density_average, result.plasma.electron_density_m3)
    assert jnp.isclose(result.plasma.stored_energy_mj, expected_target_energy_mj)
    assert result.plasma.pedestal_width == 0.0
    assert result.plasma.electron_density_profile_m3[0] > (
        result.plasma.electron_density_m3
    )
    assert result.plasma.electron_temperature_profile_keV[0] > (
        problem.initial_design.target_temperature_keV
    )
    assert result.plasma.electron_temperature_profile_keV[-1] < (
        0.01 * result.plasma.electron_temperature_profile_keV[0]
    )


def test_process_pedestal_profile_option_matches_scalar_density_target():
    problem = default_problem()
    config = problem.config._replace(
        energy_closure_mode="full_space",
        plasma_profile_model="process_pedestal",
        process_density_pedestal_radius=0.95,
        process_temperature_pedestal_radius=0.95,
        process_temperature_pedestal_keV=4.5,
        process_temperature_separatrix_keV=0.1,
    )
    result = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        config,
        problem.numerics,
    )
    density_average = jnp.sum(
        result.plasma.electron_density_profile_m3 * result.plasma.volume_weights
    )
    pedestal_temperature = jnp.interp(
        config.process_temperature_pedestal_radius,
        result.plasma.radial_grid,
        result.plasma.electron_temperature_profile_keV,
    )

    assert jnp.isclose(density_average, result.plasma.electron_density_m3)
    assert jnp.isclose(result.plasma.pedestal_width, 0.05)
    assert jnp.isclose(
        result.plasma.pedestal_temperature_keV,
        config.process_temperature_pedestal_keV,
    )
    assert jnp.abs(pedestal_temperature - config.process_temperature_pedestal_keV) < 0.3
    assert result.plasma.electron_temperature_profile_keV[0] > (
        config.process_temperature_pedestal_keV
    )
    assert result.plasma.electron_temperature_profile_keV[-1] < (
        config.process_temperature_pedestal_keV
    )


def test_energy_closure_modes_are_distinct():
    problem = default_problem()

    self_consistent = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(energy_closure_mode="self_consistent"),
        problem.numerics,
    )
    fixed_point_alias = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(energy_closure_mode="fixed_point"),
        problem.numerics,
    )
    full_space = evaluate_reactor(
        problem.initial_variables,
        problem.technology,
        problem.config._replace(energy_closure_mode="full_space"),
        problem.numerics,
    )

    assert jnp.isclose(self_consistent.residuals.plasma_power_balance, 0.0)
    assert jnp.isclose(
        self_consistent.plasma.transport_loss_mw,
        self_consistent.plasma.power_balance_transport_loss_mw,
    )
    assert jnp.isclose(
        self_consistent.plasma.stored_energy_mj,
        fixed_point_alias.plasma.stored_energy_mj,
    )
    assert jnp.isclose(
        full_space.plasma.stored_energy_mj,
        problem.initial_variables.closure.stored_energy_mj,
    )
    assert jnp.isclose(
        full_space.plasma.transport_loss_mw,
        problem.initial_variables.closure.transport_loss_mw,
    )
    assert not jnp.isclose(
        full_space.plasma.transport_loss_mw,
        full_space.plasma.ipb98_transport_loss_mw,
    )
    assert equality_vector(self_consistent, problem.config).shape == (10,)
    assert equality_vector(
        full_space,
        problem.config._replace(energy_closure_mode="full_space"),
    ).shape == (10,)


def test_jit_gradient_and_jacobians():
    problem = default_problem()
    funcs = make_vector_functions(problem)
    u0 = normalize_reactor_variables(problem.initial_variables, problem.variable_bounds)
    value, grad = funcs["objective_value_and_grad"](u0)
    eq_jac = funcs["equality_jacobian"](u0)
    ineq_jac = funcs["inequality_jacobian"](u0)

    assert jnp.isfinite(value)
    assert grad.shape == u0.shape
    assert eq_jac.shape == (10, u0.size)
    assert ineq_jac.shape == (27, u0.size)
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.all(jnp.isfinite(eq_jac))
    assert jnp.all(jnp.isfinite(ineq_jac))
    assert jnp.all(jnp.linalg.norm(eq_jac, axis=1) > 1.0e-10)

    full_space_problem = problem._replace(
        config=problem.config._replace(energy_closure_mode="full_space")
    )
    full_space_funcs = make_vector_functions(full_space_problem)
    full_space_jac = full_space_funcs["equality_jacobian"](u0)
    assert full_space_jac.shape == (10, u0.size)
    assert jnp.all(jnp.linalg.norm(full_space_jac, axis=1) > 1.0e-10)


def test_optimization_config_selects_objectives_and_constraints():
    problem = default_problem()
    custom = problem._replace(
        optimization=problem.optimization._replace(
            objective_terms=("coe",),
            enabled_equalities=("target_net_power",),
            enabled_inequalities=("q95", "tbr"),
        )
    )
    funcs = make_vector_functions(custom)
    u0 = normalize_reactor_variables(custom.initial_variables, custom.variable_bounds)
    result = funcs["result"](u0)
    objective = funcs["objective"](u0)
    equalities = funcs["equalities"](u0)
    inequalities = funcs["inequalities"](u0)
    eq_jac = funcs["equality_jacobian"](u0)
    ineq_jac = funcs["inequality_jacobian"](u0)

    assert jnp.isclose(objective, result.economics.coe_usd_mwh / 100.0)
    assert equalities.shape == (1,)
    assert inequalities.shape == (2,)
    assert eq_jac.shape == (1, u0.size)
    assert ineq_jac.shape == (2, u0.size)


def test_optimization_config_applies_custom_objective_weights():
    problem = default_problem()
    custom = problem._replace(
        optimization=problem.optimization._replace(
            objective_terms=("major_radius", "coe", "negative_net_power"),
            objective_weights=(0.2, 0.005, 0.003),
        )
    )
    funcs = make_vector_functions(custom)
    u0 = normalize_reactor_variables(custom.initial_variables, custom.variable_bounds)
    result = funcs["result"](u0)
    objective = funcs["objective"](u0)

    expected = (
        0.2 * custom.initial_design.major_radius_m
        + 0.005 * result.economics.coe_usd_mwh
        - 0.003 * result.power.net_electric_power_mw
    )
    assert jnp.isclose(objective, expected)


def test_optimization_config_boolean_objective_uses_default_weight(tmp_path):
    config_path = tmp_path / "legacy_objective_config.json"
    config_path.write_text(
        """
        {
          "objectives": {
            "major_radius": false,
            "coe": true,
            "negative_net_power": false
          }
        }
        """,
        encoding="utf-8",
    )

    custom = apply_optimization_config(default_problem(), config_path)

    assert custom.optimization.objective_terms == ("coe",)
    assert custom.optimization.objective_weights == pytest.approx((0.01,))


def test_optimization_config_file_selects_terms(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          // Minimize COE only.
          "objectives": {
            "major_radius": false,
            "coe": { "enabled": true, "weight": 0.025 },
            "negative_net_power": false
          },
          "equalities": {
            // Keep only target net power as a hard equality.
            "target_net_power": { "enabled": true, "target": 550.0 }
          },
          "inequalities": {
            // Keep only four hard inequality constraints.
            "q95": { "enabled": true, "limit": 3.2 },
            "tbr": { "enabled": true, "limit": 1.15 },
            "tf_dpa": { "enabled": true, "limit": 1.0e-3 },
            "first_wall_heat_load": { "enabled": true, "limit": 1.0 }
          },
          "reactor_config": {
            "tf_superconductor_id": 1,
            "pf_superconductor_id": 2,
            "cs_superconductor_id": 2,
            "blanket_concept_id": 2,
            "cost_model_id": 2,
            "divertor_heat_load_model_id": 0
          },
          "bounds": {
            "major_radius_m": { "lower": 5.0, "upper": 11.0 }
          },
          "closure_bounds": {
            "net_electric_power_mw": { "lower": -250.0, "upper": 2500.0 }
          },
          "tolerances": {
            "equality": 1e-3,
            "inequality": 1e-6
          }
        }
        """,
        encoding="utf-8",
    )
    custom = apply_optimization_config(default_problem(), config_path)
    funcs = make_vector_functions(custom)
    u0 = normalize_reactor_variables(custom.initial_variables, custom.variable_bounds)

    assert custom.optimization.objective_terms == ("coe",)
    assert custom.optimization.objective_weights == pytest.approx((0.025,))
    assert custom.optimization.enabled_equalities == ("target_net_power",)
    assert custom.optimization.enabled_inequalities == (
        "q95",
        "tbr",
        "tf_dpa",
        "first_wall_heat_load",
    )
    assert custom.technology.net_electric_target_mw == pytest.approx(550.0)
    assert custom.technology.q95_min == pytest.approx(3.2)
    assert custom.technology.tbr_min == pytest.approx(1.15)
    assert custom.technology.tf_dpa_limit == pytest.approx(1.0e-3)
    assert custom.technology.first_wall_surface_heat_load_limit_mw_m2 == pytest.approx(1.0)
    assert custom.config.tf_superconductor_id == 1
    assert custom.config.pf_superconductor_id == 2
    assert custom.config.cs_superconductor_id == 2
    assert custom.config.blanket_concept_id == 2
    assert custom.config.cost_model_id == 2
    assert custom.config.divertor_heat_load_model_id == 0
    assert float(custom.bounds.lower.major_radius_m) == pytest.approx(5.0)
    assert float(custom.bounds.upper.major_radius_m) == pytest.approx(11.0)
    assert float(custom.variable_bounds.lower.design.major_radius_m) == pytest.approx(5.0)
    assert float(custom.variable_bounds.upper.design.major_radius_m) == pytest.approx(11.0)
    assert float(custom.variable_bounds.lower.closure.net_electric_power_mw) == pytest.approx(-250.0)
    assert float(custom.variable_bounds.upper.closure.net_electric_power_mw) == pytest.approx(2500.0)
    assert custom.optimization.equality_tolerance == pytest.approx(1.0e-3)
    result = funcs["result"](u0)
    assert jnp.isclose(
        funcs["objective"](u0),
        0.025 * result.economics.coe_usd_mwh,
    )
    assert funcs["equalities"](u0).shape == (1,)
    assert funcs["inequalities"](u0).shape == (4,)


def test_geometry_volume_tracks_elongation_not_triangularity():
    problem = default_problem()
    base = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
    ).geometry
    elongated_design = problem.initial_design._replace(elongation=2.0)
    triangular_design = problem.initial_design._replace(triangularity=0.60)
    elongated = evaluate_reactor(
        elongated_design,
        problem.technology,
        problem.config,
    ).geometry
    triangular = evaluate_reactor(
        triangular_design,
        problem.technology,
        problem.config,
    ).geometry

    assert elongated.plasma_volume_m3 > base.plasma_volume_m3
    assert elongated.volume_prime_m3[-1] > base.volume_prime_m3[-1]
    assert jnp.isclose(triangular.plasma_volume_m3, base.plasma_volume_m3)
    assert jnp.allclose(triangular.volume_weights, base.volume_weights)


def test_current_drive_model_responds_to_power():
    problem = default_problem()
    driven = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
    ).plasma
    no_cd_design = problem.initial_design._replace(current_drive_power_mw=0.0)
    no_cd = evaluate_reactor(
        no_cd_design,
        problem.technology,
        problem.config,
    ).plasma

    assert driven.current_drive_current_ma > 0.0
    assert jnp.isclose(no_cd.current_drive_current_ma, 0.0)


def test_tf_material_choice_changes_current_density_limit():
    problem = default_problem()
    rebco = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(tf_superconductor_id=0),
    ).magnets
    nb3sn = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(tf_superconductor_id=1),
    ).magnets

    assert rebco.tf_allowable_engineering_current_density_a_m2 > (
        nb3sn.tf_allowable_engineering_current_density_a_m2
    )
    assert jnp.isclose(
        rebco.tf_engineering_current_density_a_m2,
        nb3sn.tf_engineering_current_density_a_m2,
    )


def test_pf_material_choice_changes_current_density_limit_independently():
    problem = default_problem()
    rebco_pf = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(tf_superconductor_id=0, pf_superconductor_id=0),
    ).magnets
    nb3sn_pf = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(tf_superconductor_id=0, pf_superconductor_id=1),
    ).magnets

    assert rebco_pf.pf_current_density_limit_a_m2 > (
        nb3sn_pf.pf_current_density_limit_a_m2
    )
    assert jnp.isclose(
        rebco_pf.tf_allowable_engineering_current_density_a_m2,
        nb3sn_pf.tf_allowable_engineering_current_density_a_m2,
    )
    assert jnp.isclose(
        rebco_pf.pf_max_current_density_a_m2,
        nb3sn_pf.pf_max_current_density_a_m2,
    )
    nbti_pf = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(tf_superconductor_id=0, pf_superconductor_id=2),
    ).magnets
    assert nbti_pf.pf_superconductor_temperature_k == pytest.approx(
        problem.technology.nbti_operating_temperature_k
    )
    assert not jnp.isclose(
        nbti_pf.pf_current_density_limit_a_m2,
        rebco_pf.pf_current_density_limit_a_m2,
    )


def test_cs_material_choice_changes_current_density_limit_independently():
    problem = default_problem()
    nb3sn_cs = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(tf_superconductor_id=0, cs_superconductor_id=1),
    ).magnets
    nbti_cs = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(tf_superconductor_id=0, cs_superconductor_id=2),
    ).magnets

    assert nbti_cs.cs_superconductor_temperature_k == pytest.approx(
        problem.technology.nbti_operating_temperature_k
    )
    assert not jnp.isclose(
        nbti_cs.cs_current_density_limit_a_m2,
        nb3sn_cs.cs_current_density_limit_a_m2,
    )
    assert jnp.isclose(
        nbti_cs.tf_allowable_engineering_current_density_a_m2,
        nb3sn_cs.tf_allowable_engineering_current_density_a_m2,
    )


def test_superconductor_jc_responds_to_temperature_and_strain():
    problem = default_problem()
    nominal = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(tf_superconductor_id=0),
    ).magnets
    hotter_tech = problem.technology._replace(rebco_operating_temperature_k=45.0)
    strained_tech = problem.technology._replace(rebco_operating_strain=0.005)
    hotter = evaluate_reactor(
        problem.initial_design,
        hotter_tech,
        problem.config._replace(tf_superconductor_id=0),
    ).magnets
    strained = evaluate_reactor(
        problem.initial_design,
        strained_tech,
        problem.config._replace(tf_superconductor_id=0),
    ).magnets

    assert hotter.tf_allowable_engineering_current_density_a_m2 < (
        nominal.tf_allowable_engineering_current_density_a_m2
    )
    assert strained.tf_allowable_engineering_current_density_a_m2 < (
        nominal.tf_allowable_engineering_current_density_a_m2
    )


def test_pf_coil_placement_and_limits_are_reported():
    problem = default_problem()
    base = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
    )
    higher_current = evaluate_reactor(
        problem.initial_design._replace(plasma_current_ma=18.0),
        problem.technology,
        problem.config,
    )

    assert base.magnets.pf_coil_radii_m.shape == (6,)
    assert base.magnets.pf_coil_z_m.shape == (6,)
    assert jnp.all(
        base.magnets.pf_coil_radii_m[:2] > base.geometry.inboard_tf_outer_radius_m
    )
    assert jnp.all(base.magnets.pf_coil_radii_m[:2] < base.geometry.inboard_lcfs_radius_m)
    assert jnp.all(base.magnets.pf_coil_radii_m[2:] > base.geometry.outboard_tf_outer_radius_m)
    assert jnp.all(jnp.isfinite(base.magnets.pf_coil_current_density_a_m2))
    assert jnp.all(jnp.isfinite(base.magnets.pf_coil_peak_field_t))
    assert jnp.all(jnp.isfinite(base.magnets.pf_coil_stress_pa))
    assert jnp.isclose(
        base.magnets.pf_max_current_density_a_m2,
        jnp.max(base.magnets.pf_coil_current_density_a_m2),
    )
    assert jnp.isclose(
        base.magnets.pf_max_peak_field_t,
        jnp.max(base.magnets.pf_coil_peak_field_t),
    )
    assert jnp.isclose(
        base.magnets.pf_max_stress_pa,
        jnp.max(base.magnets.pf_coil_stress_pa),
    )
    assert higher_current.magnets.pf_max_current_density_a_m2 > (
        base.magnets.pf_max_current_density_a_m2
    )


def test_pulsed_reactor_requires_more_cs_flux_than_steady_state():
    problem = default_problem()
    steady_result = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(steady_state=True),
    )
    pulsed_result = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(steady_state=False),
    )
    steady = steady_result.magnets
    pulsed = pulsed_result.magnets

    assert steady.cs_flat_top_flux_required_wb == 0.0
    assert pulsed.cs_flat_top_flux_required_wb > 0.0
    assert pulsed.cs_flux_required_wb > steady.cs_flux_required_wb
    assert pulsed.cs_required_ampere_turns_a > steady.cs_required_ampere_turns_a
    assert steady_result.power.pulse_duty_factor == 1.0
    assert pulsed_result.power.pulse_duty_factor < 1.0
    assert steady_result.power.pf_cs_electric_supply_power_mw == 0.0
    assert pulsed_result.power.pf_cs_electric_supply_power_mw > 0.0
    assert pulsed_result.power.time_averaged_gross_electric_power_mw < (
        pulsed_result.power.gross_electric_power_mw
    )
    assert pulsed_result.economics.capacity_factor < steady_result.economics.capacity_factor


def test_blanket_concepts_change_tbr_and_attenuation_surrogates():
    problem = default_problem()
    lipb = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(blanket_concept_id=0),
    ).nuclear
    flibe = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(blanket_concept_id=1),
    ).nuclear
    hcpb = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(blanket_concept_id=2),
    ).nuclear

    assert lipb.blanket_concept_id == 0.0
    assert flibe.blanket_concept_id == 1.0
    assert hcpb.blanket_concept_id == 2.0
    assert not jnp.isclose(flibe.tbr, lipb.tbr)
    assert not jnp.isclose(hcpb.tbr, lipb.tbr)
    assert not jnp.isclose(flibe.neutron_attenuation, lipb.neutron_attenuation)
    assert not jnp.isclose(hcpb.neutron_attenuation, lipb.neutron_attenuation)
    assert not jnp.isclose(flibe.blanket_pumping_power_mw, lipb.blanket_pumping_power_mw)
    assert not jnp.isclose(hcpb.blanket_pumping_power_mw, lipb.blanket_pumping_power_mw)
    assert not jnp.isclose(flibe.tritium_inventory_kg, lipb.tritium_inventory_kg)
    assert not jnp.isclose(hcpb.tritium_inventory_kg, lipb.tritium_inventory_kg)


def test_nuclear_proxies_respond_to_blanket_and_tbr():
    problem = default_problem()
    base = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
    ).nuclear
    thicker_blanket_design = problem.initial_design._replace(
        inboard_blanket_thickness_m=1.20,
        outboard_blanket_thickness_m=1.20,
    )
    thicker_blanket = evaluate_reactor(
        thicker_blanket_design,
        problem.technology,
        problem.config,
    ).nuclear
    lower_tbr_design = problem.initial_design._replace(
        inboard_blanket_thickness_m=0.45,
        outboard_blanket_thickness_m=0.45,
    )
    lower_tbr = evaluate_reactor(
        lower_tbr_design,
        problem.technology,
        problem.config,
    ).nuclear

    assert thicker_blanket.tbr > base.tbr
    assert thicker_blanket.tf_neutron_wall_loading_mw_m2 < (
        base.tf_neutron_wall_loading_mw_m2
    )
    assert thicker_blanket.tf_dpa_proxy < base.tf_dpa_proxy
    assert lower_tbr.tritium_surplus_kg_per_year < base.tritium_surplus_kg_per_year
    assert lower_tbr.tritium_doubling_time_years > base.tritium_doubling_time_years


def test_nuclear_volume_uses_inboard_outboard_surface_areas():
    problem = default_problem()
    base_design = problem.initial_design._replace(
        inboard_blanket_thickness_m=0.80,
        outboard_blanket_thickness_m=0.80,
    )
    inboard_extra = evaluate_reactor(
        base_design._replace(inboard_blanket_thickness_m=0.90),
        problem.technology,
        problem.config,
    ).nuclear
    outboard_extra = evaluate_reactor(
        base_design._replace(outboard_blanket_thickness_m=0.90),
        problem.technology,
        problem.config,
    ).nuclear
    base = evaluate_reactor(base_design, problem.technology, problem.config)

    inboard_delta = inboard_extra.blanket_volume_m3 - base.nuclear.blanket_volume_m3
    outboard_delta = outboard_extra.blanket_volume_m3 - base.nuclear.blanket_volume_m3

    assert base.geometry.outboard_first_wall_area_m2 > (
        base.geometry.inboard_first_wall_area_m2
    )
    assert outboard_delta > inboard_delta


def test_cs_stress_uses_cs_structural_fraction():
    problem = default_problem()
    low_structure = evaluate_reactor(
        problem.initial_design,
        problem.technology._replace(cs_structural_fraction=0.40),
        problem.config,
    ).magnets
    high_structure = evaluate_reactor(
        problem.initial_design,
        problem.technology._replace(cs_structural_fraction=0.70),
        problem.config,
    ).magnets

    assert high_structure.cs_stress_pa < low_structure.cs_stress_pa


def test_divertor_heat_load_scaling_responds_to_plasma_current():
    problem = default_problem()
    base = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(divertor_heat_load_model_id=1),
    ).exhaust
    high_current_design = problem.initial_design._replace(plasma_current_ma=20.0)
    high_current = evaluate_reactor(
        high_current_design,
        problem.technology,
        problem.config._replace(divertor_heat_load_model_id=1),
    ).exhaust
    fixed_lambda = evaluate_reactor(
        high_current_design,
        problem.technology,
        problem.config._replace(divertor_heat_load_model_id=0),
    ).exhaust

    assert high_current.divertor_poloidal_field_t > base.divertor_poloidal_field_t
    assert high_current.divertor_upstream_lambda_q_m < (
        base.divertor_upstream_lambda_q_m
    )
    assert high_current.peak_divertor_heat_flux_mw_m2 > (
        base.peak_divertor_heat_flux_mw_m2
    )
    assert high_current.divertor_target_temperature_c > (
        base.divertor_target_temperature_c
    )
    assert high_current.divertor_lifetime_fpy < base.divertor_lifetime_fpy
    assert jnp.isclose(
        fixed_lambda.divertor_upstream_lambda_q_m,
        problem.technology.divertor_lambda_integral_m,
    )


def test_constraint_analysis_reports_active_constraints_and_sensitivities():
    problem = default_problem()
    report = analyze_constraints(problem, active_tolerance=0.05)
    text = format_constraint_report(report, max_rows=5)

    assert report["rows"]
    assert report["num_active"] >= 1
    assert "q95" in [row["name"] for row in report["active"]]
    assert report["ranked_by_sensitivity"][0]["sensitivity_norm"] >= 0.0
    assert "active=" in text
