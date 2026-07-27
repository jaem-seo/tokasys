"""Human-readable optimization report helpers."""

from __future__ import annotations

from typing import Any

from tokasys.core.types import (
    DesignVariables,
    MarginState,
    OptimizationConfig,
    ReactorConfig,
    ReactorResult,
    ReactorVariables,
    ResidualState,
    TechnologyParameters,
)
from tokasys.models.constraints import active_equality_indices, active_inequality_indices
from tokasys.models.objectives import (
    OBJECTIVE_FUNCTIONS,
    objective_names,
    objective_weights,
)


def _f(value: Any) -> float:
    return float(value)


def objective_reports(
    variables: ReactorVariables,
    result: ReactorResult,
    config: ReactorConfig,
    optimization: OptimizationConfig,
) -> list[dict[str, Any]]:
    design = variables.design
    actual: dict[str, tuple[float, str]] = {
        "major_radius": (_f(design.major_radius_m), "m"),
        "coe": (_f(result.economics.coe_usd_mwh), "USD/MWh"),
        "negative_net_power": (_f(result.power.net_electric_power_mw), "MW"),
    }
    rows = []
    for name, weight in zip(
        objective_names(config, optimization),
        objective_weights(config, optimization),
        strict=True,
    ):
        actual_value, unit = actual[name]
        rows.append(
            {
                "name": name,
                "actual": actual_value,
                "unit": unit,
                "weight": weight,
                "objective_value": _f(
                    weight * OBJECTIVE_FUNCTIONS[name](design, result)
                ),
            }
        )
    return rows


def _equality_actuals(
    variables: ReactorVariables,
    result: ReactorResult,
    tech: TechnologyParameters,
) -> dict[str, tuple[float, str, float, str]]:
    design = variables.design
    closure = variables.closure
    plasma_power_input = (
        result.plasma.alpha_power_mw
        + design.auxiliary_heating_mw
        + design.current_drive_power_mw
        + result.plasma.ohmic_power_mw
    )
    plasma_power_output = (
        result.plasma.radiation_power_mw
        + result.plasma.transport_loss_mw
    )
    return {
        "plasma_power_balance": (
            _f(plasma_power_input),
            "MW input",
            _f(plasma_power_output),
            "MW output",
        ),
        "plasma_current_balance": (
            _f(design.plasma_current_ma),
            "MA Ip",
            _f(result.plasma.bootstrap_current_ma + result.plasma.current_drive_current_ma),
            "MA non-inductive",
        ),
        "target_net_power": (
            _f(result.power.net_electric_power_mw),
            "MW net",
            _f(tech.net_electric_target_mw),
            "MW target",
        ),
        "stored_energy_closure": (
            _f(closure.stored_energy_mj),
            "MJ closure",
            _f(result.plasma.stored_energy_mj),
            "MJ model",
        ),
        "transport_loss_closure": (
            _f(closure.transport_loss_mw),
            "MW closure",
            _f(result.plasma.ipb98_transport_loss_mw),
            "MW IPB98",
        ),
        "bootstrap_current_closure": (
            _f(closure.bootstrap_current_ma),
            "MA closure",
            _f(result.plasma.bootstrap_current_ma),
            "MA model",
        ),
        "current_drive_current_closure": (
            _f(closure.current_drive_current_ma),
            "MA closure",
            _f(result.plasma.current_drive_current_ma),
            "MA model",
        ),
        "net_electric_power_closure": (
            _f(closure.net_electric_power_mw),
            "MW closure",
            _f(result.power.net_electric_power_mw),
            "MW model",
        ),
        "tf_peak_field_closure": (
            _f(closure.tf_peak_field_t),
            "T closure",
            _f(result.magnets.peak_tf_field_t),
            "T model",
        ),
        "tbr_closure": (
            _f(closure.tbr),
            "closure",
            _f(result.nuclear.tbr),
            "model",
        ),
        "divertor_heat_flux_closure": (
            _f(closure.divertor_heat_flux_mw_m2),
            "MW/m2 closure",
            _f(result.exhaust.peak_divertor_heat_flux_mw_m2),
            "MW/m2 model",
        ),
    }


def equality_reports(
    variables: ReactorVariables,
    result: ReactorResult,
    tech: TechnologyParameters,
    config: ReactorConfig,
    optimization: OptimizationConfig,
) -> list[dict[str, Any]]:
    residuals = list(result.residuals)
    actuals = _equality_actuals(variables, result, tech)
    rows = []
    for index in active_equality_indices(config, optimization):
        name = ResidualState._fields[index]
        lhs, lhs_label, rhs, rhs_label = actuals[name]
        residual = _f(residuals[index])
        rows.append(
            {
                "name": name,
                "lhs": lhs,
                "lhs_label": lhs_label,
                "rhs": rhs,
                "rhs_label": rhs_label,
                "residual": residual,
                "satisfied": abs(residual) <= optimization.equality_tolerance,
            }
        )
    return rows


def _inequality_actuals(
    design: DesignVariables,
    result: ReactorResult,
    tech: TechnologyParameters,
) -> dict[str, tuple[float, str, float, str, str]]:
    return {
        "q95": (_f(result.plasma.q95), "", _f(tech.q95_min), "", ">="),
        "beta_n": (_f(result.plasma.beta_n), "", _f(tech.beta_n_limit), "", "<="),
        "greenwald": (
            _f(design.greenwald_fraction),
            "",
            _f(tech.greenwald_fraction_max),
            "",
            "<=",
        ),
        "fusion_gain": (
            _f(result.plasma.fusion_gain),
            "",
            _f(tech.fusion_gain_min),
            "",
            ">=",
        ),
        "target_net_power": (
            _f(result.power.net_electric_power_mw),
            "MW",
            _f(tech.net_electric_target_mw),
            "MW",
            ">=",
        ),
        "inboard_space": (
            _f(result.geometry.inboard_tf_inner_radius_m - design.cs_thickness_m),
            "m clearance",
            0.0,
            "m minimum",
            ">=",
        ),
        "tf_peak_field": (
            _f(result.magnets.peak_tf_field_t),
            "T",
            _f(tech.tf_peak_field_limit_t),
            "T",
            "<=",
        ),
        "tf_current_density": (
            _f(result.magnets.tf_engineering_current_density_a_m2),
            "A/m2",
            _f(result.magnets.tf_allowable_engineering_current_density_a_m2),
            "A/m2 allowable",
            "<=",
        ),
        "tf_stress": (
            _f(result.magnets.tf_stress_pa),
            "Pa",
            _f(tech.tf_allowable_stress_pa),
            "Pa",
            "<=",
        ),
        "tf_nuclear_heat": (
            _f(result.magnets.tf_nuclear_heat_mw),
            "MW",
            _f(tech.tf_nuclear_heat_limit_mw),
            "MW",
            "<=",
        ),
        "pf_equilibrium_ampere_turns": (
            _f(result.magnets.pf_required_equilibrium_ampere_turns_a),
            "A-turn",
            _f(tech.pf_max_equilibrium_ampere_turns_a),
            "A-turn",
            "<=",
        ),
        "pf_current_density": (
            _f(result.magnets.pf_max_current_density_a_m2),
            "A/m2",
            _f(result.magnets.pf_current_density_limit_a_m2),
            "A/m2",
            "<=",
        ),
        "pf_peak_field": (
            _f(result.magnets.pf_max_peak_field_t),
            "T",
            _f(tech.pf_peak_field_limit_t),
            "T",
            "<=",
        ),
        "pf_stress": (
            _f(result.magnets.pf_max_stress_pa),
            "Pa",
            _f(tech.pf_allowable_stress_pa),
            "Pa",
            "<=",
        ),
        "cs_flux_swing": (
            _f(result.magnets.cs_flux_capacity_wb),
            "Wb capacity",
            _f(result.magnets.cs_flux_required_wb),
            "Wb required",
            ">=",
        ),
        "cs_peak_field": (
            _f(result.magnets.cs_peak_field_t),
            "T",
            _f(tech.cs_peak_field_limit_t),
            "T",
            "<=",
        ),
        "cs_current_density": (
            _f(result.magnets.cs_current_density_a_m2),
            "A/m2",
            _f(result.magnets.cs_current_density_limit_a_m2),
            "A/m2",
            "<=",
        ),
        "cs_stress": (
            _f(result.magnets.cs_stress_pa),
            "Pa",
            _f(tech.cs_allowable_stress_pa),
            "Pa",
            "<=",
        ),
        "tbr": (_f(result.nuclear.tbr), "", _f(tech.tbr_min), "", ">="),
        "tf_fluence": (
            _f(result.nuclear.tf_fluence_n_m2),
            "n/m2",
            _f(tech.tf_fluence_limit_n_m2),
            "n/m2",
            "<=",
        ),
        "tf_dpa": (
            _f(result.nuclear.tf_dpa_proxy),
            "dpa",
            _f(tech.tf_dpa_limit),
            "dpa",
            "<=",
        ),
        "tritium_doubling_time": (
            _f(result.nuclear.tritium_doubling_time_years),
            "yr",
            _f(tech.tritium_doubling_time_max_years),
            "yr",
            "<=",
        ),
        "first_wall_heat_load": (
            _f(result.nuclear.first_wall_surface_heat_load_mw_m2),
            "MW/m2",
            _f(tech.first_wall_surface_heat_load_limit_mw_m2),
            "MW/m2",
            "<=",
        ),
        "divertor_heat_flux": (
            _f(result.exhaust.peak_divertor_heat_flux_mw_m2),
            "MW/m2",
            _f(tech.divertor_heat_flux_limit_mw_m2),
            "MW/m2",
            "<=",
        ),
        "divertor_target_temperature": (
            _f(result.exhaust.divertor_target_temperature_c),
            "C",
            _f(tech.divertor_target_temperature_limit_c),
            "C",
            "<=",
        ),
        "divertor_lifetime": (
            _f(result.exhaust.divertor_lifetime_fpy),
            "FPY",
            _f(tech.divertor_lifetime_min_fpy),
            "FPY min",
            ">=",
        ),
        "recirculating_fraction": (
            _f(result.power.recirculating_fraction),
            "",
            _f(tech.recirculating_fraction_max),
            "",
            "<=",
        ),
    }


def inequality_reports(
    variables: ReactorVariables,
    result: ReactorResult,
    tech: TechnologyParameters,
    optimization: OptimizationConfig,
) -> list[dict[str, Any]]:
    margins = list(result.margins)
    actuals = _inequality_actuals(variables.design, result, tech)
    rows = []
    for index in active_inequality_indices(optimization):
        name = MarginState._fields[index]
        actual, actual_unit, limit, limit_unit, sense = actuals[name]
        margin = _f(margins[index])
        rows.append(
            {
                "name": name,
                "actual": actual,
                "actual_unit": actual_unit,
                "sense": sense,
                "limit": limit,
                "limit_unit": limit_unit,
                "margin": margin,
                "satisfied": margin >= -optimization.inequality_tolerance,
            }
        )
    return rows


__all__ = ["equality_reports", "inequality_reports", "objective_reports"]
