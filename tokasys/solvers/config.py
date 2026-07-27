"""Configuration-file helpers for optimization runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from tokasys.core.types import (
    ClosureVariables,
    DesignVariables,
    MarginState,
    OptimizationConfig,
    ReactorProblem,
    ReactorVariableBounds,
    ReactorVariables,
    ResidualState,
    VariableBounds,
)
from tokasys.models.objectives import DEFAULT_OBJECTIVE_WEIGHTS, OBJECTIVE_FUNCTIONS


def _strip_json_comments(text: str) -> str:
    output: list[str] = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            output.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            i += 1
            continue
        if char == "/" and next_char == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if char == "/" and next_char == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        output.append(char)
        i += 1
    return "".join(output)


def _enabled_names(
    section: dict[str, Any],
    valid_names: tuple[str, ...],
    *,
    section_name: str,
) -> tuple[str, ...]:
    unknown = tuple(name for name in section if name not in valid_names)
    if unknown:
        raise ValueError(f"Unknown {section_name} name(s): {unknown}")
    return tuple(name for name in valid_names if _entry_enabled(section.get(name, False)))


def _entry_enabled(entry: Any) -> bool:
    if isinstance(entry, dict):
        return bool(entry.get("enabled", False))
    return bool(entry)


def _objective_weights(
    section: dict[str, Any],
    objective_terms: tuple[str, ...],
) -> tuple[float, ...]:
    weights = []
    for name in objective_terms:
        entry = section.get(name, False)
        value = (
            entry.get("weight", DEFAULT_OBJECTIVE_WEIGHTS[name])
            if isinstance(entry, dict)
            else DEFAULT_OBJECTIVE_WEIGHTS[name]
        )
        weight = float(value)
        if not math.isfinite(weight):
            raise ValueError(f"Objective weight for {name!r} must be finite.")
        weights.append(weight)
    return tuple(weights)


def _constraint_value_overrides(data: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for section_name in ("targets", "limits"):
        for name, value in data.get(section_name, {}).items():
            values[name] = float(value)
    for section_name in ("equalities", "inequalities"):
        for name, entry in data.get(section_name, {}).items():
            if isinstance(entry, dict):
                if "target" in entry:
                    values[name] = float(entry["target"])
                if "limit" in entry:
                    values[name] = float(entry["limit"])
    return values


_CONSTRAINT_VALUE_TO_TECH_FIELD = {
    "target_net_power": "net_electric_target_mw",
    "q95": "q95_min",
    "beta_n": "beta_n_limit",
    "greenwald": "greenwald_fraction_max",
    "fusion_gain": "fusion_gain_min",
    "tf_peak_field": "tf_peak_field_limit_t",
    "tf_current_density": "tf_engineering_current_density_limit_a_m2",
    "tf_stress": "tf_allowable_stress_pa",
    "tf_nuclear_heat": "tf_nuclear_heat_limit_mw",
    "pf_equilibrium_ampere_turns": "pf_max_equilibrium_ampere_turns_a",
    "pf_peak_field": "pf_peak_field_limit_t",
    "pf_stress": "pf_allowable_stress_pa",
    "cs_peak_field": "cs_peak_field_limit_t",
    "cs_stress": "cs_allowable_stress_pa",
    "tbr": "tbr_min",
    "tf_fluence": "tf_fluence_limit_n_m2",
    "tf_dpa": "tf_dpa_limit",
    "tritium_doubling_time": "tritium_doubling_time_max_years",
    "first_wall_heat_load": "first_wall_surface_heat_load_limit_mw_m2",
    "divertor_heat_flux": "divertor_heat_flux_limit_mw_m2",
    "divertor_target_temperature": "divertor_target_temperature_limit_c",
    "divertor_lifetime": "divertor_lifetime_min_fpy",
    "recirculating_fraction": "recirculating_fraction_max",
}


def _technology_with_constraint_values(obj: Any, values: dict[str, float]) -> Any:
    replacements: dict[str, float] = {}
    unknown = []
    for name, value in values.items():
        field = _CONSTRAINT_VALUE_TO_TECH_FIELD.get(name)
        if field is None:
            unknown.append(name)
        else:
            replacements[field] = value
    if unknown:
        raise ValueError(
            "No configurable target/limit is defined for constraint(s): "
            f"{tuple(unknown)}"
        )
    return obj._replace(**replacements)


def _expand_legacy_build_fields(values: dict[str, Any]) -> dict[str, Any]:
    expanded = dict(values)
    legacy_pairs = {
        "first_wall_thickness_m": (
            "inboard_first_wall_thickness_m",
            "outboard_first_wall_thickness_m",
        ),
        "blanket_thickness_m": (
            "inboard_blanket_thickness_m",
            "outboard_blanket_thickness_m",
        ),
        "shield_thickness_m": (
            "inboard_shield_thickness_m",
            "outboard_shield_thickness_m",
        ),
    }
    for legacy_name, split_names in legacy_pairs.items():
        if legacy_name not in expanded:
            continue
        value = expanded.pop(legacy_name)
        for split_name in split_names:
            expanded.setdefault(split_name, value)
    return expanded


def _replace_array_fields(obj: Any, values: dict[str, Any]) -> Any:
    values = _expand_legacy_build_fields(values)
    return obj._replace(
        **{name: jnp.asarray(value, dtype=float) for name, value in values.items()}
    )


def _validate_bounds(lower: Any, upper: Any, *, label: str) -> None:
    invalid = [
        name
        for name in lower._fields
        if float(getattr(upper, name)) <= float(getattr(lower, name))
    ]
    if invalid:
        raise ValueError(f"Upper {label} bound must exceed lower bound for {invalid}.")


def _validate_inside_bounds(values: Any, lower: Any, upper: Any, *, label: str) -> None:
    outside = [
        name
        for name in values._fields
        if not (
            float(getattr(lower, name))
            <= float(getattr(values, name))
            <= float(getattr(upper, name))
        )
    ]
    if outside:
        raise ValueError(
            f"Initial {label} value(s) outside configured bounds: {outside}."
        )


def _bound_values(section: dict[str, Any], kind: str) -> dict[str, Any]:
    values = dict(section.get(kind, {}))
    for name, entry in section.items():
        if name in {"lower", "upper"}:
            continue
        if isinstance(entry, dict) and kind in entry:
            values[name] = entry[kind]
    return _expand_legacy_build_fields(values)


def _configured_bounds(
    problem: ReactorProblem,
    data: dict[str, Any],
) -> tuple[VariableBounds, ReactorVariableBounds]:
    design_lower = problem.variable_bounds.lower.design
    design_upper = problem.variable_bounds.upper.design
    closure_lower = problem.variable_bounds.lower.closure
    closure_upper = problem.variable_bounds.upper.closure

    design_bounds = data.get("bounds", {})
    variable_bounds = data.get("variable_bounds", {})
    closure_bounds = data.get("closure_bounds", {})

    design_lower = _replace_array_fields(
        design_lower,
        {
            **_bound_values(design_bounds, "lower"),
            **_bound_values(variable_bounds.get("design", {}), "lower"),
        },
    )
    design_upper = _replace_array_fields(
        design_upper,
        {
            **_bound_values(design_bounds, "upper"),
            **_bound_values(variable_bounds.get("design", {}), "upper"),
        },
    )
    closure_lower = _replace_array_fields(
        closure_lower,
        {
            **_bound_values(closure_bounds, "lower"),
            **_bound_values(variable_bounds.get("closure", {}), "lower"),
        },
    )
    closure_upper = _replace_array_fields(
        closure_upper,
        {
            **_bound_values(closure_bounds, "upper"),
            **_bound_values(variable_bounds.get("closure", {}), "upper"),
        },
    )

    _validate_bounds(design_lower, design_upper, label="design")
    _validate_bounds(closure_lower, closure_upper, label="closure")

    return (
        VariableBounds(lower=design_lower, upper=design_upper),
        ReactorVariableBounds(
            lower=ReactorVariables(design=design_lower, closure=closure_lower),
            upper=ReactorVariables(design=design_upper, closure=closure_upper),
        ),
    )


def load_optimization_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return json.loads(_strip_json_comments(config_path.read_text(encoding="utf-8")))


def apply_optimization_config(
    problem: ReactorProblem,
    path: str | Path,
) -> ReactorProblem:
    data = load_optimization_config(path)
    objective_terms = _enabled_names(
        data.get("objectives", {}),
        tuple(OBJECTIVE_FUNCTIONS),
        section_name="objective",
    )
    if not objective_terms:
        raise ValueError("At least one objective must be enabled in the config file.")
    objective_weights = _objective_weights(
        data.get("objectives", {}),
        objective_terms,
    )
    enabled_equalities = _enabled_names(
        data.get("equalities", {}),
        ResidualState._fields,
        section_name="equality constraint",
    )
    enabled_inequalities = _enabled_names(
        data.get("inequalities", {}),
        MarginState._fields,
        section_name="inequality constraint",
    )
    tolerances = data.get("tolerances", {})
    optimization = problem.optimization._replace(
        objective_terms=objective_terms,
        objective_weights=objective_weights,
        enabled_equalities=enabled_equalities,
        disabled_equalities=(),
        enabled_inequalities=enabled_inequalities,
        disabled_inequalities=(),
        equality_tolerance=float(
            tolerances.get("equality", problem.optimization.equality_tolerance)
        ),
        inequality_tolerance=float(
            tolerances.get("inequality", problem.optimization.inequality_tolerance)
        ),
    )
    config = problem.config._replace(**data.get("reactor_config", {}))
    technology = problem.technology._replace(**data.get("technology", {}))
    technology = _technology_with_constraint_values(
        technology,
        _constraint_value_overrides(data),
    )
    numerics = problem.numerics._replace(**data.get("numerics", {}))
    design = _replace_array_fields(problem.initial_design, data.get("design", {}))
    closure = _replace_array_fields(
        problem.initial_variables.closure,
        data.get("closure", {}),
    )
    variables = problem.initial_variables._replace(design=design, closure=closure)
    bounds, variable_bounds = _configured_bounds(problem, data)
    _validate_inside_bounds(design, bounds.lower, bounds.upper, label="design")
    _validate_inside_bounds(
        closure,
        variable_bounds.lower.closure,
        variable_bounds.upper.closure,
        label="closure",
    )

    return problem._replace(
        initial_design=design,
        initial_variables=variables,
        bounds=bounds,
        variable_bounds=variable_bounds,
        technology=technology,
        config=config,
        optimization=optimization,
        numerics=numerics,
    )


__all__ = ["apply_optimization_config", "load_optimization_config"]
