"""Scalar design objectives."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from tokasys.core.types import DesignVariables, OptimizationConfig, ReactorConfig, ReactorResult


Objective = Callable[[DesignVariables, ReactorResult], jnp.ndarray]


def major_radius_objective(x: DesignVariables, result: ReactorResult) -> jnp.ndarray:
    del result
    return x.major_radius_m


def coe_objective(x: DesignVariables, result: ReactorResult) -> jnp.ndarray:
    del x
    return result.economics.coe_usd_mwh


def negative_net_power_objective(
    x: DesignVariables,
    result: ReactorResult,
) -> jnp.ndarray:
    del x
    return -result.power.net_electric_power_mw


OBJECTIVE_FUNCTIONS: dict[str, Objective] = {
    "major_radius": major_radius_objective,
    "coe": coe_objective,
    "negative_net_power": negative_net_power_objective,
}


OBJECTIVE_IDS: dict[int, str] = {
    0: "major_radius",
    1: "coe",
    2: "negative_net_power",
}

DEFAULT_OBJECTIVE_WEIGHTS: dict[str, float] = {
    "major_radius": 0.1,
    "coe": 0.01,
    "negative_net_power": 0.001,
}


def objective_names(
    config: ReactorConfig,
    optimization: OptimizationConfig,
) -> tuple[str, ...]:
    if optimization.objective_terms:
        names = optimization.objective_terms
    else:
        names = (OBJECTIVE_IDS[config.objective_id],)
    unknown = tuple(name for name in names if name not in OBJECTIVE_FUNCTIONS)
    if unknown:
        raise ValueError(f"Unknown objective term(s): {unknown}")
    return names


def objective_weights(
    config: ReactorConfig,
    optimization: OptimizationConfig,
) -> tuple[float, ...]:
    names = objective_names(config, optimization)
    if not optimization.objective_weights:
        return tuple(DEFAULT_OBJECTIVE_WEIGHTS[name] for name in names)
    if len(optimization.objective_weights) != len(names):
        raise ValueError(
            "objective_weights must have the same length as the active "
            f"objective terms; got {len(optimization.objective_weights)} "
            f"weight(s) for {len(names)} term(s)."
        )
    return optimization.objective_weights


def objective_value(
    x: DesignVariables,
    result: ReactorResult,
    config: ReactorConfig,
    optimization: OptimizationConfig,
) -> jnp.ndarray:
    values = [
        weight * OBJECTIVE_FUNCTIONS[name](x, result)
        for name, weight in zip(
            objective_names(config, optimization),
            objective_weights(config, optimization),
            strict=True,
        )
    ]
    return jnp.sum(jnp.stack(values))


def objective_term_values(
    x: DesignVariables,
    result: ReactorResult,
    config: ReactorConfig,
    optimization: OptimizationConfig,
) -> dict[str, float]:
    return {
        name: float(weight * OBJECTIVE_FUNCTIONS[name](x, result))
        for name, weight in zip(
            objective_names(config, optimization),
            objective_weights(config, optimization),
            strict=True,
        )
    }
