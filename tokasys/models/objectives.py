"""Scalar design objectives."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from tokasys.core.types import DesignVariables, OptimizationConfig, ReactorConfig, ReactorResult


Objective = Callable[[DesignVariables, ReactorResult], jnp.ndarray]


def major_radius_objective(x: DesignVariables, result: ReactorResult) -> jnp.ndarray:
    """Return plasma major radius as a quantity to minimize."""
    del result
    return x.major_radius_m


def coe_objective(x: DesignVariables, result: ReactorResult) -> jnp.ndarray:
    """Return levelized cost of electricity as a quantity to minimize."""
    del x
    return result.economics.coe_usd_mwh


def negative_net_power_objective(
    x: DesignVariables,
    result: ReactorResult,
) -> jnp.ndarray:
    """Negate net electric power so minimization maximizes plant output."""
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
    """Resolve active objective names, including the legacy objective ID."""
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
    """Resolve and validate one scalar weight per active objective term."""
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
    """Return the weighted sum of all configured scalar objective terms."""
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
    """Return individual weighted objective contributions for reporting."""
    return {
        name: float(weight * OBJECTIVE_FUNCTIONS[name](x, result))
        for name, weight in zip(
            objective_names(config, optimization),
            objective_weights(config, optimization),
            strict=True,
        )
    }
