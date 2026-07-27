"""Vector interfaces for constrained optimizers.

Notes:
    Equality residuals and positive-is-feasible margins are normalized for
    gradient-based constrained optimization.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import (
    MarginState,
    OptimizationConfig,
    ReactorConfig,
    ReactorResult,
    ResidualState,
)


_RESIDUAL_COUNT = len(ResidualState._fields)
_MARGIN_COUNT = len(MarginState._fields)


def _named_indices(names: tuple[str, ...], selected: tuple[str, ...]) -> tuple[int, ...]:
    unknown = tuple(name for name in selected if name not in names)
    if unknown:
        raise ValueError(f"Unknown constraint name(s): {unknown}")
    return tuple(names.index(name) for name in selected)


def _apply_selection(
    names: tuple[str, ...],
    candidate_indices: tuple[int, ...],
    enabled: tuple[str, ...],
    disabled: tuple[str, ...],
) -> tuple[int, ...]:
    candidates = set(candidate_indices)
    selected = (
        tuple(i for i in _named_indices(names, enabled) if i in candidates)
        if enabled
        else candidate_indices
    )
    disabled_indices = set(_named_indices(names, disabled))
    return tuple(i for i in selected if i not in disabled_indices)


def active_equality_indices(
    config: ReactorConfig | None,
    optimization: OptimizationConfig | None = None,
) -> tuple[int, ...]:
    if config is None:
        candidates = tuple(range(_RESIDUAL_COUNT))
        if optimization is None:
            return candidates
        return _apply_selection(
            ResidualState._fields,
            candidates,
            optimization.enabled_equalities,
            optimization.disabled_equalities,
        )
    mode = config.energy_closure_mode.lower()
    if mode == "fixed_point":
        mode = "self_consistent"
    if mode == "self_consistent":
        # Plasma power balance is enforced internally by the fixed-point energy
        # closure, so the corresponding residual is an identity with zero Jacobian.
        candidates = tuple(i for i in range(_RESIDUAL_COUNT) if i != 0)
    if mode == "full_space":
        # The full-space evaluator uses closure.stored_energy_mj directly to build
        # profiles, making the stored-energy closure residual an identity.
        candidates = tuple(i for i in range(_RESIDUAL_COUNT) if i != 3)
    if mode not in {"self_consistent", "full_space"}:
        candidates = tuple(range(_RESIDUAL_COUNT))
    if optimization is None:
        return candidates
    return _apply_selection(
        ResidualState._fields,
        candidates,
        optimization.enabled_equalities,
        optimization.disabled_equalities,
    )


def active_inequality_indices(
    optimization: OptimizationConfig | None = None,
) -> tuple[int, ...]:
    candidates = tuple(range(_MARGIN_COUNT))
    if optimization is None:
        return candidates
    return _apply_selection(
        MarginState._fields,
        candidates,
        optimization.enabled_inequalities,
        optimization.disabled_inequalities,
    )


def equality_vector(
    result: ReactorResult,
    config: ReactorConfig | None = None,
    optimization: OptimizationConfig | None = None,
) -> jnp.ndarray:
    residuals = jnp.stack(list(result.residuals))
    if config is None and optimization is None:
        return residuals
    return residuals[jnp.asarray(active_equality_indices(config, optimization))]


def inequality_vector(
    result: ReactorResult,
    optimization: OptimizationConfig | None = None,
) -> jnp.ndarray:
    """Return positive-is-feasible normalized engineering margins."""
    margins = jnp.stack(list(result.margins))
    if optimization is None:
        return margins
    return margins[jnp.asarray(active_inequality_indices(optimization))]
