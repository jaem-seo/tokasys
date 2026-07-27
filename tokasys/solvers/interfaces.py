"""Normalized-vector objective and constraint factories."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tokasys.core.types import ReactorProblem
from tokasys.core.variables import denormalize_reactor_variables
from tokasys.models.constraints import equality_vector, inequality_vector
from tokasys.models.objectives import objective_value
from tokasys.models.reactor import evaluate_reactor


def make_vector_functions(problem: ReactorProblem):
    def result_fn(u: jnp.ndarray):
        variables = denormalize_reactor_variables(u, problem.variable_bounds)
        return evaluate_reactor(
            variables,
            problem.technology,
            problem.config,
            problem.numerics,
        )

    def objective_fn(u: jnp.ndarray) -> jnp.ndarray:
        variables = denormalize_reactor_variables(u, problem.variable_bounds)
        result = evaluate_reactor(
            variables,
            problem.technology,
            problem.config,
            problem.numerics,
        )
        return objective_value(
            variables.design,
            result,
            problem.config,
            problem.optimization,
        )

    def equality_fn(u: jnp.ndarray) -> jnp.ndarray:
        return equality_vector(result_fn(u), problem.config, problem.optimization)

    def inequality_fn(u: jnp.ndarray) -> jnp.ndarray:
        return inequality_vector(result_fn(u), problem.optimization)

    return {
        "result": jax.jit(result_fn),
        "objective": jax.jit(objective_fn),
        "objective_value_and_grad": jax.jit(jax.value_and_grad(objective_fn)),
        "equalities": jax.jit(equality_fn),
        "equality_jacobian": jax.jit(jax.jacrev(equality_fn)),
        "inequalities": jax.jit(inequality_fn),
        "inequality_jacobian": jax.jit(jax.jacrev(inequality_fn)),
    }
