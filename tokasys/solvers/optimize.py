"""Constrained optimization wrapper using JAX exact derivatives with SciPy.

Notes:
    The driver supplies analytic JAX gradients/Jacobians to SLSQP. The current
    public optimizer is intentionally SLSQP-only to keep constraint semantics
    clear while the system model is still evolving.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from tokasys.core.types import DesignVariables, ReactorProblem, ReactorResult, ReactorVariables
from tokasys.core.variables import denormalize_reactor_variables, normalize_reactor_variables
from tokasys.models.constraints import active_equality_indices, active_inequality_indices
from tokasys.solvers.interfaces import make_vector_functions


@dataclass(frozen=True)
class DesignSolution:
    scipy_result: OptimizeResult
    variables: ReactorVariables
    design: DesignVariables
    reactor: ReactorResult


def _to_numpy(x):
    """Transfer a JAX value to a floating-point NumPy array for SciPy."""
    return np.asarray(jax.device_get(x), dtype=float)


def solve_slsqp(
    problem: ReactorProblem,
    *,
    maxiter: int = 300,
    ftol: float = 1.0e-8,
    disp: bool = True,
) -> DesignSolution:
    """Solve the normalized full-space design problem.

    The current version uses full-space design plus closure variables. Equality
    residuals include plant balances and closure consistency checks; inequality
    margins. SLSQP is used only as the nonlinear-program driver; all objective
    and constraint derivatives are supplied by JAX.
    """
    funcs = make_vector_functions(problem)
    u0 = _to_numpy(
        normalize_reactor_variables(problem.initial_variables, problem.variable_bounds)
    )

    def fun(u):
        """Return the scalar objective and exact gradient expected by SciPy."""
        value, grad = funcs["objective_value_and_grad"](u)
        return float(value), _to_numpy(grad)

    constraints = []
    if active_equality_indices(problem.config, problem.optimization):
        constraints.append(
            {
                "type": "eq",
                "fun": lambda u: _to_numpy(funcs["equalities"](u)),
                "jac": lambda u: _to_numpy(funcs["equality_jacobian"](u)),
            }
        )
    if active_inequality_indices(problem.optimization):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda u: _to_numpy(funcs["inequalities"](u)),
                "jac": lambda u: _to_numpy(funcs["inequality_jacobian"](u)),
            }
        )

    result = minimize(
        fun,
        u0,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * len(u0),
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": ftol, "disp": disp},
    )

    variables = denormalize_reactor_variables(result.x, problem.variable_bounds)
    design = variables.design
    reactor = funcs["result"](result.x)
    return DesignSolution(result, variables, design, reactor)
