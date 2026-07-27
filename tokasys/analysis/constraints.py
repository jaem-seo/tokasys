"""Constraint activation and sensitivity reporting.

Notes:
    Sensitivities are normalized-variable Jacobian rows of the active equality
    residuals and inequality margins. They are diagnostic local derivatives,
    not global uncertainty quantification.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tokasys.core.types import DesignVariables, MarginState, ReactorProblem, ResidualState
from tokasys.core.variables import REACTOR_FIELD_NAMES, normalize_reactor_variables
from tokasys.models.constraints import active_equality_indices, active_inequality_indices
from tokasys.solvers.interfaces import make_vector_functions


def _row_report(
    *,
    kind: str,
    name: str,
    value: float,
    jacobian_row: np.ndarray,
    active_tolerance: float,
) -> dict[str, Any]:
    abs_jacobian = np.abs(jacobian_row)
    strongest_index = int(np.argmax(abs_jacobian)) if abs_jacobian.size else 0
    if kind == "equality":
        active = abs(value) <= active_tolerance
        slack = abs(value)
    else:
        active = value <= active_tolerance
        slack = value
    return {
        "kind": kind,
        "name": name,
        "value": value,
        "active": bool(active),
        "violated": bool(kind == "inequality" and value < 0.0),
        "slack": slack,
        "sensitivity_norm": float(np.linalg.norm(jacobian_row)),
        "max_sensitivity_variable": REACTOR_FIELD_NAMES[strongest_index],
        "max_sensitivity": float(jacobian_row[strongest_index]),
    }


def analyze_constraints(
    problem: ReactorProblem,
    design: DesignVariables | None = None,
    *,
    active_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Evaluate constraint activation and normalized design sensitivities.

    Sensitivities are rows of the equality/inequality Jacobians with respect to
    normalized design variables. Positive inequality values are feasible.
    """
    if design is None:
        variables = problem.initial_variables
    else:
        variables = problem.initial_variables._replace(design=design)
    funcs = make_vector_functions(problem)
    u = normalize_reactor_variables(variables, problem.variable_bounds)

    equalities = np.asarray(funcs["equalities"](u), dtype=float)
    inequalities = np.asarray(funcs["inequalities"](u), dtype=float)
    equality_jacobian = np.asarray(funcs["equality_jacobian"](u), dtype=float)
    inequality_jacobian = np.asarray(funcs["inequality_jacobian"](u), dtype=float)

    rows = []
    equality_names = [
        ResidualState._fields[i]
        for i in active_equality_indices(problem.config, problem.optimization)
    ]
    for i, name in enumerate(equality_names):
        rows.append(
            _row_report(
                kind="equality",
                name=name,
                value=float(equalities[i]),
                jacobian_row=equality_jacobian[i],
                active_tolerance=active_tolerance,
            )
        )
    inequality_names = [
        MarginState._fields[i]
        for i in active_inequality_indices(problem.optimization)
    ]
    for i, name in enumerate(inequality_names):
        rows.append(
            _row_report(
                kind="inequality",
                name=name,
                value=float(inequalities[i]),
                jacobian_row=inequality_jacobian[i],
                active_tolerance=active_tolerance,
            )
        )

    active_rows = [row for row in rows if row["active"]]
    violated_rows = [row for row in rows if row["violated"]]
    ranked_by_slack = sorted(
        rows,
        key=lambda row: abs(row["value"]) if row["kind"] == "equality" else row["value"],
    )
    ranked_by_sensitivity = sorted(
        rows,
        key=lambda row: row["sensitivity_norm"],
        reverse=True,
    )
    return {
        "active_tolerance": active_tolerance,
        "num_active": len(active_rows),
        "num_violated": len(violated_rows),
        "rows": rows,
        "active": active_rows,
        "violated": violated_rows,
        "ranked_by_slack": ranked_by_slack,
        "ranked_by_sensitivity": ranked_by_sensitivity,
    }


def format_constraint_report(
    report: dict[str, Any],
    *,
    max_rows: int = 12,
) -> str:
    """Format an ``analyze_constraints`` result as a compact text table."""
    lines = [
        (
            f"active={report['num_active']} violated={report['num_violated']} "
            f"tol={report['active_tolerance']:.3g}"
        ),
        "kind        name                         value      sens   strongest",
    ]
    for row in report["ranked_by_slack"][:max_rows]:
        marker = "!" if row["violated"] else "*" if row["active"] else " "
        lines.append(
            f"{marker} {row['kind']:<10} {row['name']:<26} "
            f"{row['value']:>9.3g} {row['sensitivity_norm']:>9.3g} "
            f"{row['max_sensitivity_variable']}"
        )
    return "\n".join(lines)
