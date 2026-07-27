#!/usr/bin/env python3
"""Generate Figure 4: AD-assisted constrained reactor-design optimization.

The script starts from ``tokasys.default_problem()`` and solves a normalized
full-space problem with SciPy SLSQP. JAX supplies the exact objective gradient
and constraint Jacobians.

The combined demonstration objective is

    F = COE / 100 - P_net / 1000,

so minimization favors both lower cost of electricity and higher net electric
power. Figure 4 contains

(a) the optimization path in the R0-Bt plane,
(b) net electric power and COE versus SLSQP iteration,
(c) the plasma-power-balance equality residual, and
(d) the q95 inequality margin.

Run from an installed TokaSys environment, for example:

    cd system_code
    pip install -e .
    python /path/to/figure4_slsqp_trajectory.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize

jax.config.update("jax_enable_x64", True)

from tokasys import default_problem  # noqa: E402
from tokasys.core.types import ReactorProblem  # noqa: E402
from tokasys.core.variables import (  # noqa: E402
    denormalize_reactor_variables,
    normalize_reactor_variables,
)
from tokasys.models.constraints import (  # noqa: E402
    equality_vector,
    inequality_vector,
)
from tokasys.models.objectives import objective_value  # noqa: E402
from tokasys.solvers.interfaces import make_vector_functions  # noqa: E402


@dataclass(frozen=True)
class HistoryRow:
    iteration: int
    major_radius_m: float
    toroidal_field_t: float
    plasma_current_ma: float
    net_electric_power_mw: float
    coe_usd_mwh: float
    objective: float
    plasma_power_balance_residual: float
    q95: float
    q95_margin: float
    beta_n: float
    beta_n_margin: float
    max_abs_equality_residual: float
    min_inequality_margin: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the SLSQP trajectory for the differentiable reactor model."
    )
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--ftol", type=float, default=1.0e-9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure4_slsqp_trajectory"),
        help="Output path without extension.",
    )
    parser.add_argument(
        "--disp",
        action="store_true",
        help="Show the SciPy SLSQP iteration summary.",
    )
    return parser.parse_args()


def build_problem() -> ReactorProblem:
    """Return the ITER-scale full-space problem used for Figure 4."""
    problem = default_problem()

    # The power-balance residual is an explicit equality only in full-space mode.
    config = problem.config._replace(energy_closure_mode="full_space")

    # Equality constraints retained in the paper's full-space formulation.
    # ``stored_energy_closure`` is an identity in full-space mode, and the target
    # net power is imposed as an inequality rather than as an equality.
    enabled_equalities = (
        "plasma_power_balance",
        "plasma_current_balance",
        "transport_loss_closure",
        "bootstrap_current_closure",
        "current_drive_current_closure",
        "net_electric_power_closure",
        "tf_peak_field_closure",
        "tbr_closure",
        "divertor_heat_flux_closure",
    )

    # Representative plasma, magnet, nuclear, exhaust, and plant limits.
    enabled_inequalities = (
        "q95",
        "beta_n",
        "greenwald",
        "target_net_power",
        "inboard_space",
        "tf_peak_field",
        "tf_current_density",
        "tf_stress",
        "tf_nuclear_heat",
        "pf_equilibrium_ampere_turns",
        "cs_flux_swing",
        "cs_peak_field",
        "cs_current_density",
        "cs_stress",
        "tbr",
        "divertor_heat_flux",
        "recirculating_fraction",
    )

    optimization = problem.optimization._replace(
        objective_terms=("coe", "negative_net_power"),
        enabled_equalities=enabled_equalities,
        enabled_inequalities=enabled_inequalities,
    )
    return problem._replace(config=config, optimization=optimization)


def _to_numpy(value) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=float)


def run_optimization(
    problem: ReactorProblem,
    *,
    maxiter: int,
    ftol: float,
    disp: bool,
) -> tuple[OptimizeResult, list[HistoryRow]]:
    """Run SLSQP and record every accepted iterate, including the initial point."""
    funcs = make_vector_functions(problem)
    u0 = _to_numpy(
        normalize_reactor_variables(problem.initial_variables, problem.variable_bounds)
    )

    def fun(u: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = funcs["objective_value_and_grad"](u)
        return float(value), _to_numpy(gradient)

    constraints = [
        {
            "type": "eq",
            "fun": lambda u: _to_numpy(funcs["equalities"](u)),
            "jac": lambda u: _to_numpy(funcs["equality_jacobian"](u)),
        },
        {
            "type": "ineq",
            "fun": lambda u: _to_numpy(funcs["inequalities"](u)),
            "jac": lambda u: _to_numpy(funcs["inequality_jacobian"](u)),
        },
    ]

    history: list[HistoryRow] = []
    recorded_u: list[np.ndarray] = []

    def record(u: np.ndarray) -> None:
        u = np.asarray(u, dtype=float)
        if recorded_u and np.linalg.norm(u - recorded_u[-1]) < 1.0e-12:
            return

        variables = denormalize_reactor_variables(u, problem.variable_bounds)
        reactor = funcs["result"](u)
        equalities = _to_numpy(
            equality_vector(reactor, problem.config, problem.optimization)
        )
        inequalities = _to_numpy(
            inequality_vector(reactor, problem.optimization)
        )
        value = objective_value(
            variables.design,
            reactor,
            problem.config,
            problem.optimization,
        )

        history.append(
            HistoryRow(
                iteration=len(history),
                major_radius_m=float(variables.design.major_radius_m),
                toroidal_field_t=float(variables.design.toroidal_field_t),
                plasma_current_ma=float(variables.design.plasma_current_ma),
                net_electric_power_mw=float(reactor.power.net_electric_power_mw),
                coe_usd_mwh=float(reactor.economics.coe_usd_mwh),
                objective=float(value),
                plasma_power_balance_residual=float(
                    reactor.residuals.plasma_power_balance
                ),
                q95=float(reactor.plasma.q95),
                q95_margin=float(reactor.margins.q95),
                beta_n=float(reactor.plasma.beta_n),
                beta_n_margin=float(reactor.margins.beta_n),
                max_abs_equality_residual=float(np.max(np.abs(equalities))),
                min_inequality_margin=float(np.min(inequalities)),
            )
        )
        recorded_u.append(u.copy())

    record(u0)
    result = minimize(
        fun,
        u0,
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * len(u0),
        constraints=constraints,
        callback=record,
        options={"maxiter": maxiter, "ftol": ftol, "disp": disp},
    )
    record(result.x)
    return result, history


def save_history(history: list[HistoryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(HistoryRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row.__dict__)


def plot_history(history: list[HistoryRow], output: Path) -> None:
    iteration = np.asarray([row.iteration for row in history])
    radius = np.asarray([row.major_radius_m for row in history])
    field = np.asarray([row.toroidal_field_t for row in history])
    current = np.asarray([row.plasma_current_ma for row in history])
    net_power = np.asarray([row.net_electric_power_mw for row in history])
    coe = np.asarray([row.coe_usd_mwh for row in history])
    power_residual = np.asarray(
        [row.plasma_power_balance_residual for row in history]
    )
    q95_margin = np.asarray([row.q95_margin for row in history])
    beta_n = np.asarray([row.beta_n for row in history])
    beta_n_margin = np.asarray([row.beta_n_margin for row in history])

    '''fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    ax_path, ax_objective = axes[0]
    ax_equality, ax_inequality = axes[1]'''

    fig = plt.figure(figsize=(5, 8), constrained_layout=True)
    gs = fig.add_gridspec(
        4, 1,
        #width_ratios = [1.5, 1.0],
        height_ratios = [2.0, 1.0, 1.0, 1.0]
    )
    ax_path = fig.add_subplot(gs[0])
    ax_objective = fig.add_subplot(gs[1])
    ax_equality = fig.add_subplot(gs[2], sharex=ax_objective)
    ax_inequality = fig.add_subplot(gs[3], sharex=ax_objective)

    # (a) Path in a representative two-dimensional projection of design space.
    ax_path.plot(radius, current, "o-", color='k', linewidth=1.5, markersize=4)
    ax_path.plot(radius[0], current[0], "s", markersize=8, label="Initial")
    ax_path.plot(radius[-1], current[-1], "*", markersize=12, label="Optimized")
    ax_path.annotate(
        "start",
        (radius[0], current[0]),
        xytext=(6, -14),
        textcoords="offset points",
    )
    ax_path.annotate(
        "end",
        (radius[-1], current[-1]),
        xytext=(6, 8),
        textcoords="offset points",
    )
    ax_path.set_xlabel(r"Major radius $R_0$ [m]")
    ax_path.set_ylabel(r"Plasma current $I_p$ [MA]")
    ax_path.set_xlim([3.7, 7.8])
    ax_path.set_ylim([5.0, 24.0])
    #ax_path.set_title("(a) Design-space trajectory")
    ax_path.grid(alpha=0.3)
    ax_path.legend(frameon=False)

    # (b) Physical objective metrics. Two axes retain their original units.
    line_power = ax_objective.plot(
        iteration,
        net_power,
        "o-",
        linewidth=1.5,
        markersize=4,
        label=r"$P_{\mathrm{net}}$",
    )[0]
    #ax_objective.set_xlabel("SLSQP iteration")
    ax_objective.set_ylabel(r"$P_{\mathrm{net}}$ [MW]")
    ax_objective.grid(alpha=0.3)

    ax_coe = ax_objective.twinx()
    line_coe = ax_coe.plot(
        iteration[net_power>0],
        coe[net_power>0],
        "s--",
        color="tab:orange",
        linewidth=1.5,
        markersize=4,
        label="COE",
    )[0]
    ax_coe.spines['right'].set_color('tab:orange')
    ax_coe.tick_params(axis='y', colors='tab:orange')
    #ax_coe.set_yscale("log")
    ax_coe.set_ylabel("COE [USD/MWh]")
    #ax_objective.set_title("(b) Reactor-performance objectives")
    ax_objective.legend(
        [line_power, line_coe],
        [line_power.get_label(), line_coe.get_label()],
        frameon=False,
        loc="best",
    )

    # (c) Signed equality residual. Symlog shows both early infeasibility and
    # convergence to machine-level consistency without discarding its sign.
    ax_equality.plot(
        iteration,
        power_residual,
        "o-",
        linewidth=1.5,
        markersize=4,
    )
    ax_equality.axhline(0.0, linewidth=1.0, linestyle="--", color='k')
    ax_equality.set_yscale("symlog", linthresh=1.0e-4)
    #ax_equality.set_xlabel("SLSQP iteration")
    ax_equality.set_ylabel(r"Power-balance residual")
    #ax_equality.set_title("(c) Equality-constraint convergence")
    ax_equality.grid(alpha=0.3)

    # (d) Positive margin means q95 is feasible.
    ax_inequality.plot(
        iteration,
        beta_n,
        "o-",
        linewidth=1.5,
        markersize=4,
    )
    ax_inequality.axhline(3.0, linewidth=1.0, linestyle="--", color='k', label=r'$\beta_{N}$ limit')
    ax_inequality.fill_between(
        iteration,
        3.0,
        beta_n,
        where=beta_n_margin >= 0.0,
        alpha=0.12,
    )
    ax_inequality.set_xlabel("Optimization iteration")
    ax_inequality.set_ylabel(
        #r"$(q_{95}-q_{95}^{\min})/q_{95}^{\min}$"
        #r"$(\beta_{n}^{\max}-\beta_{n})/\beta_{n}^{\max}$"
        r"$\beta_{N}$"
    )
    #ax_inequality.set_title("(d) Inequality-constraint margin")
    ax_inequality.grid(alpha=0.3)
    ax_inequality.legend(frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    problem = build_problem()
    result, history = run_optimization(
        problem,
        maxiter=args.maxiter,
        ftol=args.ftol,
        disp=args.disp,
    )

    plot_history(history, args.output)
    save_history(history, args.output.with_suffix(".csv"))

    first = history[0]
    last = history[-1]
    print("\n=== Figure 4 optimization ===")
    print(f"success: {result.success}")
    print(f"message: {result.message}")
    print(f"iterations: {result.nit}")
    print(f"objective: {result.fun:.8g}")
    print(
        "R0: "
        f"{first.major_radius_m:.4f} -> {last.major_radius_m:.4f} m"
    )
    print(
        "Bt: "
        f"{first.toroidal_field_t:.4f} -> {last.toroidal_field_t:.4f} T"
    )
    print(
        "Ip: "
        f"{first.plasma_current_ma:.4f} -> {last.plasma_current_ma:.4f} MA"
    )
    print(
        "Pnet: "
        f"{first.net_electric_power_mw:.3f} -> "
        f"{last.net_electric_power_mw:.3f} MW"
    )
    print(f"COE: {first.coe_usd_mwh:.3f} -> {last.coe_usd_mwh:.3f} USD/MWh")
    print(
        "power-balance residual: "
        f"{first.plasma_power_balance_residual:.3e} -> "
        f"{last.plasma_power_balance_residual:.3e}"
    )
    print(f"q95 margin: {first.q95_margin:.3e} -> {last.q95_margin:.3e}")
    print(f"PNG: {args.output.with_suffix('.png')}")
    print(f"PDF: {args.output.with_suffix('.pdf')}")
    print(f"CSV: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
