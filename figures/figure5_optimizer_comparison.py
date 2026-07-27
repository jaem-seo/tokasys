#!/usr/bin/env python3
"""Generate Figure 5: random, finite-difference, and AD optimization.

The script compares three optimizers on the same reduced-coordinate full-space
TokaSys reactor-design problem and the same five perturbed initial designs:

1. Random parameter search
   A derivative-free stochastic local search. One candidate in the active parameter space is proposed per
   iteration and the best penalized design found so far is retained.
2. Finite-difference SLSQP
   SciPy SLSQP estimates the objective gradient and constraint Jacobians by
   forward finite differences.
3. AD SLSQP
   The same SLSQP driver receives exact JAX derivatives.

The combined objective is

    F = COE / 100 - P_net / 1000,

so lower values correspond to lower COE and larger net electric power.

Figure panels:
(a) combined objective,
(b) RMS equality residual,
(c) maximum inequality violation.

Each curve is the mean over five seeds; the shaded band is +/- one standard
deviation. Runs that terminate before ``--iterations`` are padded with their
last accepted iterate.

Important comparison note
-------------------------
Finite-difference and AD SLSQP often require a similar number of accepted
iterations, but one finite-difference iteration uses many more reactor-model
evaluations. The paper-ready default therefore uses cumulative optimizer calls
on the x-axis. Use ``--x-axis iteration`` to reproduce accepted-iteration plots.

Example
-------
    cd system_code
    pip install -e .
    python /path/to/figure5_optimizer_comparison.py --iterations 50
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, minimize

jax.config.update("jax_enable_x64", True)

from tokasys import default_problem  # noqa: E402
from tokasys.core.types import ClosureVariables, DesignVariables, ReactorProblem  # noqa: E402
from tokasys.core.variables import (  # noqa: E402
    denormalize_reactor_variables,
    normalize_reactor_variables,
)
from tokasys.models.constraints import equality_vector, inequality_vector  # noqa: E402
from tokasys.models.objectives import objective_value  # noqa: E402
from tokasys.models.reactor import evaluate_reactor  # noqa: E402


METHODS = ("Random search", "Finite difference", "Automatic differentiation")

# A reduced but physically representative optimization space keeps the
# finite-difference comparison computationally practical. All eight full-space
# closure variables remain active so the equality constraints can be satisfied.
ACTIVE_DESIGN_NAMES = (
    "major_radius_m",
    "toroidal_field_t",
    "plasma_current_ma",
    "auxiliary_heating_mw",
    "current_drive_power_mw",
)


@dataclass(frozen=True)
class TracePoint:
    method: str
    seed: int
    iteration: int
    optimizer_calls: int
    objective: float
    equality_rms: float
    inequality_violation: float
    net_electric_power_mw: float
    coe_usd_mwh: float


@dataclass(frozen=True)
class PointMetrics:
    objective: float
    equality_rms: float
    inequality_violation: float
    net_electric_power_mw: float
    coe_usd_mwh: float


@dataclass
class CallCounter:
    value: int = 0

    def increment(self) -> None:
        self.value += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare random, finite-difference, and AD reactor optimization."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Number of accepted/search iterations shown in the figure.",
    )
    parser.add_argument(
        "--evaluation-budget",
        type=int,
        default=500,
        help=(
            "Maximum optimizer-call count displayed when "
            "--x-axis evaluations is selected. Random search uses this many "
            "candidate evaluations."
        ),
    )
    parser.add_argument(
        "--x-axis",
        choices=("evaluations", "iteration"),
        default="evaluations",
        help="Use cumulative optimizer function calls (recommended) or accepted iterations.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of perturbed initial conditions.",
    )
    parser.add_argument(
        "--start-std",
        type=float,
        default=0.015,
        help="Std. dev. of the normalized design-variable perturbation.",
    )
    parser.add_argument(
        "--random-sigma-start",
        type=float,
        default=0.12,
        help="Initial normalized random-search proposal width.",
    )
    parser.add_argument(
        "--random-sigma-end",
        type=float,
        default=0.015,
        help="Final normalized random-search proposal width.",
    )
    parser.add_argument(
        "--random-global-prob",
        type=float,
        default=0.05,
        help="Probability of a global uniform random-search proposal.",
    )
    parser.add_argument(
        "--equality-penalty",
        type=float,
        default=100.0,
        help="Random-search penalty multiplying the equality RMS residual.",
    )
    parser.add_argument(
        "--inequality-penalty",
        type=float,
        default=100.0,
        help="Random-search penalty multiplying maximum inequality violation.",
    )
    parser.add_argument("--ftol", type=float, default=1.0e-9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure5_optimizer_comparison"),
        help="Output path without extension.",
    )
    parser.add_argument(
        "--disp",
        action="store_true",
        help="Print SciPy SLSQP termination messages.",
    )
    return parser.parse_args()


def build_problem() -> ReactorProblem:
    """Use the same multi-objective, full-space problem as Figure 4."""
    problem = default_problem()
    config = problem.config._replace(energy_closure_mode="full_space")

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
        # PF/CS constraints are excluded from this reduced-coordinate benchmark
        # because their coil-thickness variables are held fixed. They remain
        # active in the full reactor optimizations used elsewhere in the paper.
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


def to_numpy(value) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=float)


def active_variable_indices() -> np.ndarray:
    design_names = DesignVariables._fields
    design_indices = [design_names.index(name) for name in ACTIVE_DESIGN_NAMES]
    closure_offset = len(design_names)
    closure_indices = list(
        range(closure_offset, closure_offset + len(ClosureVariables._fields))
    )
    return np.asarray(design_indices + closure_indices, dtype=int)


def make_reduced_functions(
    problem: ReactorProblem,
    active_indices: np.ndarray,
) -> dict[str, Callable]:
    """Create JIT functions in the reduced optimization coordinate z.

    ``base_u`` contains all fixed normalized variables for one seed, while z
    replaces the selected design and closure entries. Keeping base_u as an
    explicit function argument avoids recompilation for different seeds.
    """
    index_array = jnp.asarray(active_indices)

    def full_u(z: jnp.ndarray, base_u: jnp.ndarray) -> jnp.ndarray:
        return base_u.at[index_array].set(z)

    def outputs(z: jnp.ndarray, base_u: jnp.ndarray):
        u = full_u(z, base_u)
        variables = denormalize_reactor_variables(u, problem.variable_bounds)
        reactor = evaluate_reactor(
            variables,
            problem.technology,
            problem.config,
            problem.numerics,
        )
        objective = objective_value(
            variables.design,
            reactor,
            problem.config,
            problem.optimization,
        )
        equalities = equality_vector(reactor, problem.config, problem.optimization)
        inequalities = inequality_vector(reactor, problem.optimization)
        return objective, equalities, inequalities, reactor

    def objective_fn(z: jnp.ndarray, base_u: jnp.ndarray) -> jnp.ndarray:
        return outputs(z, base_u)[0]

    def equality_fn(z: jnp.ndarray, base_u: jnp.ndarray) -> jnp.ndarray:
        return outputs(z, base_u)[1]

    def inequality_fn(z: jnp.ndarray, base_u: jnp.ndarray) -> jnp.ndarray:
        return outputs(z, base_u)[2]

    def metric_fn(z: jnp.ndarray, base_u: jnp.ndarray):
        objective, equalities, inequalities, reactor = outputs(z, base_u)
        eq_rms = jnp.sqrt(jnp.mean(jnp.square(equalities)))
        ineq_violation = jnp.max(jnp.maximum(-inequalities, 0.0))
        return (
            objective,
            eq_rms,
            ineq_violation,
            reactor.power.net_electric_power_mw,
            reactor.economics.coe_usd_mwh,
        )

    return {
        "objective": jax.jit(objective_fn),
        "objective_value_and_grad": jax.jit(
            jax.value_and_grad(objective_fn, argnums=0)
        ),
        "equalities": jax.jit(equality_fn),
        "equality_jacobian": jax.jit(jax.jacrev(equality_fn, argnums=0)),
        "inequalities": jax.jit(inequality_fn),
        "inequality_jacobian": jax.jit(jax.jacrev(inequality_fn, argnums=0)),
        "metrics": jax.jit(metric_fn),
    }


def evaluate_metrics(
    metric_fn: Callable,
    z: np.ndarray,
    base_u: np.ndarray,
) -> PointMetrics:
    values = metric_fn(jnp.asarray(z), jnp.asarray(base_u))
    numeric = [float(np.asarray(jax.device_get(v))) for v in values]
    return PointMetrics(*numeric)

def merit(
    metrics: PointMetrics,
    equality_penalty: float,
    inequality_penalty: float,
) -> float:
    values = np.asarray(
        [
            metrics.objective,
            metrics.equality_rms,
            metrics.inequality_violation,
            metrics.net_electric_power_mw,
            metrics.coe_usd_mwh,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        return 1.0e30
    return (
        metrics.objective
        + equality_penalty * metrics.equality_rms
        + inequality_penalty * metrics.inequality_violation
    )


def make_seeded_start(
    problem: ReactorProblem,
    seed: int,
    start_std: float,
) -> np.ndarray:
    """Perturb only continuous design variables; retain baseline closures."""
    rng = np.random.default_rng(seed)
    u0 = to_numpy(
        normalize_reactor_variables(problem.initial_variables, problem.variable_bounds)
    )
    perturbed = u0.copy()
    design_names = DesignVariables._fields
    design_indices = np.asarray(
        [design_names.index(name) for name in ACTIVE_DESIGN_NAMES], dtype=int
    )
    perturbed[design_indices] += rng.normal(0.0, start_std, size=len(design_indices))
    return np.clip(perturbed, 1.0e-4, 1.0 - 1.0e-4)


def point_to_trace(
    method: str,
    seed: int,
    iteration: int,
    calls: int,
    metrics: PointMetrics,
) -> TracePoint:
    return TracePoint(
        method=method,
        seed=seed,
        iteration=iteration,
        optimizer_calls=calls,
        objective=metrics.objective,
        equality_rms=metrics.equality_rms,
        inequality_violation=metrics.inequality_violation,
        net_electric_power_mw=metrics.net_electric_power_mw,
        coe_usd_mwh=metrics.coe_usd_mwh,
    )


def run_random_search(
    problem: ReactorProblem,
    reduced_funcs: dict[str, Callable],
    active_indices: np.ndarray,
    u_start: np.ndarray,
    *,
    seed: int,
    iterations: int,
    sigma_start: float,
    sigma_end: float,
    global_probability: float,
    equality_penalty: float,
    inequality_penalty: float,
) -> list[TracePoint]:
    """Derivative-free stochastic best-so-far search in normalized space."""
    rng = np.random.default_rng(10_000 + seed)
    base_u = np.asarray(u_start, dtype=float).copy()
    best_z = base_u[active_indices].copy()
    best_metrics = evaluate_metrics(reduced_funcs["metrics"], best_z, base_u)
    calls = 1
    best_merit = merit(best_metrics, equality_penalty, inequality_penalty)
    history = [point_to_trace(METHODS[0], seed, 0, calls, best_metrics)]

    for iteration in range(1, iterations):
        fraction = iteration / max(iterations - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** fraction

        if rng.random() < global_probability:
            proposal = rng.uniform(0.0, 1.0, size=best_z.size)
        else:
            proposal = best_z + rng.normal(0.0, sigma, size=best_z.size)
        proposal = np.clip(proposal, 1.0e-4, 1.0 - 1.0e-4)

        proposal_metrics = evaluate_metrics(reduced_funcs["metrics"], proposal, base_u)
        calls += 1
        proposal_merit = merit(
            proposal_metrics,
            equality_penalty,
            inequality_penalty,
        )
        if proposal_merit < best_merit:
            best_z = proposal
            best_metrics = proposal_metrics
            best_merit = proposal_merit

        history.append(
            point_to_trace(METHODS[0], seed, iteration, calls, best_metrics)
        )

    return history


def run_slsqp(
    problem: ReactorProblem,
    reduced_funcs: dict[str, Callable],
    active_indices: np.ndarray,
    u_start: np.ndarray,
    *,
    seed: int,
    iterations: int,
    ftol: float,
    use_ad: bool,
    disp: bool,
) -> tuple[OptimizeResult, list[TracePoint]]:
    """Run SLSQP with either JAX exact derivatives or finite differences."""
    base_u = np.asarray(u_start, dtype=float).copy()
    z_start = base_u[active_indices].copy()
    funcs = reduced_funcs
    counter = CallCounter()
    method_name = METHODS[2] if use_ad else METHODS[1]

    if use_ad:

        def fun(u: np.ndarray) -> tuple[float, np.ndarray]:
            counter.increment()
            value, gradient = funcs["objective_value_and_grad"](u, base_u)
            return float(value), to_numpy(gradient)

        def eq_fun(u: np.ndarray) -> np.ndarray:
            counter.increment()
            return to_numpy(funcs["equalities"](u, base_u))

        def eq_jac(u: np.ndarray) -> np.ndarray:
            counter.increment()
            return to_numpy(funcs["equality_jacobian"](u, base_u))

        def ineq_fun(u: np.ndarray) -> np.ndarray:
            counter.increment()
            return to_numpy(funcs["inequalities"](u, base_u))

        def ineq_jac(u: np.ndarray) -> np.ndarray:
            counter.increment()
            return to_numpy(funcs["inequality_jacobian"](u, base_u))

        objective_jac = True
        constraints = [
            {"type": "eq", "fun": eq_fun, "jac": eq_jac},
            {"type": "ineq", "fun": ineq_fun, "jac": ineq_jac},
        ]
    else:

        def fun(u: np.ndarray) -> float:
            counter.increment()
            return float(funcs["objective"](u, base_u))

        def eq_fun(u: np.ndarray) -> np.ndarray:
            counter.increment()
            return to_numpy(funcs["equalities"](u, base_u))

        def ineq_fun(u: np.ndarray) -> np.ndarray:
            counter.increment()
            return to_numpy(funcs["inequalities"](u, base_u))

        objective_jac = None
        # Omitting the Jacobians makes SLSQP estimate them by forward differences.
        constraints = [
            {"type": "eq", "fun": eq_fun},
            {"type": "ineq", "fun": ineq_fun},
        ]

    history: list[TracePoint] = []
    recorded_u: list[np.ndarray] = []

    def record(u: np.ndarray) -> None:
        u = np.asarray(u, dtype=float)
        if recorded_u and np.linalg.norm(u - recorded_u[-1]) < 1.0e-12:
            return
        metrics = evaluate_metrics(funcs["metrics"], u, base_u)
        history.append(
            point_to_trace(
                method_name,
                seed,
                len(history),
                counter.value,
                metrics,
            )
        )
        recorded_u.append(u.copy())

    record(z_start)
    result = minimize(
        fun,
        z_start,
        method="SLSQP",
        jac=objective_jac,
        bounds=[(0.0, 1.0)] * len(z_start),
        constraints=constraints,
        callback=record,
        options={"maxiter": iterations - 1, "ftol": ftol, "disp": disp},
    )
    record(result.x)

    # Pad early termination with the final accepted point to align seed arrays.
    while len(history) < iterations:
        final = history[-1]
        history.append(
            replace(
                final,
                iteration=len(history),
                optimizer_calls=counter.value,
            )
        )
    return result, history[:iterations]


def save_traces(traces: list[TracePoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(TracePoint.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in traces:
            writer.writerow(row.__dict__)


def aggregate_iteration(
    traces: list[TracePoint],
    method: str,
    field: str,
    seeds: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for seed in range(seeds):
        rows = sorted(
            (row for row in traces if row.method == method and row.seed == seed),
            key=lambda row: row.iteration,
        )
        if len(rows) < iterations:
            raise RuntimeError(
                f"Expected at least {iterations} points for {method}, seed {seed}; "
                f"got {len(rows)}"
            )
        rows = rows[:iterations]
        arrays.append(np.asarray([getattr(row, field) for row in rows], dtype=float))
    stacked = np.stack(arrays, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def aggregate_evaluations(
    traces: list[TracePoint],
    method: str,
    field: str,
    seeds: int,
    evaluation_budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Step-interpolate each seed's incumbent trajectory on call count."""
    grid = np.arange(evaluation_budget + 1, dtype=int)
    arrays = []
    for seed in range(seeds):
        rows = sorted(
            (row for row in traces if row.method == method and row.seed == seed),
            key=lambda row: (row.optimizer_calls, row.iteration),
        )
        if not rows:
            raise RuntimeError(f"No trace for {method}, seed {seed}")
        calls = np.asarray([row.optimizer_calls for row in rows], dtype=int)
        values = np.asarray([getattr(row, field) for row in rows], dtype=float)

        # For duplicate call counts retain the last accepted iterate.
        unique_calls = []
        unique_values = []
        for call, value in zip(calls, values):
            if unique_calls and call == unique_calls[-1]:
                unique_values[-1] = value
            else:
                unique_calls.append(int(call))
                unique_values.append(float(value))
        unique_calls = np.asarray(unique_calls, dtype=int)
        unique_values = np.asarray(unique_values, dtype=float)

        indices = np.searchsorted(unique_calls, grid, side="right") - 1
        indices = np.clip(indices, 0, len(unique_values) - 1)
        arrays.append(unique_values[indices])

    stacked = np.stack(arrays, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def plot_results(
    traces: list[TracePoint],
    *,
    seeds: int,
    iterations: int,
    evaluation_budget: int,
    x_axis: str,
    output: Path,
) -> None:
    if x_axis == "evaluations":
        x = np.arange(evaluation_budget + 1)
        #x_label = "Cumulative optimizer function calls"
        x_label = "Number of evaluations"
    else:
        x = np.arange(iterations)
        x_label = "Accepted optimization iteration"
    #fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), constrained_layout=True)

    panel_specs = (
        (
            axes[0],
            "objective",
            "Combined objective",
            r"$f=\mathrm{COE}/100-P_{\mathrm{net}}/1000$",
            False,
        ),
        (
            axes[1],
            "equality_rms",
            "Equality-constraint convergence",
            r"RMS equality residual $\|\mathbf{h}\|_2/\sqrt{N_h}$",
            True,
        ),
        (
            axes[2],
            "inequality_violation",
            "Inequality-constraint convergence",
            r"Maximum violation $\max(0,-\min\mathbf{g})$",
            True,
        ),
    )

    line_styles = ("-", "--", "-.")
    markers = ("o", "s", "^")

    for panel_index, (ax, field, title, ylabel, use_log) in enumerate(panel_specs):
        for method, linestyle, marker in zip(METHODS, line_styles, markers):
            if x_axis == "evaluations":
                mean, std = aggregate_evaluations(
                    traces, method, field, seeds, evaluation_budget
                )
            else:
                mean, std = aggregate_iteration(
                    traces, method, field, seeds, iterations
                )
            if use_log:
                floor = 1.0e-10
                mean_plot = np.maximum(mean, floor)
                lower = np.maximum(mean - std, floor)
                upper = np.maximum(mean + std, floor)
            else:
                mean_plot = mean
                lower = mean - std
                upper = mean + std

            line = ax.plot(
                x,
                mean_plot,
                linestyle=linestyle,
                marker=marker,
                markevery=max(iterations // 10, 1),
                markersize=4,
                linewidth=1.6,
                label=method,
            )[0]
            ax.fill_between(
                x,
                lower,
                upper,
                alpha=0.18,
                color=line.get_color(),
                linewidth=0.0,
            )

        if use_log:
            ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        #ax.set_title(f"({chr(ord('a') + panel_index)}) {title}")
        ax.grid(alpha=0.3)

    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def print_summary(
    traces: list[TracePoint],
    *,
    seeds: int,
    iterations: int,
    evaluation_budget: int,
    x_axis: str,
) -> None:
    print("\n=== Figure 5 summary: final mean +/- std ===")
    for method in METHODS:
        if x_axis == "evaluations":
            objective_mean, objective_std = aggregate_evaluations(
                traces, method, "objective", seeds, evaluation_budget
            )
            equality_mean, equality_std = aggregate_evaluations(
                traces, method, "equality_rms", seeds, evaluation_budget
            )
            inequality_mean, inequality_std = aggregate_evaluations(
                traces, method, "inequality_violation", seeds, evaluation_budget
            )
            calls_mean = np.full(evaluation_budget + 1, evaluation_budget, dtype=float)
            calls_std = np.zeros(evaluation_budget + 1, dtype=float)
        else:
            objective_mean, objective_std = aggregate_iteration(
                traces, method, "objective", seeds, iterations
            )
            equality_mean, equality_std = aggregate_iteration(
                traces, method, "equality_rms", seeds, iterations
            )
            inequality_mean, inequality_std = aggregate_iteration(
                traces, method, "inequality_violation", seeds, iterations
            )
            calls_mean, calls_std = aggregate_iteration(
                traces, method, "optimizer_calls", seeds, iterations
            )
        print(
            f"{method:<28} "
            f"F={objective_mean[-1]: .5f} +/- {objective_std[-1]:.5f}, "
            f"eq={equality_mean[-1]:.3e} +/- {equality_std[-1]:.3e}, "
            f"ineq={inequality_mean[-1]:.3e} +/- {inequality_std[-1]:.3e}, "
            f"optimizer calls={calls_mean[-1]:.1f} +/- {calls_std[-1]:.1f}"
        )


def main() -> None:
    args = parse_args()
    if args.iterations < 2:
        raise ValueError("--iterations must be at least 2")
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    if args.evaluation_budget < 1:
        raise ValueError("--evaluation-budget must be positive")

    problem = build_problem()
    active_indices = active_variable_indices()
    reduced_funcs = make_reduced_functions(problem, active_indices)
    traces: list[TracePoint] = []

    print("Active optimized variables:")
    print("  design:", ", ".join(ACTIVE_DESIGN_NAMES))
    print("  closures:", ", ".join(ClosureVariables._fields))

    for seed in range(args.seeds):
        print(f"\n--- seed {seed} ---")
        u_start = make_seeded_start(problem, seed, args.start_std)

        random_iterations = max(
            args.iterations,
            args.evaluation_budget if args.x_axis == "evaluations" else args.iterations,
        )
        random_trace = run_random_search(
            problem,
            reduced_funcs,
            active_indices,
            u_start,
            seed=seed,
            iterations=random_iterations,
            sigma_start=args.random_sigma_start,
            sigma_end=args.random_sigma_end,
            global_probability=args.random_global_prob,
            equality_penalty=args.equality_penalty,
            inequality_penalty=args.inequality_penalty,
        )
        traces.extend(random_trace)
        print("random search complete")

        fd_result, fd_trace = run_slsqp(
            problem,
            reduced_funcs,
            active_indices,
            u_start,
            seed=seed,
            iterations=args.iterations,
            ftol=args.ftol,
            use_ad=False,
            disp=args.disp,
        )
        traces.extend(fd_trace)
        print(
            "finite difference:",
            fd_result.success,
            fd_result.message,
            f"nit={fd_result.nit}",
            f"nfev={fd_result.nfev}",
        )

        ad_result, ad_trace = run_slsqp(
            problem,
            reduced_funcs,
            active_indices,
            u_start,
            seed=seed,
            iterations=args.iterations,
            ftol=args.ftol,
            use_ad=True,
            disp=args.disp,
        )
        traces.extend(ad_trace)
        print(
            "automatic differentiation:",
            ad_result.success,
            ad_result.message,
            f"nit={ad_result.nit}",
            f"nfev={ad_result.nfev}",
        )

    csv_path = args.output.with_suffix(".csv")
    save_traces(traces, csv_path)
    plot_results(
        traces,
        seeds=args.seeds,
        iterations=args.iterations,
        evaluation_budget=args.evaluation_budget,
        x_axis=args.x_axis,
        output=args.output,
    )
    print_summary(
        traces,
        seeds=args.seeds,
        iterations=args.iterations,
        evaluation_budget=args.evaluation_budget,
        x_axis=args.x_axis,
    )
    print(f"\nPNG: {args.output.with_suffix('.png')}")
    print(f"PDF: {args.output.with_suffix('.pdf')}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
