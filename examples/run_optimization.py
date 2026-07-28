from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from tokasys import default_problem
from tokasys.diagnostics import plot_reactor_diagnostics
from tokasys.models.constraints import (
    equality_vector,
    inequality_vector,
)
from tokasys.plasma.profiles import diagnostic_q_profile_and_shear
from tokasys.solvers.config import apply_optimization_config
from tokasys.solvers.optimize import solve_slsqp
from tokasys.solvers.reporting import (
    equality_reports,
    inequality_reports,
    objective_reports,
)


jax.config.update("jax_enable_x64", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TokaSys SLSQP optimization.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON/JSONC optimization config file.",
    )
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--ftol", type=float, default=1.0e-9)
    parser.add_argument("--no-disp", action="store_true", help="Disable SciPy output.")
    return parser.parse_args()


args = parse_args()
problem = default_problem()
if args.config is not None:
    problem = apply_optimization_config(problem, args.config)
solution = solve_slsqp(problem, maxiter=args.maxiter, ftol=args.ftol, disp=not args.no_disp)


def _status(ok: bool) -> str:
    return "OK" if ok else "VIOLATED"


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, dtype=float) * np.asarray(weights, dtype=float)))


def _id_label(value: int, labels: dict[int, str]) -> str:
    return f"{value} ({labels.get(value, 'custom')})"


def print_reactor_config() -> None:
    print("\n--- Reactor config ---")
    print(f"energy_closure_mode: {problem.config.energy_closure_mode}")
    print(
        "tf_superconductor_id:",
        _id_label(problem.config.tf_superconductor_id, {0: "REBCO", 1: "Nb3Sn", 2: "NbTi"}),
    )
    print(
        "pf_superconductor_id:",
        _id_label(problem.config.pf_superconductor_id, {0: "REBCO", 1: "Nb3Sn", 2: "NbTi"}),
    )
    print(
        "cs_superconductor_id:",
        _id_label(problem.config.cs_superconductor_id, {0: "REBCO", 1: "Nb3Sn", 2: "NbTi"}),
    )
    print(
        "blanket_concept_id:",
        _id_label(problem.config.blanket_concept_id, {0: "LiPb/WCLL", 1: "FLiBe", 2: "HCPB"}),
    )
    print(
        "cost_model_id:",
        _id_label(
            problem.config.cost_model_id,
            {
                0: "legacy proxy",
                1: "n=0.6 capacity scaling",
                2: "Jo et al. (2021)",
            },
        ),
    )
    print(
        "divertor_heat_load_model_id:",
        _id_label(
            problem.config.divertor_heat_load_model_id,
            {0: "fixed lambda", 1: "Eich-like Bp scaling"},
        ),
    )


def print_plasma_information() -> None:
    plasma = solution.reactor.plasma
    q_profile, _, _, _ = diagnostic_q_profile_and_shear(
        plasma.radial_grid,
        plasma.q95,
    )
    weights = np.asarray(plasma.volume_weights, dtype=float)
    te = np.asarray(plasma.electron_temperature_profile_keV, dtype=float)
    ti = np.asarray(plasma.ion_temperature_profile_keV, dtype=float)
    ne = np.asarray(plasma.electron_density_profile_m3, dtype=float)
    non_inductive_current = (
        float(plasma.bootstrap_current_ma) + float(plasma.current_drive_current_ma)
    )
    non_inductive_fraction = non_inductive_current / max(
        float(solution.design.plasma_current_ma),
        1.0e-30,
    )

    print("\n--- Plasma information ---")
    print(f"Q: {float(plasma.fusion_gain):.4f}")
    print(
        "temperature:",
        f"Te0/Ti0={float(te[0]):.4f}/{float(ti[0]):.4f} keV, "
        f"<Te>/<Ti>={_weighted_average(te, weights):.4f}/{_weighted_average(ti, weights):.4f} keV, "
        f"Tped={float(plasma.pedestal_temperature_keV):.4f} keV",
    )
    print(f"average density: {_weighted_average(ne, weights) / 1.0e19:.4f} 1e19 m^-3")
    print(f"non-inductive fraction: {non_inductive_fraction:.4f}")
    print(
        "q profile:",
        f"q0={float(q_profile[0]):.4f}, "
        f"qmin={float(np.min(np.asarray(q_profile))):.4f}, "
        f"q95={float(plasma.q95):.4f}, "
        f"qedge={float(q_profile[-1]):.4f}",
    )


def print_optimization_report() -> None:
    print("\n--- Objective terms ---")
    for row in objective_reports(
        solution.variables,
        solution.reactor,
        problem.config,
        problem.optimization,
    ):
        print(
            f"{row['name']:<32} actual={row['actual']: .6g} {row['unit']:<8} "
            f"objective={row['objective_value']: .6g}"
        )

    print("\n--- Equality constraints ---")
    eq_rows = equality_reports(
        solution.variables,
        solution.reactor,
        problem.technology,
        problem.config,
        problem.optimization,
    )
    if not eq_rows:
        print("(none)")
    for row in eq_rows:
        print(
            f"{row['name']:<32} lhs={row['lhs']: .6g} {row['lhs_label']:<16} "
            f"rhs={row['rhs']: .6g} {row['rhs_label']:<16} "
            f"residual={row['residual']: .6g} {_status(row['satisfied'])}"
        )

    print("\n--- Inequality constraints ---")
    ineq_rows = inequality_reports(
        solution.variables,
        solution.reactor,
        problem.technology,
        problem.optimization,
    )
    if not ineq_rows:
        print("(none)")
    for row in ineq_rows:
        print(
            f"{row['name']:<32} actual={row['actual']: .6g} {row['actual_unit']:<10} "
            f"{row['sense']} limit={row['limit']: .6g} {row['limit_unit']:<14} "
            f"margin={row['margin']: .6g} {_status(row['satisfied'])}"
        )

print("\n=== Optimization result ===")
if args.config is not None:
    print(f"config: {args.config}")
print("success:", solution.scipy_result.success)
print("message:", solution.scipy_result.message)
print("objective:", solution.scipy_result.fun)
print(f"R0: {float(solution.design.major_radius_m):.4f} m")
print(f"A : {float(solution.design.aspect_ratio):.4f}")
print(f"B0: {float(solution.design.toroidal_field_t):.4f} T")
print(f"Ip: {float(solution.design.plasma_current_ma):.4f} MA")
print(f"H98: {float(solution.design.confinement_h98):.4f}")
print(f"TF coil thickness: {float(solution.design.tf_coil_thickness_m):.4f} m")
print(f"PF coil thickness: {float(solution.design.pf_coil_thickness_m):.4f} m")
print(f"CS radial build: {float(solution.design.cs_thickness_m):.4f} m")
external_heating_mw = (
    float(solution.design.auxiliary_heating_mw)
    + float(solution.design.current_drive_power_mw)
)
external_wallplug_mw = (
    float(solution.reactor.power.heating_wallplug_power_mw)
    + float(solution.reactor.power.current_drive_wallplug_power_mw)
)
print(
    "External heating:",
    f"{external_heating_mw:.3f} MW injected "
    f"(aux={float(solution.design.auxiliary_heating_mw):.3f}, "
    f"CD={float(solution.design.current_drive_power_mw):.3f})",
)
print(
    "External heating wall-plug:",
    f"{external_wallplug_mw:.3f} MW "
    f"(aux={float(solution.reactor.power.heating_wallplug_power_mw):.3f}, "
    f"CD={float(solution.reactor.power.current_drive_wallplug_power_mw):.3f})",
)
print(f"Fusion power: {float(solution.reactor.plasma.fusion_power_mw):.3f} MW")
print(f"Net power: {float(solution.reactor.power.net_electric_power_mw):.3f} MW")
print(f"COE: {float(solution.reactor.economics.coe_usd_mwh):.3f} USD/MWh")
print(
    "active equalities:",
    np.asarray(equality_vector(solution.reactor, problem.config, problem.optimization)),
)
active_margins = np.asarray(inequality_vector(solution.reactor, problem.optimization))
if active_margins.size:
    print("active min margin:", float(np.min(active_margins)))
else:
    print("active min margin: (none)")
print_reactor_config()
print_plasma_information()
print_optimization_report()

figure_path = Path(__file__).with_name("optimization_diagnostics.png")
plot_reactor_diagnostics(
    solution.design,
    problem.technology,
    solution.reactor,
    figure_path,
    title="TokaSys Optimization Diagnostics",
)
print(f"diagnostic figure: {figure_path}")
