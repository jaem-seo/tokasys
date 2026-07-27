from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from tokasys import default_problem, evaluate_reactor
from tokasys.core.variables import normalize_reactor_variables
from tokasys.diagnostics import plot_reactor_diagnostics
from tokasys.models.constraints import equality_vector, inequality_vector
from tokasys.solvers.interfaces import make_vector_functions


jax.config.update("jax_enable_x64", True)


problem = default_problem()
result = evaluate_reactor(
    problem.initial_design,
    problem.technology,
    problem.config,
    problem.numerics,
)
funcs = make_vector_functions(problem)
u0 = normalize_reactor_variables(problem.initial_variables, problem.variable_bounds)
value, gradient = funcs["objective_value_and_grad"](u0)

print("=== Baseline TokaSys evaluation ===")
print(f"Major radius            : {float(problem.initial_design.major_radius_m):8.3f} m")
print(f"Fusion power            : {float(result.plasma.fusion_power_mw):8.2f} MW")
print(f"Alpha power             : {float(result.plasma.alpha_power_mw):8.2f} MW")
print(f"Net electric power      : {float(result.power.net_electric_power_mw):8.2f} MW")
print(f"Cost of electricity     : {float(result.economics.coe_usd_mwh):8.2f} USD/MWh")
print(
    f"q95 / beta_N            : {float(result.plasma.q95):8.3f} / "
    f"{float(result.plasma.beta_n):8.3f}"
)
print(
    f"s_edge / s_95           : {float(result.plasma.magnetic_shear_edge):8.3f} / "
    f"{float(result.plasma.magnetic_shear_95):8.3f}"
)
print(f"Pedestal width / Tped   : {float(result.plasma.pedestal_width):8.3f} / "
      f"{float(result.plasma.pedestal_temperature_keV):8.3f} keV")
print(
    f"Bootstrap / CD current  : {float(result.plasma.bootstrap_current_ma):8.3f} / "
    f"{float(result.plasma.current_drive_current_ma):8.3f} MA"
)
print(f"Peak TF field           : {float(result.magnets.peak_tf_field_t):8.3f} T")
print(f"TBR                     : {float(result.nuclear.tbr):8.3f}")
print(
    f"Divertor heat flux      : "
    f"{float(result.exhaust.peak_divertor_heat_flux_mw_m2):8.3f} MW/m2"
)
print(
    f"TF / PF / CS thickness  : "
    f"{float(problem.initial_design.tf_coil_thickness_m):8.3f} / "
    f"{float(problem.initial_design.pf_coil_thickness_m):8.3f} / "
    f"{float(problem.initial_design.cs_thickness_m):8.3f} m"
)
print("Equality residuals      :", np.asarray(equality_vector(result)))
print("Minimum margin          :", float(np.min(np.asarray(inequality_vector(result)))))
print("Normalized objective    :", float(value))
print("Objective grad shape    :", gradient.shape)
print("Objective grad norm     :", float(np.linalg.norm(np.asarray(gradient))))

figure_path = Path(__file__).with_name("baseline_diagnostics.png")
plot_reactor_diagnostics(
    problem.initial_design,
    problem.technology,
    result,
    figure_path,
    title="TokaSys Baseline Diagnostics",
)
print(f"Diagnostic figure       : {figure_path}")
