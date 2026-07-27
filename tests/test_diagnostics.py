import numpy as np

from tokasys import default_problem, evaluate_reactor
from tokasys.diagnostics import diagnostic_current_and_q_profiles


def test_diagnostic_current_profile_includes_bootstrap_component():
    problem = default_problem()
    result = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config,
        problem.numerics,
    )
    profiles = diagnostic_current_and_q_profiles(problem.initial_design, result)

    assert np.max(profiles["bootstrap_current_density_ma_m2"]) > 0.0
    assert np.max(profiles["current_drive_current_density_ma_m2"]) > 0.0
    assert np.max(profiles["inductive_current_density_ma_m2"]) > 0.0
    np.testing.assert_allclose(
        profiles["total_current_density_ma_m2"],
        profiles["bootstrap_current_density_ma_m2"]
        + profiles["current_drive_current_density_ma_m2"]
        + profiles["inductive_current_density_ma_m2"],
    )
    assert profiles["enclosed_current_fraction"][0] == 0.0
    assert np.isclose(profiles["enclosed_current_fraction"][-1], 1.0)
