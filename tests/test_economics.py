import jax.numpy as jnp

from tokasys.core.defaults import default_problem
from tokasys.models.reactor import evaluate_reactor


def test_magnet_cost_includes_pf_coil_cost():
    problem = default_problem()
    result = evaluate_reactor(
        problem.initial_variables,
        problem.technology,
        problem.config,
        problem.numerics,
    )

    assert result.economics.tf_coil_cost_busd > 0.0
    assert result.economics.pf_coil_cost_busd > 0.0
    assert jnp.isclose(
        result.economics.magnet_cost_busd,
        result.economics.tf_coil_cost_busd + result.economics.pf_coil_cost_busd,
    )


def test_pf_coil_cost_increases_with_pf_coil_size():
    problem = default_problem()
    base_result = evaluate_reactor(
        problem.initial_variables,
        problem.technology,
        problem.config,
        problem.numerics,
    )
    thicker_design = problem.initial_design._replace(
        pf_coil_thickness_m=1.2 * problem.initial_design.pf_coil_thickness_m
    )
    thicker_result = evaluate_reactor(
        thicker_design,
        problem.technology,
        problem.config,
        problem.numerics,
    )

    assert thicker_result.economics.pf_coil_cost_busd > (
        base_result.economics.pf_coil_cost_busd
    )
    assert jnp.isclose(
        thicker_result.economics.tf_coil_cost_busd,
        base_result.economics.tf_coil_cost_busd,
    )
