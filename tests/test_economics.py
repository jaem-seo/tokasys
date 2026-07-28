import jax.numpy as jnp
import pytest

from tokasys.core.defaults import default_problem
from tokasys.models.reactor import evaluate_reactor
from tokasys.solvers.config import apply_optimization_config


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


def test_cost_model_zero_preserves_legacy_default():
    problem = default_problem()
    default_result = evaluate_reactor(
        problem.initial_variables,
        problem.technology,
        problem.config,
        problem.numerics,
    )
    explicit_result = evaluate_reactor(
        problem.initial_variables,
        problem.technology,
        problem.config._replace(cost_model_id=0),
        problem.numerics,
    )

    assert problem.config.cost_model_id == 0
    assert explicit_result.economics == default_result.economics


def test_cost_model_one_uses_point_six_capacity_scaling():
    problem = default_problem()
    result = evaluate_reactor(
        problem.initial_variables,
        problem.technology,
        problem.config._replace(cost_model_id=1),
        problem.numerics,
    )
    scale = (result.power.gross_electric_power_mw / 1000.0) ** 0.6
    tf_multiplier = problem.technology.rebco_magnet_cost_multiplier
    expected_tf = problem.technology.base_magnet_cost_busd * tf_multiplier * scale
    expected_pf = (
        problem.technology.base_pf_coil_cost_busd * tf_multiplier * scale
    )

    assert result.economics.tf_coil_cost_busd == pytest.approx(expected_tf)
    assert result.economics.pf_coil_cost_busd == pytest.approx(expected_pf)
    assert result.economics.magnet_cost_busd == pytest.approx(
        expected_tf + expected_pf
    )


def test_cost_model_two_follows_jo_capital_and_coe_accounting():
    problem = default_problem()
    result = evaluate_reactor(
        problem.initial_variables,
        problem.technology,
        problem.config._replace(cost_model_id=2),
        problem.numerics,
    )
    economics = result.economics

    assert economics.direct_capital_cost_busd == pytest.approx(
        economics.magnet_cost_busd
        + economics.nuclear_island_cost_busd
        + economics.balance_of_plant_cost_busd
        + economics.hcd_cost_busd
    )
    assert economics.contingency_cost_busd == pytest.approx(
        0.15 * economics.direct_capital_cost_busd
    )
    jo_direct_capital = (
        economics.direct_capital_cost_busd + economics.contingency_cost_busd
    )
    assert economics.total_capital_cost_busd == pytest.approx(
        jo_direct_capital * 1.075 * 1.375
    )
    assert economics.total_capital_cost_busd == pytest.approx(
        economics.direct_capital_cost_busd
        + economics.contingency_cost_busd
        + economics.indirect_cost_busd
    )
    assert economics.annualized_capital_cost_musd == pytest.approx(
        0.1 * economics.total_capital_cost_busd * 1000.0
    )
    assert economics.annual_om_cost_musd == pytest.approx(
        108.0 * (result.power.net_electric_power_mw / 1200.0) ** 0.5
    )
    assert economics.coe_usd_mwh == pytest.approx(
        economics.capital_coe_usd_mwh
        + economics.om_coe_usd_mwh
        + economics.maintenance_coe_usd_mwh
    )
    assert economics.tf_coil_cost_busd > 0.0
    assert economics.pf_coil_cost_busd > 0.0


def test_optimization_config_rejects_unknown_cost_model(tmp_path):
    config_path = tmp_path / "invalid_cost_model.json"
    config_path.write_text(
        """
        {
          "objectives": {"major_radius": true},
          "reactor_config": {"cost_model_id": 3}
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cost_model_id must be 0"):
        apply_optimization_config(default_problem(), config_path)
