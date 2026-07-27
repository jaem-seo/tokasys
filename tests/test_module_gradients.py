import jax
import jax.numpy as jnp

from tokasys import default_problem, evaluate_reactor
from tokasys.exhaust.divertor import evaluate_exhaust
from tokasys.plasma.fusion import dt_reactivity_m3_s
from tokasys.plasma.radiation import electron_ion_bremsstrahlung_mw


def assert_positive_finite(value):
    assert jnp.isfinite(value)
    assert value > 0.0


def assert_negative_finite(value):
    assert jnp.isfinite(value)
    assert value < 0.0


def test_dt_reactivity_is_monotone_with_positive_gradient():
    temperatures = jnp.asarray([6.0, 12.0, 20.0])
    reactivities = dt_reactivity_m3_s(temperatures)
    gradient = jax.grad(lambda t: dt_reactivity_m3_s(t))(12.0)

    assert jnp.all(jnp.diff(reactivities) > 0.0)
    assert_positive_finite(gradient)


def test_fusion_power_target_temperature_gradient_is_positive():
    problem = default_problem()

    def fusion_power(target_temperature_keV):
        design = problem.initial_design._replace(
            target_temperature_keV=target_temperature_keV
        )
        return evaluate_reactor(
            design,
            problem.technology,
            problem.config,
            problem.numerics,
        ).plasma.fusion_power_mw

    assert fusion_power(16.0) > fusion_power(12.0)
    assert_positive_finite(jax.grad(fusion_power)(14.0))


def test_synchrotron_gradients_track_reflectivity_field_and_temperature():
    problem = default_problem()

    def synchrotron_with_reflectivity(reflectivity):
        config = problem.config._replace(synchrotron_wall_reflectivity=reflectivity)
        return evaluate_reactor(
            problem.initial_design,
            problem.technology,
            config,
            problem.numerics,
        ).plasma.synchrotron_mw

    def synchrotron_with_field(field_t):
        design = problem.initial_design._replace(toroidal_field_t=field_t)
        return evaluate_reactor(
            design,
            problem.technology,
            problem.config,
            problem.numerics,
        ).plasma.synchrotron_mw

    def synchrotron_with_temperature(target_temperature_keV):
        design = problem.initial_design._replace(
            target_temperature_keV=target_temperature_keV
        )
        return evaluate_reactor(
            design,
            problem.technology,
            problem.config,
            problem.numerics,
        ).plasma.synchrotron_mw

    assert synchrotron_with_reflectivity(0.90) < synchrotron_with_reflectivity(0.70)
    assert synchrotron_with_field(6.0) > synchrotron_with_field(5.0)
    assert synchrotron_with_temperature(15.0) > synchrotron_with_temperature(12.0)
    assert_negative_finite(jax.grad(synchrotron_with_reflectivity)(0.80))
    assert_positive_finite(jax.grad(synchrotron_with_field)(5.7))
    assert_positive_finite(jax.grad(synchrotron_with_temperature)(14.0))


def test_bremsstrahlung_uses_profile_power_density_formula():
    power = electron_ion_bremsstrahlung_mw(
        electron_density_m3=jnp.asarray(1.0e20),
        temperature_keV=jnp.asarray(16.0),
        zeff=2.0,
        volume_m3=jnp.asarray(1.0),
    )

    assert jnp.isclose(power, 5.35e-3 * 2.0 * 4.0)


def test_synchrotron_profile_beta_is_derived_from_temperature_profile():
    problem = default_problem()
    base = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(synchrotron_profile_beta=1.0),
        problem.numerics,
    ).plasma
    changed_config = evaluate_reactor(
        problem.initial_design,
        problem.technology,
        problem.config._replace(synchrotron_profile_beta=6.0),
        problem.numerics,
    ).plasma
    hotter = evaluate_reactor(
        problem.initial_design._replace(target_temperature_keV=18.0),
        problem.technology,
        problem.config,
        problem.numerics,
    ).plasma

    assert base.synchrotron_profile_beta > 0.0
    assert jnp.isclose(base.synchrotron_mw, changed_config.synchrotron_mw)
    assert not jnp.isclose(base.synchrotron_profile_beta, hotter.synchrotron_profile_beta)


def test_cs_flux_gradient_is_positive_with_pulse_duration():
    problem = default_problem()
    config = problem.config._replace(steady_state=False)

    def required_flux(flat_top_duration_s):
        tech = problem.technology._replace(
            cs_pulse_flat_top_duration_s=flat_top_duration_s
        )
        return evaluate_reactor(
            problem.initial_design,
            tech,
            config,
            problem.numerics,
        ).magnets.cs_flux_required_wb

    assert required_flux(9000.0) > required_flux(3600.0)
    assert_positive_finite(jax.grad(required_flux)(7200.0))


def test_ohmic_power_tracks_inductive_current_fraction():
    problem = default_problem()

    def ohmic_power(current_drive_power_mw):
        design = problem.initial_design._replace(
            current_drive_power_mw=current_drive_power_mw,
        )
        return evaluate_reactor(
            design,
            problem.technology,
            problem.config,
            problem.numerics,
        ).plasma.ohmic_power_mw

    assert ohmic_power(160.0) < ohmic_power(0.0)
    assert_negative_finite(jax.grad(ohmic_power)(80.0))


def test_tbr_gradient_is_positive_with_blanket_thickness():
    problem = default_problem()

    def tbr(blanket_thickness_m):
        design = problem.initial_design._replace(
            inboard_blanket_thickness_m=blanket_thickness_m,
            outboard_blanket_thickness_m=blanket_thickness_m,
        )
        return evaluate_reactor(
            design,
            problem.technology,
            problem.config,
            problem.numerics,
        ).nuclear.tbr

    assert tbr(1.10) > tbr(0.70)
    assert_positive_finite(jax.grad(tbr)(0.85))


def test_divertor_heat_flux_gradient_tracks_plasma_current_scaling():
    problem = default_problem()
    tech = problem.technology._replace(divertor_spreading_factor_m=0.005)
    base = evaluate_reactor(
        problem.initial_design,
        tech,
        problem.config,
        problem.numerics,
    )

    def peak_heat_flux(plasma_current_ma):
        design = problem.initial_design._replace(plasma_current_ma=plasma_current_ma)
        return evaluate_exhaust(
            design,
            base.geometry,
            base.plasma,
            tech,
            problem.config,
        ).peak_divertor_heat_flux_mw_m2

    def target_temperature(plasma_current_ma):
        design = problem.initial_design._replace(plasma_current_ma=plasma_current_ma)
        return evaluate_exhaust(
            design,
            base.geometry,
            base.plasma,
            tech,
            problem.config,
        ).divertor_target_temperature_c

    def divertor_lifetime(plasma_current_ma):
        design = problem.initial_design._replace(plasma_current_ma=plasma_current_ma)
        return evaluate_exhaust(
            design,
            base.geometry,
            base.plasma,
            tech,
            problem.config,
        ).divertor_lifetime_fpy

    assert peak_heat_flux(20.0) > peak_heat_flux(16.0)
    assert target_temperature(20.0) > target_temperature(16.0)
    assert divertor_lifetime(20.0) < divertor_lifetime(16.0)
    assert_positive_finite(jax.grad(peak_heat_flux)(16.0))
    assert_positive_finite(jax.grad(target_temperature)(16.0))
    assert_negative_finite(jax.grad(divertor_lifetime)(16.0))


def test_coe_gradients_track_availability_and_capital_cost():
    problem = default_problem()

    def coe_with_availability(availability):
        tech = problem.technology._replace(availability=availability)
        return evaluate_reactor(
            problem.initial_design,
            tech,
            problem.config,
            problem.numerics,
        ).economics.coe_usd_mwh

    def coe_with_magnet_capital_scale(scale):
        tech = problem.technology._replace(
            base_magnet_cost_busd=scale * problem.technology.base_magnet_cost_busd
        )
        return evaluate_reactor(
            problem.initial_design,
            tech,
            problem.config,
            problem.numerics,
        ).economics.coe_usd_mwh

    assert coe_with_availability(0.80) < coe_with_availability(0.70)
    assert coe_with_magnet_capital_scale(1.10) > coe_with_magnet_capital_scale(0.90)
    assert_negative_finite(jax.grad(coe_with_availability)(0.75))
    assert_positive_finite(jax.grad(coe_with_magnet_capital_scale)(1.0))
