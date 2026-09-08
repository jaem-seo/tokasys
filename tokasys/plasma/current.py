"""Ohmic, bootstrap, and non-inductive current-drive closures.

Refs:
    Spitzer, Physics of Fully Ionized Gases (1962), for classical resistivity.
    Sauter, Angioni and Lin-Liu, Phys. Plasmas 6, 2834-2839 (1999), for the
    bootstrap-current/collisionality structure used here as a reduced closure.
    Fisch, Rev. Mod. Phys. 59, 175-234 (1987), for normalized current-drive
    efficiency concepts. The CD efficiency here remains a generic surrogate.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import DesignVariables, GeometryState, TechnologyParameters
from tokasys.plasma.profiles import profile_average, radial_derivative


def spitzer_resistivity_ohm_m(
    temperature_keV: jnp.ndarray,
    zeff: float,
    coulomb_logarithm: float = 17.0,
) -> jnp.ndarray:
    """Evaluate parallel Spitzer-Harm resistivity from temperature and Zeff."""
    te_ev = jnp.maximum(temperature_keV, 1.0e-3) * 1.0e3
    z = zeff
    finite_z = (1.0 + 1.198 * z + 0.222 * z**2) / (
        1.0 + 2.966 * z + 0.753 * z**2
    )
    return 1.03136e-4 * z * coulomb_logarithm * finite_z / te_ev**1.5


def bootstrap_current_model(
    rho: jnp.ndarray,
    weights: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    electron_temperature_keV: jnp.ndarray,
    pressure_profile_pa: jnp.ndarray,
    q95: jnp.ndarray,
    beta_p: jnp.ndarray,
    geom: GeometryState,
    x: DesignVariables,
    tech: TechnologyParameters,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Estimate bootstrap fraction, collisionality, and pressure-gradient drive."""
    pressure_average = jnp.maximum(profile_average(pressure_profile_pa, weights), 1.0)
    dp_drho = radial_derivative(pressure_profile_pa, rho)
    gradient_drive = profile_average(jnp.maximum(-rho * dp_drho, 0.0), weights)
    gradient_factor = jnp.clip(
        gradient_drive / jnp.maximum(2.0 * pressure_average, 1.0e-30),
        0.25,
        3.0,
    )

    eps = jnp.maximum(geom.inverse_aspect_ratio, 1.0e-4)
    te_ev = jnp.maximum(electron_temperature_keV * 1.0e3, 1.0)
    collisionality_profile = (
        6.921e-18
        * jnp.maximum(q95, 1.0e-6)
        * x.major_radius_m
        * electron_density_m3
        * tech.zeff
        * 17.0
        / (eps**1.5 * te_ev**2)
    )
    pressure_weight = pressure_profile_pa * weights / jnp.maximum(
        profile_average(pressure_profile_pa, weights),
        1.0e-30,
    )
    collisionality = jnp.sum(collisionality_profile * pressure_weight)
    collisionality_factor = 1.0 / (1.0 + jnp.sqrt(jnp.maximum(collisionality, 0.0)))

    shape_factor = jnp.clip(
        1.0 + 0.15 * (x.elongation - 1.0) + 0.10 * x.triangularity,
        0.75,
        1.35,
    )
    bootstrap_fraction = (
        tech.bootstrap_coefficient
        * jnp.sqrt(eps)
        * beta_p
        * gradient_factor
        * collisionality_factor
        * shape_factor
    )
    return jnp.clip(bootstrap_fraction, 0.0, 0.95), collisionality, gradient_factor


def current_drive_model(
    x: DesignVariables,
    geom: GeometryState,
    tech: TechnologyParameters,
    weights: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    electron_temperature_keV: jnp.ndarray,
    pressure_profile_pa: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate driven current and effective normalized current-drive efficiency."""
    pressure_weight = pressure_profile_pa * weights / jnp.maximum(
        profile_average(pressure_profile_pa, weights),
        1.0e-30,
    )
    te_cd = jnp.sum(electron_temperature_keV * pressure_weight)
    te_avg = profile_average(electron_temperature_keV, weights)
    te_factor = jnp.clip(jnp.sqrt(jnp.maximum(te_cd, 1.0e-6) / 10.0), 0.50, 1.70)
    zeff_factor = 2.0 / (1.0 + tech.zeff)
    trapped_particle_factor = 1.0 / (
        1.0 + 0.35 * jnp.sqrt(jnp.maximum(geom.inverse_aspect_ratio, 0.0))
    )
    profile_factor = jnp.clip(
        1.0
        + 0.08
        * (electron_temperature_keV[0] / jnp.maximum(te_avg, 1.0e-30) - 1.0),
        0.80,
        1.25,
    )
    efficiency_20 = (
        tech.current_drive_efficiency_20
        * te_factor
        * zeff_factor
        * trapped_particle_factor
        * profile_factor
    )
    ne20 = profile_average(electron_density_m3, weights) / 1.0e20
    current_ma = (
        efficiency_20
        * x.current_drive_power_mw
        / jnp.maximum(ne20 * x.major_radius_m, 1.0e-9)
    )
    return current_ma, efficiency_20


def _current_density_from_shape(
    shape: jnp.ndarray,
    current_ma: jnp.ndarray,
    geom: GeometryState,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Normalize a radial shape to carry the specified total plasma current."""
    normalized_shape = shape / jnp.maximum(profile_average(shape, weights), 1.0e-30)
    average_current_density = current_ma * 1.0e6 / jnp.maximum(
        geom.plasma_cross_section_m2,
        1.0e-30,
    )
    return average_current_density * normalized_shape


def current_density_profiles(
    rho: jnp.ndarray,
    weights: jnp.ndarray,
    pressure_profile_pa: jnp.ndarray,
    geom: GeometryState,
    x: DesignVariables,
    bootstrap_current_ma: jnp.ndarray,
    current_drive_current_ma: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return total, inductive, bootstrap and current-drive j_phi profiles."""
    requested_non_inductive = jnp.maximum(
        bootstrap_current_ma + current_drive_current_ma,
        0.0,
    )
    non_inductive_scale = jnp.minimum(
        1.0,
        x.plasma_current_ma / jnp.maximum(requested_non_inductive, 1.0e-30),
    )
    bootstrap_current = jnp.maximum(bootstrap_current_ma, 0.0) * non_inductive_scale
    current_drive_current = jnp.maximum(current_drive_current_ma, 0.0) * non_inductive_scale
    inductive_current = jnp.maximum(
        x.plasma_current_ma - bootstrap_current - current_drive_current,
        0.0,
    )

    inductive_shape = jnp.maximum(1.0 - rho**1.2, 0.0) ** 2.0
    dp_drho = radial_derivative(pressure_profile_pa, rho)
    bootstrap_shape = jnp.maximum(-rho * dp_drho, 0.0)
    current_drive_shape = jnp.maximum(pressure_profile_pa, 0.0)

    inductive_j = _current_density_from_shape(
        inductive_shape,
        inductive_current,
        geom,
        weights,
    )
    bootstrap_j = _current_density_from_shape(
        bootstrap_shape,
        bootstrap_current,
        geom,
        weights,
    )
    current_drive_j = _current_density_from_shape(
        current_drive_shape,
        current_drive_current,
        geom,
        weights,
    )
    total_j = inductive_j + bootstrap_j + current_drive_j
    return total_j, inductive_j, bootstrap_j, current_drive_j


def ohmic_power_density_mw_m3(
    resistivity_ohm_m: jnp.ndarray,
    inductive_current_density_a_m2: jnp.ndarray,
    total_current_density_a_m2: jnp.ndarray,
) -> jnp.ndarray:
    """Ohmic heating density eta*j_ind*j_tot in MW/m3."""
    return (
        resistivity_ohm_m
        * inductive_current_density_a_m2
        * total_current_density_a_m2
        / 1.0e6
    )
