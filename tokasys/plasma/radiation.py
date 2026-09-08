"""Plasma radiation models.

Refs:
    Albajar, Johner and Granata, Nucl. Fusion 41, 665-678 (2001), for the
    global synchrotron-loss fit with wall reflectivity/optical-depth effects.
    Bremsstrahlung uses the standard nonrelativistic Z_eff n_e^2 sqrt(T_e)
    engineering form commonly tabulated in tokamak system-code references.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import ReactorConfig
from tokasys.plasma.profiles import effective_profile_alpha, profile_average


def electron_ion_bremsstrahlung_mw(
    electron_density_m3: jnp.ndarray,
    temperature_keV: jnp.ndarray,
    zeff: float,
    volume_m3: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate nonrelativistic electron-ion bremsstrahlung over the plasma."""
    ne20 = electron_density_m3 / 1.0e20
    te_kev = jnp.maximum(temperature_keV, 1.0e-6)
    power_density_mw_m3 = 5.35e-3 * zeff * ne20**2 * jnp.sqrt(te_kev)
    return power_density_mw_m3 * volume_m3


def temperature_profile_beta(
    radial_grid: jnp.ndarray,
    temperature_keV: jnp.ndarray,
    alpha_t: jnp.ndarray,
) -> jnp.ndarray:
    """Estimate the Albajar/Fidone temperature beta shape from the profile."""
    te0 = jnp.maximum(temperature_keV[0], 1.0e-9)
    te_edge = jnp.maximum(temperature_keV[-1], 0.0)
    te_mid = jnp.interp(0.5, radial_grid, temperature_keV)
    normalized_mid = jnp.clip(
        (te_mid - te_edge) / jnp.maximum(te0 - te_edge, 1.0e-9),
        1.0e-6,
        1.0 - 1.0e-6,
    )
    profile_inner = jnp.clip(
        1.0 - normalized_mid ** (1.0 / jnp.maximum(alpha_t, 1.0e-6)),
        1.0e-6,
        1.0 - 1.0e-6,
    )
    beta_t = jnp.log(profile_inner) / jnp.log(0.5)
    return jnp.clip(beta_t, 1.0e-3, 8.0)


def albajar_fidone_synchrotron_radiation_mw(
    electron_density_m3: jnp.ndarray,
    temperature_keV: jnp.ndarray,
    radial_grid: jnp.ndarray,
    magnetic_field_t: jnp.ndarray,
    major_radius_m: jnp.ndarray,
    minor_radius_m: jnp.ndarray,
    elongation: jnp.ndarray,
    aspect_ratio: jnp.ndarray,
    weights: jnp.ndarray,
    config: ReactorConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate the Albajar/Fidone synchrotron radiation fit and diagnostics."""
    ne0_20 = jnp.maximum(electron_density_m3[0] / 1.0e20, 1.0e-12)
    te0 = jnp.maximum(temperature_keV[0], 1.0e-6)
    ne_avg = profile_average(electron_density_m3, weights)
    te_avg = profile_average(temperature_keV, weights)
    alpha_n = effective_profile_alpha(electron_density_m3[0], ne_avg, upper=2.0)
    alpha_t = effective_profile_alpha(te0, te_avg, upper=8.0)
    beta_t = temperature_profile_beta(radial_grid, temperature_keV, alpha_t)
    reflectivity = jnp.clip(config.synchrotron_wall_reflectivity, 0.0, 0.999)

    p_a0 = 6.04e3 * minor_radius_m * ne0_20 / jnp.maximum(magnetic_field_t, 1.0e-9)
    g_function = 0.93 * (1.0 + 0.85 * jnp.exp(-0.82 * aspect_ratio))
    k_function = (
        (alpha_n + 3.87 * alpha_t + 1.46) ** -0.79
        * (1.98 + alpha_t) ** 1.36
        * beta_t**2.14
        * (beta_t**1.53 + 1.87 * alpha_t - 0.16) ** -1.33
    )
    reflect_term = (1.0 - reflectivity) ** 0.41
    trapping = (
        1.0 + 0.12 * (te0 / jnp.maximum(p_a0, 1.0e-12) ** 0.41) * reflect_term
    ) ** -1.51

    net_mw = (
        3.84e-8
        * (1.0 - reflectivity) ** 0.62
        * major_radius_m
        * minor_radius_m**1.38
        * elongation**0.79
        * magnetic_field_t**2.62
        * ne0_20**0.38
        * te0
        * (16.0 + te0) ** 2.61
        * trapping
        * g_function
        * k_function
    )

    no_reflect_trapping = (
        1.0 + 0.12 * (te0 / jnp.maximum(p_a0, 1.0e-12) ** 0.41)
    ) ** -1.51
    no_reflect_mw = (
        3.84e-8
        * major_radius_m
        * minor_radius_m**1.38
        * elongation**0.79
        * magnetic_field_t**2.62
        * ne0_20**0.38
        * te0
        * (16.0 + te0) ** 2.61
        * no_reflect_trapping
        * g_function
        * k_function
    )
    reflectivity_factor = net_mw / jnp.maximum(no_reflect_mw, 1.0e-30)
    return net_mw, no_reflect_mw, p_a0, reflectivity_factor, beta_t
