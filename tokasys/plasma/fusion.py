"""Fusion reactivity and fusion power-density closures.

Refs:
    Bosch and Hale, Nucl. Fusion 32, 611-631 (1992), for the Maxwellian
    D-T reactivity parameterization.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.constants import DT_FUSION_ENERGY_J


def dt_reactivity_m3_s(temperature_keV: jnp.ndarray) -> jnp.ndarray:
    """Maxwellian D-T reactivity from the Bosch-Hale parameterization."""
    t = jnp.maximum(temperature_keV, 1.0e-3)
    c1 = 1.17302e-9
    c2 = 1.51361e-2
    c3 = 7.51886e-2
    c4 = 4.60643e-3
    c5 = 1.35000e-2
    c6 = -1.06750e-4
    c7 = 1.36600e-5
    bg = 34.3827
    reduced_mass_energy_keV = 1124656.0

    numerator = t * (c2 + t * (c4 + t * c6))
    denominator = 1.0 + t * (c3 + t * (c5 + t * c7))
    theta = t / (1.0 - numerator / denominator)
    xi = (bg**2 / (4.0 * theta)) ** (1.0 / 3.0)
    reactivity_cm3_s = (
        c1
        * theta
        * jnp.sqrt(xi / (reduced_mass_energy_keV * t**3))
        * jnp.exp(-3.0 * xi)
    )
    return 1.0e-6 * reactivity_cm3_s


def dt_fusion_power_density_mw_m3(
    electron_density_m3: jnp.ndarray,
    ion_temperature_keV: jnp.ndarray,
    fuel_ion_fraction: float,
    profile_fusion_enhancement: float,
) -> jnp.ndarray:
    """D-T fusion power density in MW/m3 for a 50:50 D-T ion mixture."""
    reactivity_22 = dt_reactivity_m3_s(ion_temperature_keV) / 1.0e-22
    return (
        profile_fusion_enhancement
        * 0.25
        * fuel_ion_fraction**2
        * (electron_density_m3 / 1.0e20) ** 2
        * reactivity_22
        * (1.0e40 * 1.0e-22 * DT_FUSION_ENERGY_J / 1.0e6)
    )
