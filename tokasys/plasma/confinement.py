"""Energy confinement and transport-loss closures.

Refs:
    ITER Physics Basis, Chapter 2, Nucl. Fusion 39, 2175-2249 (1999),
    for the IPB98(y,2) H-mode energy-confinement scaling.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import DesignVariables, GeometryState, TechnologyParameters


def ipb98y2_confinement_coefficient(
    electron_density_m3: jnp.ndarray,
    x: DesignVariables,
    geom: GeometryState,
    tech: TechnologyParameters,
) -> jnp.ndarray:
    """IPB98(y,2) coefficient excluding the power-loss dependence."""
    n19 = electron_density_m3 / 1.0e19
    return (
        0.0562
        * x.confinement_h98
        * x.plasma_current_ma**0.93
        * x.toroidal_field_t**0.15
        * n19**0.41
        * x.major_radius_m**1.97
        * x.elongation**0.78
        * geom.inverse_aspect_ratio**0.58
        * tech.effective_ion_mass_amu**0.19
    )


def ipb98y2_confinement_time_s(
    transport_loss_mw: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    x: DesignVariables,
    geom: GeometryState,
    tech: TechnologyParameters,
) -> jnp.ndarray:
    """IPB98(y,2) confinement time for a given transport power loss."""
    c_tau = ipb98y2_confinement_coefficient(
        electron_density_m3=electron_density_m3,
        x=x,
        geom=geom,
        tech=tech,
    )
    return c_tau * jnp.maximum(transport_loss_mw, 1.0e-9) ** (-0.69)


def ipb98y2_transport_loss_mw(
    stored_energy_mj: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    x: DesignVariables,
    geom: GeometryState,
    tech: TechnologyParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """IPB98(y,2) confinement with algebraic elimination of P_loss."""
    c_tau = ipb98y2_confinement_coefficient(
        electron_density_m3=electron_density_m3,
        x=x,
        geom=geom,
        tech=tech,
    )
    transport_loss_mw = (stored_energy_mj / jnp.maximum(c_tau, 1.0e-9)) ** (
        1.0 / 0.31
    )
    confinement_time_s = stored_energy_mj / jnp.maximum(transport_loss_mw, 1.0e-9)
    return transport_loss_mw, confinement_time_s
