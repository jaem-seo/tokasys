"""Magnet stress and nuclear-heating helper models.

Refs:
    PROCESS engineering model papers and standard magnetic-pressure
    engineering estimates for conceptual TF-coil design.

Notes:
    These functions are intentional 0D stress/fluence surrogates; detailed
    structural FEA and 3D neutron transport are outside the current model.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.constants import MU0
from tokasys.core.types import DesignVariables, GeometryState, NuclearState, TechnologyParameters


def tf_structural_fractions(
    tech: TechnologyParameters,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    structural_fraction = jnp.clip(tech.tf_structural_factor, 0.05, 0.90)
    return structural_fraction, 1.0 - structural_fraction


def tf_stress_pa(
    peak_field_t: jnp.ndarray,
    x: DesignVariables,
    safe_tf_radius_m: jnp.ndarray,
    structural_fraction: jnp.ndarray,
    winding_pack_fraction: jnp.ndarray,
) -> jnp.ndarray:
    magnetic_pressure = peak_field_t**2 / (2.0 * MU0)
    lever_arm = x.major_radius_m - safe_tf_radius_m
    load_sharing = winding_pack_fraction / jnp.maximum(structural_fraction, 1.0e-6)
    return (
        magnetic_pressure
        * lever_arm
        / jnp.maximum(x.tf_coil_thickness_m, 1.0e-6)
        * load_sharing
    )


def tf_centering_force_per_coil_mn(
    peak_field_t: jnp.ndarray,
    x: DesignVariables,
    geom: GeometryState,
    tech: TechnologyParameters,
) -> jnp.ndarray:
    magnetic_pressure = peak_field_t**2 / (2.0 * MU0)
    return (
        magnetic_pressure
        * geom.tf_coil_height_m
        * x.tf_coil_thickness_m
        / jnp.maximum(tech.tf_n_coils, 1.0)
        / 1.0e6
    )


def plasma_neutron_leakage_proxy(
    nuclear: NuclearState,
    tech: TechnologyParameters,
) -> jnp.ndarray:
    # A small configurable fraction of unattenuated neutron power is assumed
    # geometrically incident on cold structures before blanket/shield attenuation.
    unattenuated_reference_mw = (
        nuclear.blanket_thermal_power_mw
        / jnp.maximum(
            nuclear.blanket_coverage_fraction * nuclear.blanket_energy_multiplication,
            1.0e-9,
        )
    )
    return (
        tech.tf_neutron_geometric_view_factor
        * unattenuated_reference_mw
        * nuclear.neutron_attenuation
    )
