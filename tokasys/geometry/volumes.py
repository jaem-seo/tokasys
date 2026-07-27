"""Plasma volume, surface, and radial integration weights.

Refs:
    Freidberg, Plasma Physics and Fusion Energy (2007), for standard tokamak
    volume/beta engineering relations. The inboard/outboard surface split uses
    a differentiable Miller-like boundary quadrature for system-code geometry.

Notes:
    Triangularity is included in surface-area splitting; plasma volume remains
    the standard elongated cross-section approximation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class PlasmaVolumeGeometry(NamedTuple):
    cross_section_m2: jnp.ndarray
    volume_m3: jnp.ndarray
    surface_area_m2: jnp.ndarray
    first_wall_area_m2: jnp.ndarray
    inboard_first_wall_area_m2: jnp.ndarray
    outboard_first_wall_area_m2: jnp.ndarray


def shaped_plasma_volumes(
    major_radius_m: jnp.ndarray,
    minor_radius_m: jnp.ndarray,
    elongation: jnp.ndarray,
    triangularity: jnp.ndarray = jnp.asarray(0.0),
) -> PlasmaVolumeGeometry:
    """Reduced self-similar plasma volume and inboard/outboard surface geometry."""
    cross_section = jnp.pi * minor_radius_m**2 * elongation
    volume = 2.0 * jnp.pi * major_radius_m * cross_section

    num_theta = 256
    theta = 2.0 * jnp.pi * (jnp.arange(num_theta, dtype=float) + 0.5) / num_theta
    shaped_angle = theta + triangularity * jnp.sin(theta)
    radius = major_radius_m + minor_radius_m * jnp.cos(shaped_angle)
    d_radius_dtheta = (
        -minor_radius_m
        * jnp.sin(shaped_angle)
        * (1.0 + triangularity * jnp.cos(theta))
    )
    d_z_dtheta = elongation * minor_radius_m * jnp.cos(theta)
    ds = jnp.sqrt(d_radius_dtheta**2 + d_z_dtheta**2)
    dtheta = 2.0 * jnp.pi / num_theta
    surface_elements = 2.0 * jnp.pi * jnp.maximum(radius, 0.05) * ds * dtheta
    inboard_surface = jnp.sum(
        jnp.where(radius <= major_radius_m, surface_elements, 0.0)
    )
    outboard_surface = jnp.sum(
        jnp.where(radius > major_radius_m, surface_elements, 0.0)
    )
    surface = inboard_surface + outboard_surface
    first_wall_area = 1.10 * surface
    return PlasmaVolumeGeometry(
        cross_section,
        volume,
        surface,
        first_wall_area,
        1.10 * inboard_surface,
        1.10 * outboard_surface,
    )


def profile_volume_elements(
    major_radius_m: jnp.ndarray,
    minor_radius_m: jnp.ndarray,
    elongation: jnp.ndarray,
    num_points: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Midpoint rho grid and shaped, self-similar volume elements."""
    rho = (jnp.arange(num_points, dtype=float) + 0.5) / num_points
    drho = 1.0 / num_points
    volume_prime = (
        4.0 * jnp.pi**2 * major_radius_m * elongation * minor_radius_m**2 * rho
    )
    shell_volumes = volume_prime * drho
    weights = shell_volumes / jnp.sum(shell_volumes)
    return rho, volume_prime, shell_volumes, weights
