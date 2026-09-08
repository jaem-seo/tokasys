"""Radial profile, pedestal, and profile-integration helpers.

Refs:
    Greenwald et al., Nucl. Fusion 28, 2199-2207 (1988), for the density-limit
    normalization used upstream of these profile shapes.
    The alpha-critical pedestal closure follows the TokaGrad-style reduced
    pedestal model. PROCESS-style parabolic/pedestal profiles are provided for
    validation against PROCESS reference cases.

Notes:
    The q-profile/shear helper is diagnostic only; it is not a Grad-Shafranov
    equilibrium solve. Treat the alpha-critical pedestal as a reduced model.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.constants import KEV_TO_J, MU0
from tokasys.core.types import (
    DesignVariables,
    GeometryState,
    ReactorConfig,
    TechnologyParameters,
)


def profile_average(values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Return a volume-weighted average over the fixed radial grid."""
    return jnp.sum(values * weights)


def stored_energy_from_temperature_profile_mj(
    weights: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    electron_temperature_keV: jnp.ndarray,
    tech: TechnologyParameters,
    volume_m3: jnp.ndarray,
) -> jnp.ndarray:
    """Integrate electron and ion thermal energy over the plasma volume [MJ]."""
    density_factor = electron_density_m3 * (
        1.0 + tech.fuel_ion_fraction * tech.ion_to_electron_temperature_ratio
    )
    return (
        1.5
        * KEV_TO_J
        * volume_m3
        / 1.0e6
        * profile_average(density_factor * electron_temperature_keV, weights)
    )


def _stored_energy_scaled_temperature_profile(
    weights: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    base_temperature_keV: jnp.ndarray,
    core_shape: jnp.ndarray,
    target_stored_energy_mj: jnp.ndarray,
    tech: TechnologyParameters,
    volume_m3: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scale a core temperature shape to match a requested stored energy."""
    density_factor = electron_density_m3 * (
        1.0 + tech.fuel_ion_fraction * tech.ion_to_electron_temperature_ratio
    )
    stored_per_kev_mj = (
        1.5
        * KEV_TO_J
        * volume_m3
        / 1.0e6
        * profile_average(density_factor * core_shape, weights)
    )
    base_stored_mj = (
        1.5
        * KEV_TO_J
        * volume_m3
        / 1.0e6
        * profile_average(density_factor * base_temperature_keV, weights)
    )
    core_scale = jnp.maximum(
        (target_stored_energy_mj - base_stored_mj)
        / jnp.maximum(stored_per_kev_mj, 1.0e-12),
        0.0,
    )
    return base_temperature_keV + core_scale * core_shape, core_scale


def radial_derivative(values: jnp.ndarray, rho: jnp.ndarray) -> jnp.ndarray:
    """Second-order interior radial derivative on the fixed profile grid."""
    left = (values[1] - values[0]) / (rho[1] - rho[0])
    center = (values[2:] - values[:-2]) / (rho[2:] - rho[:-2])
    right = (values[-1] - values[-2]) / (rho[-1] - rho[-2])
    return jnp.concatenate((left[None], center, right[None]))


def effective_profile_alpha(
    central_value: jnp.ndarray,
    average_value: jnp.ndarray,
    lower: float = 1.0e-3,
    upper: float = 8.0,
) -> jnp.ndarray:
    """Map an arbitrary profile to a parabolic-profile peaking exponent."""
    alpha = central_value / jnp.maximum(average_value, 1.0e-30) - 1.0
    return jnp.clip(alpha, lower, upper)


def diagnostic_q_profile_and_shear(
    rho: jnp.ndarray,
    q95: jnp.ndarray,
    current_profile_alpha: float = 2.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Wesson-like diagnostic q profile and magnetic shear estimates.

    This is not an equilibrium solve; it supplies a smooth q-profile diagnostic
    for reduced pedestal scaling when only scalar Ip/q95 are available.
    """
    current_shape = jnp.maximum(1.0 - rho**2, 0.0) ** current_profile_alpha
    current_kernel = current_shape * rho
    shell_integrals = 0.5 * (current_kernel[1:] + current_kernel[:-1]) * (
        rho[1:] - rho[:-1]
    )
    enclosed = jnp.concatenate((jnp.asarray([0.0]), jnp.cumsum(shell_integrals)))
    enclosed_fraction = enclosed / jnp.maximum(enclosed[-1], 1.0e-30)
    raw_q = jnp.maximum(rho**2, 1.0e-8) / jnp.maximum(enclosed_fraction, 1.0e-8)
    raw_q = raw_q.at[0].set(raw_q[1])
    q_at_95 = jnp.interp(0.95, rho, raw_q)
    q_profile = raw_q * q95 / jnp.maximum(q_at_95, 1.0e-30)
    dq_drho = radial_derivative(q_profile, rho)
    shear_profile = rho * dq_drho / jnp.maximum(q_profile, 1.0e-30)
    shear_edge = jnp.maximum(shear_profile[-1], 0.0)
    shear_95 = jnp.maximum(jnp.interp(0.95, rho, shear_profile), 0.0)
    return q_profile, shear_profile, shear_edge, shear_95


def alpha_critical_pedestal(
    x: DesignVariables,
    geom: GeometryState,
    tech: TechnologyParameters,
    config: ReactorConfig,
    q95: jnp.ndarray,
    magnetic_shear: jnp.ndarray,
    separatrix_density_m3: jnp.ndarray,
    pedestal_top_density_m3: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduced alpha-critical pedestal closure from TokaGrad Eqs. 13-14."""
    alpha_crit = 0.4 * config.alpha_critical_shear * magnetic_shear * (
        1.0 + x.elongation**2 * (1.0 + 5.0 * x.triangularity**2)
    )
    grad_crit_pa = (
        alpha_crit
        * geom.minor_radius_m
        * x.toroidal_field_t**2
        / (2.0 * MU0 * x.major_radius_m * q95**2)
    )

    te_sep = config.separatrix_temperature_keV
    ti_sep = tech.ion_to_electron_temperature_ratio * te_sep
    ion_density_sep = tech.fuel_ion_fraction * separatrix_density_m3
    p_sep = (separatrix_density_m3 * te_sep + ion_density_sep * ti_sep) * KEV_TO_J

    pedestal_width = jnp.clip(config.pedestal_fixed_width, 1.0e-3, 0.2)
    pedestal_pressure = p_sep + grad_crit_pa * pedestal_width
    pedestal_temperature = pedestal_pressure / (
        KEV_TO_J
        * jnp.maximum(
            pedestal_top_density_m3
            * (1.0 + tech.fuel_ion_fraction * tech.ion_to_electron_temperature_ratio),
            1.0e-30,
        )
    )
    return pedestal_width, pedestal_pressure, pedestal_temperature


def pedestal_tanh_shape(
    rho: jnp.ndarray,
    pedestal_width: jnp.ndarray,
    edge_fraction: float,
    transition_scale_fraction: float = 0.5,
) -> jnp.ndarray:
    """Unit-core tanh shape with transition width tied to pedestal width."""
    width = jnp.maximum(pedestal_width, 1.0e-3)
    transition_scale = jnp.maximum(transition_scale_fraction * width, 1.0e-4)
    center = 1.0 - 0.5 * width
    return edge_fraction + 0.5 * (1.0 - edge_fraction) * (
        1.0 - jnp.tanh((rho - center) / transition_scale)
    )


def density_profile_m3(
    rho: jnp.ndarray,
    weights: jnp.ndarray,
    average_density_m3: jnp.ndarray,
    pedestal_width: jnp.ndarray,
    config: ReactorConfig,
) -> jnp.ndarray:
    """Pedestal-tied density shape rescaled to the Greenwald-fraction average."""
    pedestal_shape = pedestal_tanh_shape(
        rho,
        pedestal_width=pedestal_width,
        edge_fraction=config.density_edge_fraction,
        transition_scale_fraction=config.density_tanh_width,
    )
    rho_ped_top = jnp.maximum(1.0 - pedestal_width, 1.0e-3)
    pedestal_top_shape = pedestal_tanh_shape(
        rho_ped_top,
        pedestal_width=pedestal_width,
        edge_fraction=config.density_edge_fraction,
        transition_scale_fraction=config.density_tanh_width,
    )
    central_shape = pedestal_tanh_shape(
        0.0,
        pedestal_width=pedestal_width,
        edge_fraction=config.density_edge_fraction,
        transition_scale_fraction=config.density_tanh_width,
    )
    core_bump = jnp.maximum(1.0 - (rho / rho_ped_top) ** 1.5, 0.0) ** (
        config.density_profile_exponent
    )
    target_central_shape = (
        (1.0 + config.density_core_peaking_fraction) * pedestal_top_shape
    )
    bump_amplitude = jnp.maximum(target_central_shape - central_shape, 0.0)
    shape = pedestal_shape + bump_amplitude * core_bump
    return average_density_m3 * shape / profile_average(shape, weights)


def density_at_normalized_radius_m3(
    rho_value: jnp.ndarray,
    rho_grid: jnp.ndarray,
    weights: jnp.ndarray,
    average_density_m3: jnp.ndarray,
    pedestal_width: jnp.ndarray,
    config: ReactorConfig,
) -> jnp.ndarray:
    """Evaluate the normalized density profile at an arbitrary radial location."""
    rho_ped_top = jnp.maximum(1.0 - pedestal_width, 1.0e-3)
    pedestal_top_shape = pedestal_tanh_shape(
        rho_ped_top,
        pedestal_width=pedestal_width,
        edge_fraction=config.density_edge_fraction,
        transition_scale_fraction=config.density_tanh_width,
    )
    central_shape = pedestal_tanh_shape(
        0.0,
        pedestal_width=pedestal_width,
        edge_fraction=config.density_edge_fraction,
        transition_scale_fraction=config.density_tanh_width,
    )
    bump_amplitude = jnp.maximum(
        (1.0 + config.density_core_peaking_fraction) * pedestal_top_shape
        - central_shape,
        0.0,
    )

    def shaped_density(rho: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the unnormalized pedestal-plus-core density shape."""
        pedestal_shape = pedestal_tanh_shape(
            rho,
            pedestal_width=pedestal_width,
            edge_fraction=config.density_edge_fraction,
            transition_scale_fraction=config.density_tanh_width,
        )
        core_bump = jnp.maximum(1.0 - (rho / rho_ped_top) ** 1.5, 0.0) ** (
            config.density_profile_exponent
        )
        return pedestal_shape + bump_amplitude * core_bump

    shape_grid = shaped_density(rho_grid)
    shape_value = shaped_density(rho_value)
    return average_density_m3 * shape_value / profile_average(shape_grid, weights)


def process_parabolic_density_profile_m3(
    rho: jnp.ndarray,
    weights: jnp.ndarray,
    average_density_m3: jnp.ndarray,
    config: ReactorConfig,
) -> jnp.ndarray:
    """PROCESS i_plasma_pedestal=0 density profile rescaled to the target average."""
    shape = jnp.maximum(1.0 - rho**2, 0.0) ** config.process_profile_alphan
    return average_density_m3 * shape / jnp.maximum(
        profile_average(shape, weights),
        1.0e-30,
    )


def process_parabolic_temperature_profile_keV(
    rho: jnp.ndarray,
    weights: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    target_stored_energy_mj: jnp.ndarray,
    tech: TechnologyParameters,
    config: ReactorConfig,
    volume_m3: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """PROCESS i_plasma_pedestal=0 temperature shape with stored-energy scaling."""
    core_shape = jnp.maximum(1.0 - rho**2, 0.0) ** config.process_profile_alphat
    te_profile, core_scale = _stored_energy_scaled_temperature_profile(
        weights=weights,
        electron_density_m3=electron_density_m3,
        base_temperature_keV=jnp.zeros_like(rho),
        core_shape=core_shape,
        target_stored_energy_mj=target_stored_energy_mj,
        tech=tech,
        volume_m3=volume_m3,
    )
    return jnp.maximum(te_profile, 1.0e-8), jnp.zeros_like(rho), core_scale


def process_pedestal_density_profile_m3(
    rho: jnp.ndarray,
    weights: jnp.ndarray,
    average_density_m3: jnp.ndarray,
    greenwald_density_m3: jnp.ndarray,
    config: ReactorConfig,
) -> jnp.ndarray:
    """PROCESS i_plasma_pedestal=1 density profile matched to target average."""
    rho_ped = jnp.clip(config.process_density_pedestal_radius, 1.0e-3, 0.999)
    nped = (
        config.process_density_pedestal_greenwald_fraction * greenwald_density_m3
    )
    nsep = (
        config.process_density_separatrix_greenwald_fraction * greenwald_density_m3
    )
    core_shape = jnp.where(
        rho <= rho_ped,
        jnp.maximum(1.0 - (rho / rho_ped) ** 2, 0.0)
        ** config.process_profile_alphan,
        0.0,
    )
    pedestal_base = jnp.where(
        rho <= rho_ped,
        nped,
        nsep + (nped - nsep) * (1.0 - rho) / (1.0 - rho_ped),
    )
    core_average = jnp.maximum(profile_average(core_shape, weights), 1.0e-30)
    core_amplitude = (
        average_density_m3 - profile_average(pedestal_base, weights)
    ) / core_average
    profile = pedestal_base + core_amplitude * core_shape
    return jnp.maximum(profile, 1.0e-6)


def process_pedestal_temperature_profile_keV(
    rho: jnp.ndarray,
    weights: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    target_stored_energy_mj: jnp.ndarray,
    tech: TechnologyParameters,
    config: ReactorConfig,
    volume_m3: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """PROCESS i_plasma_pedestal=1 temperature shape with stored-energy scaling."""
    rho_ped = jnp.clip(config.process_temperature_pedestal_radius, 1.0e-3, 0.999)
    tped = config.process_temperature_pedestal_keV
    tsep = config.process_temperature_separatrix_keV
    core_shape = jnp.where(
        rho <= rho_ped,
        jnp.maximum(1.0 - (rho / rho_ped) ** config.process_profile_tbeta, 0.0)
        ** config.process_profile_alphat,
        0.0,
    )
    pedestal_base = jnp.where(
        rho <= rho_ped,
        tped,
        tsep + (tped - tsep) * (1.0 - rho) / (1.0 - rho_ped),
    )
    te_profile, core_scale = _stored_energy_scaled_temperature_profile(
        weights=weights,
        electron_density_m3=electron_density_m3,
        base_temperature_keV=pedestal_base,
        core_shape=core_shape,
        target_stored_energy_mj=target_stored_energy_mj,
        tech=tech,
        volume_m3=volume_m3,
    )
    return jnp.maximum(te_profile, 1.0e-8), pedestal_base, core_scale


def temperature_profile_keV(
    rho: jnp.ndarray,
    weights: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    target_stored_energy_mj: jnp.ndarray,
    pedestal_width: jnp.ndarray,
    pedestal_temperature_keV: jnp.ndarray,
    tech: TechnologyParameters,
    config: ReactorConfig,
    volume_m3: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pedestal plus core-shape temperature profile with stored-energy closure."""
    width = jnp.maximum(pedestal_width, 1.0e-3)
    rho_ped = 1.0 - 0.5 * width
    transition_scale = jnp.maximum(0.5 * width, 1.0e-4)
    te_sep = config.separatrix_temperature_keV
    pedestal_height = jnp.maximum(pedestal_temperature_keV - te_sep, 0.0)
    pedestal = te_sep + 0.5 * pedestal_height * (
        1.0 - jnp.tanh((rho - rho_ped) / transition_scale)
    )
    core_shape = (1.0 - rho**1.5) ** config.temperature_profile_exponent

    te_profile, core_scale = _stored_energy_scaled_temperature_profile(
        weights=weights,
        electron_density_m3=electron_density_m3,
        base_temperature_keV=pedestal,
        core_shape=core_shape,
        target_stored_energy_mj=target_stored_energy_mj,
        tech=tech,
        volume_m3=volume_m3,
    )
    return te_profile, pedestal, core_scale
