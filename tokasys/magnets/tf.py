"""Top-level TF magnet engineering evaluator.

Refs:
    PROCESS engineering practice: Kovari et al., Fusion Eng. Des. 104, 9-20
    (2016), and standard magnetic-pressure/stress scaling for conceptual
    superconducting tokamak magnets.

Notes:
    TF stress, centering force, winding-pack packing and cryogenic heat are
    reduced engineering surrogates suitable for differentiable optimization.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.constants import MU0
from tokasys.core.types import (
    DesignVariables,
    GeometryState,
    MagnetState,
    NuclearState,
    PlasmaState,
    ReactorConfig,
    TechnologyParameters,
)
from tokasys.magnets.cs import cs_pf_requirements
from tokasys.magnets.pf import pf_coil_placement_and_loads
from tokasys.magnets.stress import (
    plasma_neutron_leakage_proxy,
    tf_centering_force_per_coil_mn,
    tf_stress_pa,
    tf_structural_fractions,
)
from tokasys.magnets.superconductors import (
    superconductor_critical_current_density,
    winding_pack_current_density_limit,
)


def evaluate_magnets(
    x: DesignVariables,
    geom: GeometryState,
    plasma: PlasmaState,
    nuclear: NuclearState,
    tech: TechnologyParameters,
    config: ReactorConfig,
) -> MagnetState:
    r_tf = geom.inboard_tf_outer_radius_m
    safe_r_tf = jnp.maximum(r_tf, 0.05)
    peak_field = x.toroidal_field_t * x.major_radius_m / safe_r_tf

    total_ampere_turns = 2.0 * jnp.pi * x.major_radius_m * x.toroidal_field_t / MU0
    ampere_turns_per_coil = total_ampere_turns / tech.tf_n_coils
    toroidal_pitch_width = (
        2.0
        * jnp.pi
        * geom.inboard_tf_centroid_radius_m
        / jnp.maximum(tech.tf_n_coils, 1.0)
    )
    usable_toroidal_width = (
        jnp.clip(tech.tf_coil_toroidal_pack_fraction, 0.05, 0.95)
        * toroidal_pitch_width
    )
    total_coil_area_per_coil = x.tf_coil_thickness_m * usable_toroidal_width

    structural_fraction, winding_pack_fraction = tf_structural_fractions(tech)
    structural_area = structural_fraction * total_coil_area_per_coil
    winding_pack_area = winding_pack_fraction * total_coil_area_per_coil
    superconductor_area = (
        jnp.clip(tech.tf_winding_pack_superconductor_fraction, 0.02, 0.90)
        * winding_pack_area
    )

    engineering_j = ampere_turns_per_coil / jnp.maximum(winding_pack_area, 1.0e-9)
    superconductor_j = ampere_turns_per_coil / jnp.maximum(
        superconductor_area,
        1.0e-9,
    )
    (
        allowable_engineering_j,
        critical_engineering_j,
        superconductor_temperature,
        superconductor_strain,
        field_derating,
        temperature_derating,
        strain_derating,
    ) = winding_pack_current_density_limit(peak_field, tech, config)

    magnetic_pressure = peak_field**2 / (2.0 * MU0)
    stress = tf_stress_pa(
        peak_field_t=peak_field,
        x=x,
        safe_tf_radius_m=safe_r_tf,
        structural_fraction=structural_fraction,
        winding_pack_fraction=winding_pack_fraction,
    )
    centering_force_per_coil_mn = tf_centering_force_per_coil_mn(
        peak_field,
        x,
        geom,
        tech,
    )

    field_volume = 2.0 * jnp.pi**2 * x.major_radius_m * geom.minor_radius_m**2
    stored_energy_gj = (
        x.toroidal_field_t**2 / (2.0 * MU0) * field_volume / 1.0e9
    )

    tf_nuclear_heat = plasma_neutron_leakage_proxy(nuclear, tech)
    cryogenic_static_heat = jnp.asarray(tech.static_cryo_load_mw, dtype=float)
    cryogenic_nuclear_heat = tech.nuclear_cryo_multiplier * tf_nuclear_heat
    cryogenic_current_lead_joint_heat = (
        tech.current_lead_cryo_load_mw_per_ma_turn * total_ampere_turns / 1.0e6
        + tech.joint_cryo_load_mw_per_gj * stored_energy_gj
    )
    cryogenic_static_electric = cryogenic_static_heat / tech.cryogenic_cop
    cryogenic_nuclear_electric = cryogenic_nuclear_heat / tech.cryogenic_cop
    cryogenic_current_lead_joint_electric = (
        cryogenic_current_lead_joint_heat / tech.cryogenic_cop
    )
    cryogenic_electric = (
        cryogenic_static_electric
        + cryogenic_nuclear_electric
        + cryogenic_current_lead_joint_electric
    )

    (
        pf_vertical_field,
        pf_required_ampere_turns,
        cs_required_ampere_turns,
        cs_available_ampere_turns,
        cs_startup_flux_required,
        cs_flat_top_flux_required,
        cs_flux_required,
        cs_flux_capacity,
        cs_peak_field,
        cs_current_density,
        cs_current_density_limit,
        cs_critical_current_density,
        cs_superconductor_temperature,
        cs_superconductor_strain,
        cs_field_derating,
        cs_temperature_derating,
        cs_strain_derating,
        cs_current_density_utilization,
        cs_stress,
        _loop_voltage,
    ) = cs_pf_requirements(
        x=x,
        geom=geom,
        plasma=plasma,
        tech=tech,
        config=config,
    )
    (
        pf_coil_radii,
        pf_coil_z,
        pf_coil_ampere_turns,
        pf_coil_current_density,
        pf_coil_peak_field,
        pf_coil_stress,
        pf_max_current_density,
        pf_current_density_limit,
        pf_critical_current_density,
        pf_superconductor_temperature,
        pf_superconductor_strain,
        pf_field_derating,
        pf_temperature_derating,
        pf_strain_derating,
        pf_current_density_utilization,
        pf_max_peak_field,
        pf_max_stress,
    ) = pf_coil_placement_and_loads(
        x=x,
        geom=geom,
        vertical_field_t=pf_vertical_field,
        required_ampere_turns_a=pf_required_ampere_turns,
        tech=tech,
        config=config,
    )

    return MagnetState(
        peak_tf_field_t=peak_field,
        tf_required_ampere_turns_a=total_ampere_turns,
        tf_engineering_current_density_a_m2=engineering_j,
        tf_allowable_engineering_current_density_a_m2=allowable_engineering_j,
        tf_winding_pack_area_m2=winding_pack_area,
        tf_structural_area_m2=structural_area,
        tf_superconductor_area_m2=superconductor_area,
        tf_superconductor_current_density_a_m2=superconductor_j,
        tf_superconductor_critical_current_density_a_m2=critical_engineering_j,
        tf_superconductor_temperature_k=superconductor_temperature,
        tf_superconductor_strain=superconductor_strain,
        tf_field_derating=field_derating,
        tf_temperature_derating=temperature_derating,
        tf_strain_derating=strain_derating,
        tf_structural_fraction=structural_fraction,
        tf_winding_pack_fraction=winding_pack_fraction,
        tf_magnetic_pressure_pa=magnetic_pressure,
        tf_centering_force_per_coil_mn=centering_force_per_coil_mn,
        tf_stress_pa=stress,
        tf_stored_energy_gj=stored_energy_gj,
        tf_nuclear_heat_mw=tf_nuclear_heat,
        cryogenic_static_heat_mw=cryogenic_static_heat,
        cryogenic_nuclear_heat_mw=cryogenic_nuclear_heat,
        cryogenic_current_lead_joint_heat_mw=cryogenic_current_lead_joint_heat,
        cryogenic_static_electric_power_mw=cryogenic_static_electric,
        cryogenic_nuclear_electric_power_mw=cryogenic_nuclear_electric,
        cryogenic_current_lead_joint_electric_power_mw=(
            cryogenic_current_lead_joint_electric
        ),
        cryogenic_electric_power_mw=cryogenic_electric,
        pf_vertical_field_t=pf_vertical_field,
        pf_required_equilibrium_ampere_turns_a=pf_required_ampere_turns,
        pf_coil_radii_m=pf_coil_radii,
        pf_coil_z_m=pf_coil_z,
        pf_coil_ampere_turns_a=pf_coil_ampere_turns,
        pf_coil_current_density_a_m2=pf_coil_current_density,
        pf_coil_peak_field_t=pf_coil_peak_field,
        pf_coil_stress_pa=pf_coil_stress,
        pf_max_current_density_a_m2=pf_max_current_density,
        pf_current_density_limit_a_m2=pf_current_density_limit,
        pf_superconductor_critical_current_density_a_m2=pf_critical_current_density,
        pf_superconductor_temperature_k=pf_superconductor_temperature,
        pf_superconductor_strain=pf_superconductor_strain,
        pf_field_derating=pf_field_derating,
        pf_temperature_derating=pf_temperature_derating,
        pf_strain_derating=pf_strain_derating,
        pf_current_density_utilization=pf_current_density_utilization,
        pf_max_peak_field_t=pf_max_peak_field,
        pf_max_stress_pa=pf_max_stress,
        cs_required_ampere_turns_a=cs_required_ampere_turns,
        cs_available_ampere_turns_a=cs_available_ampere_turns,
        cs_startup_flux_required_wb=cs_startup_flux_required,
        cs_flat_top_flux_required_wb=cs_flat_top_flux_required,
        cs_flux_required_wb=cs_flux_required,
        cs_flux_capacity_wb=cs_flux_capacity,
        cs_peak_field_t=cs_peak_field,
        cs_current_density_a_m2=cs_current_density,
        cs_current_density_limit_a_m2=cs_current_density_limit,
        cs_superconductor_critical_current_density_a_m2=cs_critical_current_density,
        cs_superconductor_temperature_k=cs_superconductor_temperature,
        cs_superconductor_strain=cs_superconductor_strain,
        cs_field_derating=cs_field_derating,
        cs_temperature_derating=cs_temperature_derating,
        cs_strain_derating=cs_strain_derating,
        cs_current_density_utilization=cs_current_density_utilization,
        cs_stress_pa=cs_stress,
    )


__all__ = [
    "cs_pf_requirements",
    "evaluate_magnets",
    "plasma_neutron_leakage_proxy",
    "superconductor_critical_current_density",
    "winding_pack_current_density_limit",
]
