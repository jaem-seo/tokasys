"""Top-level plasma-state evaluator.

Refs:
    Greenwald et al. (1988) for n_G, Troyon et al., Plasma Phys. Control.
    Fusion 26, 209-215 (1984), for normalized beta limits, and the module-level
    references in fusion/confinement/radiation/current for the coupled closures.

Notes:
    Fast-ion pressure combines an IPDG89/PROCESS-like alpha-beta option with a
    slowing-down stored-energy estimate for external fast ions; this is still a
    reduced 0D closure, not a Fokker-Planck fast-particle calculation.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.constants import (
    ALPHA_FRACTION,
    ATOMIC_MASS_UNIT_KG,
    E_CHARGE,
    ELECTRON_MASS_KG,
    EPSILON_0,
    KEV_TO_J,
    MEV_TO_J,
    MU0,
    NEUTRON_FRACTION,
)
from tokasys.core.types import (
    ClosureVariables,
    DesignVariables,
    GeometryState,
    PlasmaState,
    ReactorConfig,
    TechnologyParameters,
)
from tokasys.plasma.confinement import (
    ipb98y2_confinement_time_s,
    ipb98y2_transport_loss_mw,
)
from tokasys.plasma.current import (
    bootstrap_current_model,
    current_density_profiles,
    current_drive_model,
    ohmic_power_density_mw_m3,
    spitzer_resistivity_ohm_m,
)
from tokasys.plasma.fusion import dt_fusion_power_density_mw_m3, dt_reactivity_m3_s
from tokasys.plasma.profiles import (
    alpha_critical_pedestal,
    density_profile_m3,
    density_at_normalized_radius_m3,
    diagnostic_q_profile_and_shear,
    effective_profile_alpha,
    profile_average,
    process_parabolic_density_profile_m3,
    process_parabolic_temperature_profile_keV,
    process_pedestal_density_profile_m3,
    process_pedestal_temperature_profile_keV,
    radial_derivative,
    temperature_profile_keV,
)
from tokasys.plasma.radiation import (
    albajar_fidone_synchrotron_radiation_mw,
    electron_ion_bremsstrahlung_mw,
    temperature_profile_beta,
)


def fast_ion_slowing_down_time_s(
    electron_density_m3: jnp.ndarray,
    electron_temperature_keV: jnp.ndarray,
    initial_energy_mev: float,
    fast_ion_mass_kg: float,
    fast_ion_charge_number: float,
    coulomb_log: float,
) -> jnp.ndarray:
    """Electron-drag slowing-down time for a suprathermal ion."""
    thermal_energy_j = jnp.maximum(electron_temperature_keV, 1.0e-6) * KEV_TO_J
    numerator = (
        3.0
        * (2.0 * jnp.pi) ** 1.5
        * EPSILON_0**2
        * fast_ion_mass_kg
        * thermal_energy_j**1.5
        / jnp.sqrt(ELECTRON_MASS_KG)
    )
    denominator = (
        jnp.maximum(electron_density_m3, 1.0e-30)
        * fast_ion_charge_number**2
        * E_CHARGE**4
        * coulomb_log
    )
    electron_drag_time = numerator / denominator
    critical_energy_j = (
        14.8 * jnp.maximum(electron_temperature_keV, 1.0e-6) * KEV_TO_J
    )
    initial_energy_j = jnp.maximum(initial_energy_mev, 1.0e-9) * MEV_TO_J
    slowing_factor = jnp.log1p((initial_energy_j / critical_energy_j) ** 1.5) / 3.0
    return electron_drag_time * slowing_factor


def fast_ion_pressure_pa(
    power_mw: jnp.ndarray,
    slowing_down_time_s: jnp.ndarray,
    volume_m3: jnp.ndarray,
) -> jnp.ndarray:
    """Isotropic fast-ion pressure from stored fast-particle energy."""
    energy_j = power_mw * 1.0e6 * slowing_down_time_s
    return (2.0 / 3.0) * energy_j / jnp.maximum(volume_m3, 1.0e-30)


def ipdg89_fast_alpha_beta(
    thermal_beta: jnp.ndarray,
    electron_density_m3: jnp.ndarray,
    fuel_ion_density_m3: jnp.ndarray,
    electron_temperature_keV: jnp.ndarray,
    ion_temperature_keV: jnp.ndarray,
    alpha_power_total_mw: jnp.ndarray,
    alpha_power_plasma_mw: jnp.ndarray,
) -> jnp.ndarray:
    """IPDG89-like fast-alpha beta fraction used by PROCESS."""
    fuel_fraction = fuel_ion_density_m3 / jnp.maximum(electron_density_m3, 1.0e-30)
    temperature_factor = (electron_temperature_keV + ion_temperature_keV) / 20.0 - 0.37
    alpha_fraction = 0.29 * fuel_fraction**2 * temperature_factor
    alpha_fraction = jnp.clip(alpha_fraction, 0.0, 0.30)
    total_to_plasma_alpha = alpha_power_total_mw / jnp.maximum(
        alpha_power_plasma_mw,
        1.0e-30,
    )
    return thermal_beta * alpha_fraction * total_to_plasma_alpha


def evaluate_plasma(
    x: DesignVariables,
    geom: GeometryState,
    tech: TechnologyParameters,
    config: ReactorConfig,
    closure: ClosureVariables | None = None,
) -> PlasmaState:
    """Evaluate profiles, fusion, radiation, current, and energy closure.

    Density is fixed from the Greenwald fraction, while temperature and stored
    energy are reconciled using the configured full-space or self-consistent
    closure. The returned state retains scalar totals and radial profiles.
    """
    a = geom.minor_radius_m
    r0 = x.major_radius_m
    ip_ma = x.plasma_current_ma
    b0 = x.toroidal_field_t
    te = x.target_temperature_keV
    ti = tech.ion_to_electron_temperature_ratio * te

    n_gw_20 = ip_ma / (jnp.pi * a**2)
    n_gw = n_gw_20 * 1.0e20
    ne = x.greenwald_fraction * n_gw
    ni = tech.fuel_ion_fraction * ne

    rho = geom.radial_grid
    weights = geom.volume_weights

    target_thermal_pressure = (ne * te + ni * ti) * KEV_TO_J
    initial_target_stored_energy_mj = (
        1.5 * target_thermal_pressure * geom.plasma_volume_m3 / 1.0e6
    )

    shape_factor = 0.5 * (1.0 + x.elongation**2 * (1.0 + 2.0 * x.triangularity**2 - 1.2 * x.triangularity**3))
    
    q95 = 5.0 * a**2 * b0 * shape_factor / (r0 * ip_ma)
    _, _, magnetic_shear_edge, magnetic_shear_95 = diagnostic_q_profile_and_shear(
        rho,
        q95,
    )
    alpha_critical_shear = jnp.maximum(magnetic_shear_edge, magnetic_shear_95)

    profile_model = config.plasma_profile_model.lower()
    if profile_model == "tokasys":
        pedestal_width, pedestal_pressure, pedestal_temperature = (
            alpha_critical_pedestal(
                x=x,
                geom=geom,
                tech=tech,
                config=config,
                q95=q95,
                magnetic_shear=alpha_critical_shear,
                separatrix_density_m3=ne * config.density_edge_fraction,
                pedestal_top_density_m3=ne,
            )
        )
        for _ in range(2):
            separatrix_density = density_at_normalized_radius_m3(
                1.0,
                rho,
                weights,
                ne,
                pedestal_width,
                config,
            )
            pedestal_top_density = density_at_normalized_radius_m3(
                1.0 - pedestal_width,
                rho,
                weights,
                ne,
                pedestal_width,
                config,
            )
            pedestal_width, pedestal_pressure, pedestal_temperature = (
                alpha_critical_pedestal(
                    x=x,
                    geom=geom,
                    tech=tech,
                    config=config,
                    q95=q95,
                    magnetic_shear=alpha_critical_shear,
                    separatrix_density_m3=separatrix_density,
                    pedestal_top_density_m3=pedestal_top_density,
                )
            )
        ne_profile = density_profile_m3(rho, weights, ne, pedestal_width, config)
    elif profile_model == "process_parabolic":
        pedestal_width = jnp.asarray(0.0)
        pedestal_pressure = jnp.asarray(0.0)
        pedestal_temperature = jnp.asarray(0.0)
        ne_profile = process_parabolic_density_profile_m3(
            rho=rho,
            weights=weights,
            average_density_m3=ne,
            config=config,
        )
    elif profile_model == "process_pedestal":
        pedestal_width = 1.0 - jnp.clip(
            config.process_temperature_pedestal_radius,
            1.0e-3,
            0.999,
        )
        pedestal_temperature = jnp.asarray(config.process_temperature_pedestal_keV)
        pedestal_density = (
            config.process_density_pedestal_greenwald_fraction * n_gw
        )
        pedestal_ion_density = tech.fuel_ion_fraction * pedestal_density
        pedestal_pressure = (
            pedestal_density * pedestal_temperature
            + pedestal_ion_density
            * tech.ion_to_electron_temperature_ratio
            * pedestal_temperature
        ) * KEV_TO_J
        ne_profile = process_pedestal_density_profile_m3(
            rho=rho,
            weights=weights,
            average_density_m3=ne,
            greenwald_density_m3=n_gw,
            config=config,
        )
    else:
        raise ValueError(
            f"Unknown plasma_profile_model={config.plasma_profile_model!r}"
        )

    def evaluate_at_stored_energy(target_stored_energy_mj: jnp.ndarray) -> dict:
        """Evaluate all coupled plasma quantities at a trial stored energy."""
        if profile_model == "tokasys":
            te_profile, _, core_temperature_scale = temperature_profile_keV(
                rho=rho,
                weights=weights,
                electron_density_m3=ne_profile,
                target_stored_energy_mj=target_stored_energy_mj,
                pedestal_width=pedestal_width,
                pedestal_temperature_keV=pedestal_temperature,
                tech=tech,
                config=config,
                volume_m3=geom.plasma_volume_m3,
            )
        elif profile_model == "process_parabolic":
            te_profile, _, core_temperature_scale = (
                process_parabolic_temperature_profile_keV(
                    rho=rho,
                    weights=weights,
                    electron_density_m3=ne_profile,
                    target_stored_energy_mj=target_stored_energy_mj,
                    tech=tech,
                    config=config,
                    volume_m3=geom.plasma_volume_m3,
                )
            )
        else:
            te_profile, _, core_temperature_scale = (
                process_pedestal_temperature_profile_keV(
                    rho=rho,
                    weights=weights,
                    electron_density_m3=ne_profile,
                    target_stored_energy_mj=target_stored_energy_mj,
                    tech=tech,
                    config=config,
                    volume_m3=geom.plasma_volume_m3,
                )
            )
        ti_profile = tech.ion_to_electron_temperature_ratio * te_profile
        ni_profile = tech.fuel_ion_fraction * ne_profile
        pressure_profile = (
            ne_profile * te_profile + ni_profile * ti_profile
        ) * KEV_TO_J
        thermal_pressure = profile_average(pressure_profile, weights)
        thermal_beta_t = 2.0 * MU0 * thermal_pressure / b0**2
        stored_energy_mj = (
            1.5 * profile_average(pressure_profile, weights) * geom.plasma_volume_m3
        ) / 1.0e6
        ipb98_transport_loss_mw, ipb98_confinement_time_s = (
            ipb98y2_transport_loss_mw(
                stored_energy_mj=stored_energy_mj,
                electron_density_m3=ne,
                x=x,
                geom=geom,
                tech=tech,
            )
        )

        fusion_power_density_mw_m3 = dt_fusion_power_density_mw_m3(
            electron_density_m3=ne_profile,
            ion_temperature_keV=ti_profile,
            fuel_ion_fraction=tech.fuel_ion_fraction,
            profile_fusion_enhancement=tech.profile_fusion_enhancement,
        )
        fusion_mw = geom.plasma_volume_m3 * profile_average(
            fusion_power_density_mw_m3,
            weights,
        )
        alpha_mw = ALPHA_FRACTION * fusion_mw
        neutron_mw = NEUTRON_FRACTION * fusion_mw

        te_density_weighted = profile_average(ne_profile * te_profile, weights) / (
            profile_average(ne_profile, weights) + 1.0e-30
        )
        ti_density_weighted = profile_average(ni_profile * ti_profile, weights) / (
            profile_average(ni_profile, weights) + 1.0e-30
        )
        alpha_fast_beta_t = ipdg89_fast_alpha_beta(
            thermal_beta=thermal_beta_t,
            electron_density_m3=ne,
            fuel_ion_density_m3=tech.fuel_ion_fraction * ne,
            electron_temperature_keV=te_density_weighted,
            ion_temperature_keV=ti_density_weighted,
            alpha_power_total_mw=alpha_mw,
            alpha_power_plasma_mw=alpha_mw,
        )
        alpha_fast_pressure = alpha_fast_beta_t * b0**2 / (2.0 * MU0)
        alpha_slowing_time = (
            1.5
            * alpha_fast_pressure
            * geom.plasma_volume_m3
            / jnp.maximum(alpha_mw * 1.0e6, 1.0e-30)
        )
        auxiliary_slowing_time = fast_ion_slowing_down_time_s(
            electron_density_m3=ne,
            electron_temperature_keV=profile_average(te_profile, weights),
            initial_energy_mev=config.auxiliary_fast_ion_energy_mev,
            fast_ion_mass_kg=2.0 * ATOMIC_MASS_UNIT_KG,
            fast_ion_charge_number=1.0,
            coulomb_log=config.fast_ion_coulomb_log,
        )
        alpha_fast_pressure = fast_ion_pressure_pa(
            power_mw=alpha_mw,
            slowing_down_time_s=alpha_slowing_time,
            volume_m3=geom.plasma_volume_m3,
        )
        external_heating_mw = x.auxiliary_heating_mw + x.current_drive_power_mw
        auxiliary_fast_pressure = fast_ion_pressure_pa(
            power_mw=external_heating_mw,
            slowing_down_time_s=auxiliary_slowing_time,
            volume_m3=geom.plasma_volume_m3,
        )
        fast_ion_pressure = alpha_fast_pressure + auxiliary_fast_pressure
        auxiliary_fast_beta_t = 2.0 * MU0 * auxiliary_fast_pressure / b0**2
        fast_ion_beta_t = alpha_fast_beta_t + auxiliary_fast_beta_t
        beta_t = thermal_beta_t + fast_ion_beta_t
        beta_n = 100.0 * beta_t * a * b0 / ip_ma
        beta_p = beta_t * (
            b0 / (MU0 * ip_ma * 1.0e6 / (2.0 * jnp.pi * a))
        ) ** 2

        brems_mw = electron_ion_bremsstrahlung_mw(
            electron_density_m3=ne_profile,
            temperature_keV=te_profile,
            zeff=tech.zeff,
            volume_m3=1.0,
        )
        brems_mw = geom.plasma_volume_m3 * profile_average(brems_mw, weights)
        (
            synch_mw,
            synch_no_reflect_mw,
            synch_optical_depth,
            synch_reflectivity_factor,
            synch_profile_beta,
        ) = albajar_fidone_synchrotron_radiation_mw(
            electron_density_m3=ne_profile,
            temperature_keV=te_profile,
            radial_grid=rho,
            magnetic_field_t=b0,
            major_radius_m=r0,
            minor_radius_m=a,
            elongation=x.elongation,
            aspect_ratio=x.aspect_ratio,
            weights=weights,
            config=config,
        )
        radiation_mw = brems_mw + synch_mw

        (
            bootstrap_fraction,
            bootstrap_collisionality,
            bootstrap_pressure_gradient_factor,
        ) = bootstrap_current_model(
            rho=rho,
            weights=weights,
            electron_density_m3=ne_profile,
            electron_temperature_keV=te_profile,
            pressure_profile_pa=pressure_profile,
            q95=q95,
            beta_p=beta_p,
            geom=geom,
            x=x,
            tech=tech,
        )
        bootstrap_current_ma = bootstrap_fraction * ip_ma
        current_drive_ma, current_drive_efficiency_20 = current_drive_model(
            x=x,
            geom=geom,
            tech=tech,
            weights=weights,
            electron_density_m3=ne_profile,
            electron_temperature_keV=te_profile,
            pressure_profile_pa=pressure_profile,
        )
        resistivity_profile = spitzer_resistivity_ohm_m(te_profile, tech.zeff)
        (
            total_current_density,
            inductive_current_density,
            bootstrap_current_density,
            current_drive_current_density,
        ) = current_density_profiles(
            rho=rho,
            weights=weights,
            pressure_profile_pa=pressure_profile,
            geom=geom,
            x=x,
            bootstrap_current_ma=bootstrap_current_ma,
            current_drive_current_ma=current_drive_ma,
        )
        ohmic_power_density_profile = ohmic_power_density_mw_m3(
            resistivity_ohm_m=resistivity_profile,
            inductive_current_density_a_m2=inductive_current_density,
            total_current_density_a_m2=total_current_density,
        )
        ohmic_mw = geom.plasma_volume_m3 * profile_average(
            ohmic_power_density_profile,
            weights,
        )
        power_balance_transport_loss_mw = jnp.maximum(
            alpha_mw + external_heating_mw + ohmic_mw - radiation_mw,
            1.0e-9,
        )
        fusion_gain = fusion_mw / jnp.maximum(
            external_heating_mw,
            1.0e-6,
        )
        return {
            "thermal_pressure": thermal_pressure,
            "fast_ion_pressure": fast_ion_pressure,
            "beta_t": beta_t,
            "thermal_beta_t": thermal_beta_t,
            "fast_ion_beta_t": fast_ion_beta_t,
            "alpha_fast_beta_t": alpha_fast_beta_t,
            "auxiliary_fast_beta_t": auxiliary_fast_beta_t,
            "beta_n": beta_n,
            "beta_p": beta_p,
            "stored_energy_mj": stored_energy_mj,
            "ipb98_transport_loss_mw": ipb98_transport_loss_mw,
            "ipb98_confinement_time_s": ipb98_confinement_time_s,
            "power_balance_transport_loss_mw": power_balance_transport_loss_mw,
            "fusion_mw": fusion_mw,
            "alpha_mw": alpha_mw,
            "neutron_mw": neutron_mw,
            "brems_mw": brems_mw,
            "synch_mw": synch_mw,
            "synch_no_reflect_mw": synch_no_reflect_mw,
            "synch_optical_depth": synch_optical_depth,
            "synch_reflectivity_factor": synch_reflectivity_factor,
            "synch_profile_beta": synch_profile_beta,
            "alpha_slowing_time": alpha_slowing_time,
            "auxiliary_slowing_time": auxiliary_slowing_time,
            "radiation_mw": radiation_mw,
            "bootstrap_fraction": bootstrap_fraction,
            "bootstrap_current_ma": bootstrap_current_ma,
            "bootstrap_collisionality": bootstrap_collisionality,
            "bootstrap_pressure_gradient_factor": bootstrap_pressure_gradient_factor,
            "current_drive_ma": current_drive_ma,
            "current_drive_efficiency_20": current_drive_efficiency_20,
            "resistivity_profile": resistivity_profile,
            "total_current_density": total_current_density,
            "inductive_current_density": inductive_current_density,
            "bootstrap_current_density": bootstrap_current_density,
            "current_drive_current_density": current_drive_current_density,
            "ohmic_power_density_profile": ohmic_power_density_profile,
            "ohmic_mw": ohmic_mw,
            "fusion_gain": fusion_gain,
            "te_profile": te_profile,
            "ti_profile": ti_profile,
            "pressure_profile": pressure_profile,
            "core_temperature_scale": core_temperature_scale,
        }

    mode = config.energy_closure_mode.lower()
    if mode == "fixed_point":
        mode = "self_consistent"

    if mode == "full_space":
        target_stored_energy_mj = (
            closure.stored_energy_mj
            if closure is not None
            else initial_target_stored_energy_mj
        )
        plasma_data = evaluate_at_stored_energy(target_stored_energy_mj)
        transport_loss_mw = (
            closure.transport_loss_mw
            if closure is not None
            else plasma_data["ipb98_transport_loss_mw"]
        )
        confinement_time_s = plasma_data["stored_energy_mj"] / jnp.maximum(
            transport_loss_mw,
            1.0e-9,
        )
    elif mode == "self_consistent":
        target_stored_energy_mj = initial_target_stored_energy_mj
        for _ in range(config.energy_closure_iterations):
            plasma_data = evaluate_at_stored_energy(target_stored_energy_mj)
            transport_power = plasma_data["power_balance_transport_loss_mw"]
            confinement_time_s = ipb98y2_confinement_time_s(
                transport_loss_mw=transport_power,
                electron_density_m3=ne,
                x=x,
                geom=geom,
                tech=tech,
            )
            target_stored_energy_mj = confinement_time_s * transport_power
        plasma_data = evaluate_at_stored_energy(target_stored_energy_mj)
        transport_loss_mw = plasma_data["power_balance_transport_loss_mw"]
        confinement_time_s = ipb98y2_confinement_time_s(
            transport_loss_mw=transport_loss_mw,
            electron_density_m3=ne,
            x=x,
            geom=geom,
            tech=tech,
        )
    else:
        raise ValueError(f"Unknown energy_closure_mode={config.energy_closure_mode!r}")

    return PlasmaState(
        electron_density_m3=ne,
        greenwald_density_m3=n_gw,
        ion_density_m3=ni,
        thermal_pressure_pa=plasma_data["thermal_pressure"],
        fast_ion_pressure_pa=plasma_data["fast_ion_pressure"],
        beta_toroidal=plasma_data["beta_t"],
        thermal_beta_toroidal=plasma_data["thermal_beta_t"],
        fast_ion_beta_toroidal=plasma_data["fast_ion_beta_t"],
        alpha_fast_ion_beta_toroidal=plasma_data["alpha_fast_beta_t"],
        auxiliary_fast_ion_beta_toroidal=plasma_data["auxiliary_fast_beta_t"],
        beta_n=plasma_data["beta_n"],
        beta_p=plasma_data["beta_p"],
        q95=q95,
        stored_energy_mj=plasma_data["stored_energy_mj"],
        confinement_time_s=confinement_time_s,
        transport_loss_mw=transport_loss_mw,
        ipb98_transport_loss_mw=plasma_data["ipb98_transport_loss_mw"],
        power_balance_transport_loss_mw=plasma_data[
            "power_balance_transport_loss_mw"
        ],
        fusion_power_mw=plasma_data["fusion_mw"],
        alpha_power_mw=plasma_data["alpha_mw"],
        neutron_power_mw=plasma_data["neutron_mw"],
        bremsstrahlung_mw=plasma_data["brems_mw"],
        synchrotron_mw=plasma_data["synch_mw"],
        radiation_power_mw=plasma_data["radiation_mw"],
        ohmic_power_mw=plasma_data["ohmic_mw"],
        bootstrap_fraction=plasma_data["bootstrap_fraction"],
        bootstrap_current_ma=plasma_data["bootstrap_current_ma"],
        current_drive_current_ma=plasma_data["current_drive_ma"],
        fusion_gain=plasma_data["fusion_gain"],
        radial_grid=rho,
        volume_weights=weights,
        electron_temperature_profile_keV=plasma_data["te_profile"],
        ion_temperature_profile_keV=plasma_data["ti_profile"],
        electron_density_profile_m3=ne_profile,
        thermal_pressure_profile_pa=plasma_data["pressure_profile"],
        total_current_density_profile_a_m2=plasma_data["total_current_density"],
        inductive_current_density_profile_a_m2=plasma_data[
            "inductive_current_density"
        ],
        bootstrap_current_density_profile_a_m2=plasma_data[
            "bootstrap_current_density"
        ],
        current_drive_current_density_profile_a_m2=plasma_data[
            "current_drive_current_density"
        ],
        resistivity_profile_ohm_m=plasma_data["resistivity_profile"],
        ohmic_power_density_profile_mw_m3=plasma_data[
            "ohmic_power_density_profile"
        ],
        pedestal_width=pedestal_width,
        pedestal_pressure_pa=pedestal_pressure,
        pedestal_temperature_keV=pedestal_temperature,
        core_temperature_scale_keV=plasma_data["core_temperature_scale"],
        synchrotron_no_reflect_mw=plasma_data["synch_no_reflect_mw"],
        synchrotron_optical_depth=plasma_data["synch_optical_depth"],
        synchrotron_reflectivity_factor=plasma_data["synch_reflectivity_factor"],
        synchrotron_profile_beta=plasma_data["synch_profile_beta"],
        alpha_slowing_down_time_s=plasma_data["alpha_slowing_time"],
        auxiliary_slowing_down_time_s=plasma_data["auxiliary_slowing_time"],
        magnetic_shear_edge=magnetic_shear_edge,
        magnetic_shear_95=magnetic_shear_95,
        alpha_critical_shear=alpha_critical_shear,
        bootstrap_collisionality=plasma_data["bootstrap_collisionality"],
        bootstrap_pressure_gradient_factor=plasma_data[
            "bootstrap_pressure_gradient_factor"
        ],
        current_drive_efficiency_20=plasma_data["current_drive_efficiency_20"],
    )


__all__ = [
    "albajar_fidone_synchrotron_radiation_mw",
    "alpha_critical_pedestal",
    "bootstrap_current_model",
    "current_density_profiles",
    "current_drive_model",
    "density_at_normalized_radius_m3",
    "density_profile_m3",
    "diagnostic_q_profile_and_shear",
    "dt_fusion_power_density_mw_m3",
    "dt_reactivity_m3_s",
    "effective_profile_alpha",
    "electron_ion_bremsstrahlung_mw",
    "evaluate_plasma",
    "fast_ion_pressure_pa",
    "fast_ion_slowing_down_time_s",
    "ipdg89_fast_alpha_beta",
    "profile_average",
    "radial_derivative",
    "ohmic_power_density_mw_m3",
    "process_parabolic_density_profile_m3",
    "process_parabolic_temperature_profile_keV",
    "process_pedestal_density_profile_m3",
    "process_pedestal_temperature_profile_keV",
    "spitzer_resistivity_ohm_m",
    "temperature_profile_keV",
    "temperature_profile_beta",
]
