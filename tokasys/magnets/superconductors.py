"""Superconductor critical-current and winding-pack limits.

Refs:
    Bottura, IEEE Trans. Appl. Supercond. 10, 1054-1057 (2000), as a reference
    for practical critical-surface fitting. The REBCO/Nb3Sn functions here are
    reduced differentiable engineering derating curves, not vendor-specific
    conductor qualification data.
"""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import ReactorConfig, TechnologyParameters


def superconductor_critical_current_density(
    peak_field_t: jnp.ndarray,
    tech: TechnologyParameters,
    config: ReactorConfig,
    superconductor_id: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduced REBCO/Nb3Sn/NbTi critical-current surface Jc(B, T, strain)."""
    selected_id = config.tf_superconductor_id if superconductor_id is None else superconductor_id
    use_nb3sn = selected_id == 1
    use_nbti = selected_id == 2
    reference_j = jnp.where(
        use_nbti,
        tech.nbti_reference_current_density_a_m2,
        jnp.where(
            use_nb3sn,
            tech.nb3sn_reference_current_density_a_m2,
            tech.rebco_reference_current_density_a_m2,
        ),
    )
    upper_critical_field = jnp.where(
        use_nbti,
        tech.nbti_upper_critical_field_t,
        jnp.where(
            use_nb3sn,
            tech.nb3sn_upper_critical_field_t,
            tech.rebco_upper_critical_field_t,
        ),
    )
    operating_temperature = jnp.where(
        use_nbti,
        tech.nbti_operating_temperature_k,
        jnp.where(
            use_nb3sn,
            tech.nb3sn_operating_temperature_k,
            tech.rebco_operating_temperature_k,
        ),
    )
    reference_temperature = jnp.where(
        use_nbti,
        tech.nbti_reference_temperature_k,
        jnp.where(
            use_nb3sn,
            tech.nb3sn_reference_temperature_k,
            tech.rebco_reference_temperature_k,
        ),
    )
    critical_temperature = jnp.where(
        use_nbti,
        tech.nbti_critical_temperature_k,
        jnp.where(
            use_nb3sn,
            tech.nb3sn_critical_temperature_k,
            tech.rebco_critical_temperature_k,
        ),
    )
    operating_strain = jnp.where(
        use_nbti,
        tech.nbti_operating_strain,
        jnp.where(
            use_nb3sn,
            tech.nb3sn_operating_strain,
            tech.rebco_operating_strain,
        ),
    )
    reference_strain = jnp.where(
        use_nbti,
        tech.nbti_reference_strain,
        jnp.where(
            use_nb3sn,
            tech.nb3sn_reference_strain,
            tech.rebco_reference_strain,
        ),
    )
    strain_scale = jnp.where(
        use_nbti,
        tech.nbti_strain_scale,
        jnp.where(
            use_nb3sn,
            tech.nb3sn_strain_scale,
            tech.rebco_strain_scale,
        ),
    )

    field_fraction = peak_field_t / jnp.maximum(upper_critical_field, 1.0e-9)
    field_exponent = jnp.where(use_nbti, 1.9, jnp.where(use_nb3sn, 1.7, 1.35))
    field_derating = jnp.clip(1.0 - field_fraction**field_exponent, 0.02, 1.0)

    temperature_exponent = jnp.where(use_nbti, 1.5, jnp.where(use_nb3sn, 1.7, 2.0))
    operating_tbar = jnp.clip(
        operating_temperature / jnp.maximum(critical_temperature, 1.0e-9),
        0.0,
        0.999,
    )
    reference_tbar = jnp.clip(
        reference_temperature / jnp.maximum(critical_temperature, 1.0e-9),
        0.0,
        0.999,
    )
    raw_temperature_derating = jnp.clip(
        1.0 - operating_tbar**temperature_exponent,
        0.02,
        1.0,
    )
    raw_reference_temperature = jnp.clip(
        1.0 - reference_tbar**temperature_exponent,
        0.02,
        1.0,
    )
    temperature_derating = raw_temperature_derating / jnp.maximum(
        raw_reference_temperature,
        1.0e-9,
    )

    strain_offset = (operating_strain - reference_strain) / jnp.maximum(
        strain_scale,
        1.0e-9,
    )
    raw_strain_derating = jnp.clip(1.0 - strain_offset**2, 0.05, 1.0)
    strain_exponent = jnp.where(use_nbti, 1.0, jnp.where(use_nb3sn, 1.4, 0.7))
    strain_derating = raw_strain_derating**strain_exponent

    critical_j = (
        reference_j * field_derating * temperature_derating * strain_derating
    )
    return (
        critical_j,
        operating_temperature,
        operating_strain,
        field_derating,
        temperature_derating,
        strain_derating,
    )


def winding_pack_current_density_limit(
    peak_field_t: jnp.ndarray,
    tech: TechnologyParameters,
    config: ReactorConfig,
    superconductor_id: int | None = None,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Operating winding-pack current density limit from Jc(B,T,strain)."""
    (
        critical_j,
        operating_temperature,
        operating_strain,
        field_derating,
        temperature_derating,
        strain_derating,
    ) = superconductor_critical_current_density(
        peak_field_t,
        tech,
        config,
        superconductor_id=superconductor_id,
    )
    material_limit = tech.tf_superconductor_operating_fraction * critical_j
    return (
        jnp.minimum(material_limit, tech.tf_engineering_current_density_limit_a_m2),
        critical_j,
        operating_temperature,
        operating_strain,
        field_derating,
        temperature_derating,
        strain_derating,
    )
