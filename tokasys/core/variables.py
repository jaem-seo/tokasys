"""Packing and affine normalization of design variables."""

from __future__ import annotations

import jax.numpy as jnp

from tokasys.core.types import (
    ClosureVariables,
    DesignVariables,
    ReactorVariableBounds,
    ReactorVariables,
    VariableBounds,
)


FIELD_NAMES = DesignVariables._fields
CLOSURE_FIELD_NAMES = ClosureVariables._fields
REACTOR_FIELD_NAMES = FIELD_NAMES + tuple(f"closure.{name}" for name in CLOSURE_FIELD_NAMES)


def design_to_vector(design: DesignVariables) -> jnp.ndarray:
    """Pack named design variables into a vector using canonical field order."""
    return jnp.stack([jnp.asarray(getattr(design, name)) for name in FIELD_NAMES])


def vector_to_design(vector: jnp.ndarray) -> DesignVariables:
    """Unpack a canonical one-dimensional vector into named design variables."""
    if vector.shape != (len(FIELD_NAMES),):
        raise ValueError(f"Expected vector shape {(len(FIELD_NAMES),)}, got {vector.shape}")
    return DesignVariables(*[vector[i] for i in range(len(FIELD_NAMES))])


def normalize_design(design: DesignVariables, bounds: VariableBounds) -> jnp.ndarray:
    """Map physical design variables affinely into the unit hypercube."""
    x = design_to_vector(design)
    lo = design_to_vector(bounds.lower)
    hi = design_to_vector(bounds.upper)
    return (x - lo) / (hi - lo)


def denormalize_design(normalized: jnp.ndarray, bounds: VariableBounds) -> DesignVariables:
    """Map a unit-hypercube design vector back to physical variables."""
    lo = design_to_vector(bounds.lower)
    hi = design_to_vector(bounds.upper)
    return vector_to_design(lo + normalized * (hi - lo))


def closure_to_vector(closure: ClosureVariables) -> jnp.ndarray:
    """Pack closure variables into a vector using canonical field order."""
    return jnp.stack([jnp.asarray(getattr(closure, name)) for name in CLOSURE_FIELD_NAMES])


def vector_to_closure(vector: jnp.ndarray) -> ClosureVariables:
    """Unpack a canonical one-dimensional vector into closure variables."""
    if vector.shape != (len(CLOSURE_FIELD_NAMES),):
        raise ValueError(
            f"Expected closure vector shape {(len(CLOSURE_FIELD_NAMES),)}, got {vector.shape}"
        )
    return ClosureVariables(*[vector[i] for i in range(len(CLOSURE_FIELD_NAMES))])


def reactor_variables_to_vector(variables: ReactorVariables) -> jnp.ndarray:
    """Concatenate design and closure variables into the optimizer vector."""
    return jnp.concatenate(
        [
            design_to_vector(variables.design),
            closure_to_vector(variables.closure),
        ]
    )


def vector_to_reactor_variables(vector: jnp.ndarray) -> ReactorVariables:
    """Split an optimizer vector into its design and closure components."""
    expected = len(FIELD_NAMES) + len(CLOSURE_FIELD_NAMES)
    if vector.shape != (expected,):
        raise ValueError(f"Expected reactor vector shape {(expected,)}, got {vector.shape}")
    return ReactorVariables(
        design=vector_to_design(vector[: len(FIELD_NAMES)]),
        closure=vector_to_closure(vector[len(FIELD_NAMES) :]),
    )


def normalize_reactor_variables(
    variables: ReactorVariables,
    bounds: ReactorVariableBounds,
) -> jnp.ndarray:
    """Map coupled reactor variables affinely into the unit hypercube."""
    x = reactor_variables_to_vector(variables)
    lo = reactor_variables_to_vector(bounds.lower)
    hi = reactor_variables_to_vector(bounds.upper)
    return (x - lo) / (hi - lo)


def denormalize_reactor_variables(
    normalized: jnp.ndarray,
    bounds: ReactorVariableBounds,
) -> ReactorVariables:
    """Recover physical reactor variables from a normalized optimizer vector."""
    lo = reactor_variables_to_vector(bounds.lower)
    hi = reactor_variables_to_vector(bounds.upper)
    return vector_to_reactor_variables(lo + normalized * (hi - lo))
