"""Public API for the TokaSys differentiable systems-code skeleton.

64-bit mode is enabled because reactor systems calculations combine densities of
order 1e20 with engineering/economic quantities across many decades. Users who
need a different global JAX policy should configure JAX before importing this
package and remove this line in their fork.
"""

import jax

jax.config.update("jax_enable_x64", True)

from tokasys.core.defaults import (
    default_config,
    default_numerics,
    default_optimization,
    default_problem,
    default_technology,
)
from tokasys.core.types import (
    ClosureVariables,
    DesignVariables,
    NumericalOptions,
    OptimizationConfig,
    ReactorConfig,
    ReactorVariables,
    TechnologyParameters,
)
from tokasys.core.variables import (
    normalize_design,
    normalize_reactor_variables,
)
from tokasys.models.reactor import evaluate_reactor
from tokasys.analysis.constraints import analyze_constraints, format_constraint_report

__all__ = [
    "analyze_constraints",
    "ClosureVariables",
    "default_config",
    "default_numerics",
    "default_optimization",
    "default_problem",
    "default_technology",
    "DesignVariables",
    "evaluate_reactor",
    "format_constraint_report",
    "normalize_design",
    "normalize_reactor_variables",
    "NumericalOptions",
    "OptimizationConfig",
    "ReactorConfig",
    "ReactorVariables",
    "TechnologyParameters",
]
