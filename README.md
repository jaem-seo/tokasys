# TokaSys skeleton

`TokaSys` is a deliberately compact, end-to-end differentiable 0D tokamak
power-plant systems-code skeleton. It is intended as a clean starting point for
reactor design research, not as a validated engineering design tool.

The package connects:

1. plasma geometry and radial build,
2. 0D plasma performance and power balance,
3. TF magnet field/current/stress and cryogenic load,
4. blanket, shielding, tritium breeding and nuclear heating,
5. divertor heat exhaust,
6. gross/net electric power and parametric economics,
7. full-space closure residuals and positive-is-feasible inequality margins.

All subsystem evaluations are pure JAX functions. The optimizer wrapper uses a
full-space `ReactorVariables = (DesignVariables, ClosureVariables)`
formulation and SciPy SLSQP while supplying exact objective gradients and
constraint Jacobians computed by JAX automatic differentiation.

> **Warning**
> The equations and coefficients are intentionally simplified placeholders.
> They provide a complete differentiable software architecture and qualitatively
> sensible couplings, but they must be replaced and validated before scientific
> or engineering use.

## Install

```bash
python -m pip install -e .
```

## Run the baseline and inspect gradients

```bash
python -m examples.run_baseline
```

## Run a constrained design optimization

```bash
python -m examples.run_optimization
```

## Test

```bash
python -m pytest
```

## Main API

```python
from tokasys import default_problem, evaluate_reactor, normalize_reactor_variables
from tokasys.models.objectives import major_radius_objective
from tokasys.models.constraints import equality_vector, inequality_vector

problem = default_problem()

# Design-only evaluation remains available for direct model inspection.
result = evaluate_reactor(
    problem.initial_design,
    problem.technology,
    problem.config,
    problem.numerics,
)

# Optimizers operate on the full normalized variable vector:
#   [design variables, closure variables]
u0 = normalize_reactor_variables(problem.initial_variables, problem.variable_bounds)
```

The normalized-vector interface in `tokasys.core.variables` is intended for
optimizers. Design variables remain the physical control knobs. Closure
variables currently include stored energy, transport loss, bootstrap current,
current-drive current, net electric power, TF peak field, TBR, and divertor
heat flux. Equality residuals enforce consistency between those closure
variables and the coupled subsystem models.

The problem definition keeps three non-optimized option groups separate:
`TechnologyParameters` for plant/engineering constants, `ReactorConfig` for
model and scenario choices, and `NumericalOptions` for discretization choices
such as the radial profile grid size.

Bounds are always `[0, 1]`; physical lower and upper limits are stored in
`VariableBounds` for design-only vectors and `ReactorVariableBounds` for the
full-space formulation.
