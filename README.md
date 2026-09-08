# TokaSys

`TokaSys` is a compact, end-to-end differentiable 0D tokamak power-plant
systems code. It is intended for rapid design-space exploration, sensitivity
analysis, and constrained reactor optimization rather than detailed component
design.

The package connects:

1. plasma geometry and radial build,
2. 0D plasma performance and power balance,
3. TF, PF, and CS magnet field/current/stress models and cryogenic loads,
4. blanket, shielding, tritium breeding, and nuclear heating,
5. divertor heat exhaust,
6. gross/net electric power and parametric economics, and
7. full-space closure residuals and positive-is-feasible inequality margins.

All subsystem evaluations are pure JAX functions. The optimizer uses a
full-space `ReactorVariables = (DesignVariables, ClosureVariables)` formulation
and SciPy SLSQP, with exact objective gradients and constraint Jacobians from
JAX automatic differentiation.

> **Warning**
> The equations and coefficients are simplified systems-level models. They are
> useful for software development and qualitative design studies, but individual
> models and coefficients must be validated before scientific or engineering use.

## Installation

Install TokaSys in editable mode from the repository root. Editable installation
keeps imports pointed at the working tree, so code changes are immediately used
by the examples, tests, and GUI.

```bash
python -m pip install -e .
```

Python 3.10 or newer is required. The package dependencies install JAX, NumPy,
SciPy, and Matplotlib; the desktop GUI additionally requires a Python build with
Tkinter support.

## Baseline Evaluation and Gradient Inspection

The baseline example evaluates the default reactor design and demonstrates the
differentiable model interface. Use it as a quick installation check and to
inspect representative outputs and local derivatives before optimizing.

```bash
python -m examples.run_baseline
```

## Constrained Design Optimization

The optimization example constructs the default full-space problem and solves
it with SLSQP. Design and closure variables are normalized to `[0, 1]`, while
JAX supplies exact gradients and constraint Jacobians to SciPy.

```bash
python -m examples.run_optimization
```

The terminal report lists the solver status, optimized design, objective
contributions, equality residuals, and inequality margins. A diagnostic figure
is written to `examples/optimization_diagnostics.png`.

## Configuration-Driven Optimization

A JSON or JSONC file can select objectives and weights, change initial values
and bounds, enable constraints, and override model or technology settings. This
keeps optimization studies reproducible without editing Python source code.

```bash
python -m examples.run_optimization \
  --config examples/optimization_reactor_config.json
```

The main configuration sections are:

- `objectives`: enabled objective terms and their scalar weights.
- `design` and `bounds`: initial design variables and physical search limits.
- `closure` and `closure_bounds`: full-space closure initial values and limits.
- `equalities` and `inequalities`: active constraints and configurable targets.
- `reactor_config`, `technology`, and `numerics`: fixed model choices and assumptions.
- `tolerances`: thresholds used to label reported constraints as satisfied.

See `examples/optimization_reactor_config.json` for a fully annotated starting
point. Solver-specific controls can also be supplied on the command line, for
example `--maxiter 400 --ftol 1e-9`.

## Desktop Optimization GUI

`GUI.py` provides the same configuration-driven workflow through a single
scrollable desktop window. It is useful for interactively changing objectives,
bounds, constraints, reactor options, and closure variables without editing JSON.

Run it from the repository root:

```bash
PYTHONPATH=. python GUI.py
```

The window is organized into three columns:

- Column 1 contains objectives, design-variable bounds and initial values, and
  equality constraints.
- Column 2 contains inequality constraints and their configurable limits or
  targets.
- Column 3 contains reactor configuration, technology assumptions, tolerances,
  solver controls, and closure-variable bounds and initial values.

Hover over a section or variable name to display an English description. The
three buttons at the bottom implement the complete interactive workflow:

- **Default** reloads `examples/optimization_reactor_config.json` and fills any
  omitted initial values from `default_problem()`.
- **Optimize** validates the inputs and runs SLSQP in a background thread. Solver
  progress and the detailed result are printed in the launching terminal, and a
  diagnostic plot window opens when the run finishes.
- **Save** writes the most recent optimization to a timestamped directory under
  `outputs/optimization_YYYYMMDD_HHMMSS/`.

Each saved run contains the configuration, optimized design and closure values,
all subsystem metrics, a constraint/objective summary, a text report, and the
diagnostic plot:

```text
outputs/optimization_YYYYMMDD_HHMMSS/
├── optimization_config.json
├── optimized_design.json
├── metrics.json
├── optimization_result.json
├── optimization_report.txt
└── optimization_diagnostics.png
```

## Testing

The test suite checks subsystem behavior, gradients, optimization configuration,
economics, and bundled PROCESS reference comparisons. Run it after changing a
model, default value, constraint, or solver interface.

```bash
python -m pytest
```

## Main Python API

The public API exposes default problem construction, direct reactor evaluation,
and normalized-variable conversion. Direct evaluation is useful for inspecting a
fixed design; the normalized full-space representation is intended for optimizers.

```python
from tokasys import default_problem, evaluate_reactor, normalize_reactor_variables
from tokasys.models.constraints import equality_vector, inequality_vector
from tokasys.models.objectives import major_radius_objective

problem = default_problem()

# Evaluate a fixed design without explicitly supplying closure variables.
result = evaluate_reactor(
    problem.initial_design,
    problem.technology,
    problem.config,
    problem.numerics,
)

# Build the full normalized vector used by constrained optimizers:
# [design variables, closure variables].
u0 = normalize_reactor_variables(
    problem.initial_variables,
    problem.variable_bounds,
)
```

`evaluate_reactor` returns a `ReactorResult` containing geometry, plasma,
magnets, nuclear island, exhaust, power-plant, economics, residual, and margin
states. The result is an immutable JAX-compatible data structure and can be
used inside differentiated functions.

## Variables, Bounds, and Closure Formulation

Design variables are the physical control knobs, including machine size,
plasma shape and operating point, heating, and radial-build thicknesses. Their
physical limits are stored in `VariableBounds`.

Closure variables include stored energy, transport loss, bootstrap current,
current-drive current, net electric power, peak TF field, TBR, and divertor heat
flux. Equality residuals enforce consistency between these variables and the
coupled subsystem calculations.

The combined design-and-closure vector uses `ReactorVariableBounds` and is
affinely normalized so every optimizer bound is `[0, 1]`. This reduces numerical
scale differences between quantities such as meters, amperes, and watts.

## Model and Technology Configuration

Non-optimized assumptions are separated into three immutable groups so that
physics choices are not confused with continuous design variables.
`ReactorConfig` selects scenarios and model variants, `TechnologyParameters`
stores engineering and economic assumptions, and `NumericalOptions` controls
discretization such as the radial profile grid size.

This separation makes a study configuration explicit and allows the same design
to be evaluated under different technologies or model choices. Configuration
files override only the specified fields and retain baseline defaults elsewhere.

## Constraint Convention

Equality constraints are dimensionless residuals that should converge to zero.
Inequality constraints are dimensionless margins defined to be positive when
feasible, which matches the convention expected by SciPy SLSQP.

The reporting helpers translate normalized residuals and margins back into
physical values, units, limits, and satisfaction flags. This makes it possible
to distinguish solver convergence from actual engineering feasibility.
