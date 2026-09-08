"""PROCESS reference-case validation utilities.

Refs:
    PROCESS: Kovari et al., Fusion Eng. Des. 89, 3054-3069 (2014), Kovari et
    al., Fusion Eng. Des. 104, 9-20 (2016), and the UKAEA PROCESS repository
    regression input/MFILE cases bundled as JSON mappings here.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

import jax.numpy as jnp

from tokasys.core.defaults import default_problem
from tokasys.core.types import DesignVariables, ReactorProblem, ReactorResult, ReactorVariables
from tokasys.models.reactor import evaluate_reactor


def _open_json_resource(path: Any) -> dict[str, Any]:
    """Read a JSON mapping from a filesystem or package-resource path."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def bundled_reference_case_paths() -> list[Any]:
    """Return bundled PROCESS reference-case JSON resources."""
    reference_dir = resources.files("tokasys.validation.reference_cases")
    return sorted(
        (
            path
            for path in reference_dir.iterdir()
            if path.name.endswith(".json")
        ),
        key=lambda path: path.name,
    )


def load_reference_case(path: str | Path | Any | None = None) -> dict[str, Any]:
    """Load a PROCESS reference case JSON document."""
    if path is None:
        resource = resources.files("tokasys.validation.reference_cases").joinpath(
            "process_reference_baseline.json"
        )
        return _open_json_resource(resource)
    if hasattr(path, "open"):
        return _open_json_resource(path)
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_reference_cases(
    paths: Iterable[str | Path | Any] | None = None,
) -> list[dict[str, Any]]:
    """Load all bundled reference cases, or the explicit JSON paths given."""
    selected_paths = bundled_reference_case_paths() if paths is None else list(paths)
    return [load_reference_case(path) for path in selected_paths]


def load_reference_cases_from_directory(path: str | Path) -> list[dict[str, Any]]:
    """Load every JSON reference case in a directory."""
    directory = Path(path)
    return load_reference_cases(sorted(directory.glob("*.json")))


def _expand_legacy_build_fields(values: dict[str, Any]) -> dict[str, Any]:
    """Expand symmetric legacy build inputs into inboard/outboard values."""
    expanded = dict(values)
    legacy_pairs = {
        "first_wall_thickness_m": (
            "inboard_first_wall_thickness_m",
            "outboard_first_wall_thickness_m",
        ),
        "blanket_thickness_m": (
            "inboard_blanket_thickness_m",
            "outboard_blanket_thickness_m",
        ),
        "shield_thickness_m": (
            "inboard_shield_thickness_m",
            "outboard_shield_thickness_m",
        ),
    }
    for legacy_name, split_names in legacy_pairs.items():
        if legacy_name not in expanded:
            continue
        value = expanded.pop(legacy_name)
        for split_name in split_names:
            expanded.setdefault(split_name, value)
    return expanded


def _replace_array_fields(obj: Any, values: dict[str, Any]) -> Any:
    """Replace selected NamedTuple fields with floating-point JAX arrays."""
    values = _expand_legacy_build_fields(values)
    return obj._replace(
        **{name: jnp.asarray(value, dtype=float) for name, value in values.items()}
    )


def build_process_reference_problem(
    reference: dict[str, Any] | None = None,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> ReactorProblem:
    """Construct a TokaSys problem from the PROCESS reference-case mapping."""
    if reference is None:
        reference = load_reference_case()

    problem = default_problem()
    overrides = reference.get("tokasys_problem", {})

    design = _replace_array_fields(
        problem.initial_design,
        overrides.get("design", {}),
    )
    closure = _replace_array_fields(
        problem.initial_variables.closure,
        overrides.get("closure", {}),
    )

    bounds = overrides.get("bounds", {})
    lower = _replace_array_fields(problem.bounds.lower, bounds.get("lower", {}))
    upper = _replace_array_fields(problem.bounds.upper, bounds.get("upper", {}))
    variable_lower = problem.variable_bounds.lower._replace(design=lower)
    variable_upper = problem.variable_bounds.upper._replace(design=upper)

    technology = problem.technology._replace(**overrides.get("technology", {}))
    config = problem.config._replace(**overrides.get("config", {}))
    if config_overrides:
        config = config._replace(**config_overrides)
    optimization = problem.optimization._replace(**overrides.get("optimization", {}))
    numerics = problem.numerics._replace(**overrides.get("numerics", {}))

    return problem._replace(
        initial_design=design,
        initial_variables=ReactorVariables(
            design=design,
            closure=closure,
        ),
        bounds=problem.bounds._replace(lower=lower, upper=upper),
        variable_bounds=problem.variable_bounds._replace(
            lower=variable_lower,
            upper=variable_upper,
        ),
        technology=technology,
        config=config,
        optimization=optimization,
        numerics=numerics,
    )


def _resolve_path(
    path: str,
    *,
    design: DesignVariables,
    result: ReactorResult,
) -> float:
    """Resolve a dotted design or reactor-result path to a scalar value."""
    root_name, *parts = path.split(".")
    if root_name == "design":
        obj: Any = design
    else:
        obj = getattr(result, root_name)
    for part in parts:
        obj = getattr(obj, part)
    return float(obj)


def compare_to_reference(
    problem: ReactorProblem | None = None,
    reference: dict[str, Any] | None = None,
    *,
    use_full_space: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a reactor evaluation against a PROCESS reference case.

    By default this is a design-point forward evaluation: PROCESS-derived design
    variables are evaluated directly without imposing TokaSys closure variables.
    Set use_full_space=True to compare the full-space closure formulation.
    """
    if reference is None:
        reference = load_reference_case()
    if problem is None:
        problem = build_process_reference_problem(
            reference,
            config_overrides=config_overrides,
        )

    evaluation_input = problem.initial_variables if use_full_space else problem.initial_design
    result = evaluate_reactor(
        evaluation_input,
        problem.technology,
        problem.config,
        problem.numerics,
    )
    metric_reports = []
    for metric in reference["metrics"]:
        value = _resolve_path(
            metric["path"],
            design=problem.initial_design,
            result=result,
        )
        expected = float(metric["reference"])
        abs_error = value - expected
        rel_error = abs_error / max(abs(expected), 1.0e-30)
        passed = (
            abs(abs_error) <= float(metric.get("atol", 0.0))
            or abs(rel_error) <= float(metric.get("rtol", 0.0))
        )
        metric_reports.append(
            {
                "name": metric["name"],
                "path": metric["path"],
                "process_variable": metric.get("process_variable", ""),
                "unit": metric.get("unit", ""),
                "value": value,
                "reference": expected,
                "absolute_error": abs_error,
                "relative_error": rel_error,
                "rtol": float(metric.get("rtol", 0.0)),
                "atol": float(metric.get("atol", 0.0)),
                "passed": bool(passed),
            }
        )
    num_passed = sum(1 for row in metric_reports if row["passed"])
    return {
        "case_id": reference["case_id"],
        "description": reference.get("description", ""),
        "source": reference.get("source", {}),
        "passed": num_passed == len(metric_reports),
        "num_passed": num_passed,
        "num_failed": len(metric_reports) - num_passed,
        "metrics": metric_reports,
    }


def compare_reference_suite(
    references: Iterable[dict[str, Any]] | None = None,
    *,
    paths: Iterable[str | Path | Any] | None = None,
    directory: str | Path | None = None,
    use_full_space: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare TokaSys against a collection of PROCESS reference cases."""
    if references is None:
        if directory is not None:
            references = load_reference_cases_from_directory(directory)
        else:
            references = load_reference_cases(paths)
    reports = [
        compare_to_reference(
            reference=reference,
            use_full_space=use_full_space,
            config_overrides=config_overrides,
        )
        for reference in references
    ]
    num_passed_cases = sum(1 for report in reports if report["passed"])
    num_metrics = sum(len(report["metrics"]) for report in reports)
    num_passed_metrics = sum(report["num_passed"] for report in reports)
    return {
        "passed": num_passed_cases == len(reports),
        "num_cases": len(reports),
        "num_passed_cases": num_passed_cases,
        "num_failed_cases": len(reports) - num_passed_cases,
        "num_metrics": num_metrics,
        "num_passed_metrics": num_passed_metrics,
        "num_failed_metrics": num_metrics - num_passed_metrics,
        "use_full_space": bool(use_full_space),
        "cases": reports,
    }


def format_reference_comparison(report: dict[str, Any]) -> str:
    """Format a reference comparison as a compact text table."""
    status = "passed" if report["passed"] else "failed"
    lines = [
        f"{report['case_id']} {status}",
        "metric                         value       reference   rel.err",
    ]
    for row in report["metrics"]:
        marker = " " if row["passed"] else "!"
        lines.append(
            f"{marker} {row['name']:<28} "
            f"{row['value']:>10.4g} {row['reference']:>12.4g} "
            f"{row['relative_error']:>9.2%}"
        )
    return "\n".join(lines)


def format_reference_suite(report: dict[str, Any]) -> str:
    """Format a multi-case PROCESS reference suite comparison."""
    status = "passed" if report["passed"] else "failed"
    lines = [
        (
            f"PROCESS reference suite {status}: "
            f"{report['num_passed_cases']}/{report['num_cases']} cases, "
            f"{report['num_passed_metrics']}/{report['num_metrics']} metrics"
        )
    ]
    for case_report in report["cases"]:
        case_status = "passed" if case_report["passed"] else "failed"
        worst = max_relative_error(case_report)
        scenario = case_report.get("source", {}).get("scenario", "")
        scenario_text = f" ({scenario})" if scenario else ""
        lines.append(
            f"- {case_report['case_id']}{scenario_text}: "
            f"{case_status}, {case_report['num_passed']}/"
            f"{len(case_report['metrics'])} metrics, max |rel.err|={worst:.2%}"
        )
    return "\n".join(lines)


def failed_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only metric rows that exceed their reference tolerance."""
    return [row for row in report["metrics"] if not row["passed"]]


def max_relative_error(report: dict[str, Any]) -> float:
    """Return the largest absolute relative error in a comparison report."""
    return float(max(abs(row["relative_error"]) for row in report["metrics"]))
