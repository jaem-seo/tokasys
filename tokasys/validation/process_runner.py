"""Run local PROCESS cases and convert MFILE outputs to TokaSys references."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESS_EXECUTABLE = PROJECT_ROOT / ".venv-process" / "bin" / "process"
DEFAULT_PROCESS_BASE_INPUT = (
    PROJECT_ROOT
    / "external"
    / "PROCESS"
    / "tests"
    / "regression"
    / "input_files"
    / "large_tokamak_eval.IN.DAT"
)
ITER_BASELINE_OVERLAY = (
    PROJECT_ROOT / "tokasys" / "validation" / "process_inputs" / "iter_baseline_overlay.IN.DAT"
)
PROCESS_SCENARIO_INPUTS = {
    "large_tokamak_eval": DEFAULT_PROCESS_BASE_INPUT,
    "low_aspect_ratio_DEMO": PROJECT_ROOT
    / "external"
    / "PROCESS"
    / "tests"
    / "regression"
    / "input_files"
    / "low_aspect_ratio_DEMO.IN.DAT",
    "spherical_tokamak_eval": PROJECT_ROOT
    / "external"
    / "PROCESS"
    / "tests"
    / "regression"
    / "input_files"
    / "spherical_tokamak_eval.IN.DAT",
}


class ProcessMetricSpec(NamedTuple):
    name: str
    process_variable: str
    path: str
    unit: str
    rtol: float
    atol: float
    transform: Callable[[float], float] = lambda value: value


PROCESS_METRIC_SPECS = (
    ProcessMetricSpec("major_radius", "rmajor", "design.major_radius_m", "m", 1.0e-3, 1.0e-3),
    ProcessMetricSpec("aspect_ratio", "aspect", "design.aspect_ratio", "", 1.0e-3, 1.0e-3),
    ProcessMetricSpec("elongation", "kappa", "design.elongation", "", 1.0e-3, 1.0e-3),
    ProcessMetricSpec("triangularity", "triang", "design.triangularity", "", 1.0e-3, 1.0e-3),
    ProcessMetricSpec(
        "toroidal_field_on_axis",
        "b_plasma_toroidal_on_axis",
        "design.toroidal_field_t",
        "T",
        1.0e-3,
        1.0e-3,
    ),
    ProcessMetricSpec("plasma_current", "plasma_current_MA", "design.plasma_current_ma", "MA", 1.0e-3, 1.0e-3),
    ProcessMetricSpec(
        "electron_density",
        "nd_plasma_electrons_vol_avg",
        "plasma.electron_density_m3",
        "m^-3",
        0.02,
        1.0e18,
    ),
    ProcessMetricSpec(
        "electron_temperature",
        "temp_plasma_electron_vol_avg_kev",
        "design.target_temperature_keV",
        "keV",
        1.0e-3,
        1.0e-3,
    ),
    ProcessMetricSpec("q95", "q95", "plasma.q95", "", 0.12, 0.1),
    ProcessMetricSpec("beta_toroidal", "beta_total_vol_avg", "plasma.beta_toroidal", "", 0.35, 0.005),
    ProcessMetricSpec("fusion_power", "p_fusion_total_mw", "plasma.fusion_power_mw", "MW", 0.75, 150.0),
    ProcessMetricSpec(
        "auxiliary_heat_power",
        "p_hcd_primary_extra_heat_mw",
        "design.auxiliary_heating_mw",
        "MW",
        1.0e-3,
        1.0e-3,
    ),
    ProcessMetricSpec(
        "current_drive_power",
        "p_hcd_injected_current_total_mw",
        "design.current_drive_power_mw",
        "MW",
        1.0e-3,
        1.0e-3,
    ),
    ProcessMetricSpec(
        "gross_electric_power",
        "p_plant_electric_gross_mw",
        "power.gross_electric_power_mw",
        "MW",
        0.75,
        100.0,
    ),
    ProcessMetricSpec(
        "net_electric_power",
        "p_plant_electric_net_mw",
        "power.flat_top_net_electric_power_mw",
        "MW",
        0.75,
        100.0,
    ),
    ProcessMetricSpec(
        "recirculating_fraction",
        "f_p_plant_electric_recirc",
        "power.recirculating_fraction",
        "",
        0.75,
        0.1,
    ),
    ProcessMetricSpec("peak_tf_field", "ineq_value_con025", "magnets.peak_tf_field_t", "T", 0.35, 0.5),
    ProcessMetricSpec(
        "cs_startup_flux_required",
        "vs_plasma_ramp_required",
        "magnets.cs_startup_flux_required_wb",
        "Wb",
        1.0,
        100.0,
    ),
    ProcessMetricSpec(
        "capital_cost",
        "capcost",
        "economics.total_capital_cost_busd",
        "BUSD",
        0.75,
        2.0,
        lambda value: value / 1.0e3,
    ),
    ProcessMetricSpec("coe", "coe", "economics.coe_usd_mwh", "USD/MWh", 1.0, 100.0),
)
PROCESS_METRIC_FALLBACKS = {
    # Some PROCESS inputs do not emit the constraint residual value but do emit
    # the corresponding peak inboard TF field directly.
    "ineq_value_con025": ("b_tf_inboard_peak_with_ripple",),
}


_MFILE_LINE_RE = re.compile(r"\s\(([^()]+)\)_+\s+(.+?)\s*$")


def _parse_value(text: str) -> Any:
    tokens = shlex.split(text, posix=False)
    if not tokens:
        return text.strip()
    token = tokens[0].strip()
    if len(token) >= 2 and token[0] in {"'", '"'} and token[-1] == token[0]:
        return token[1:-1]
    try:
        value = float(token)
    except ValueError:
        return token
    if math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def parse_mfile(path: str | Path) -> dict[str, Any]:
    """Parse scalar values from a PROCESS MFILE.DAT."""
    values: dict[str, Any] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        match = _MFILE_LINE_RE.search(line)
        if match is None:
            continue
        values[match.group(1).strip()] = _parse_value(match.group(2))
    return values


def _required_float(values: dict[str, Any], name: str) -> float:
    if name not in values:
        raise KeyError(f"PROCESS MFILE variable {name!r} was not found.")
    return float(values[name])


def _optional_float(values: dict[str, Any], name: str, default: float) -> float:
    if name not in values:
        return default
    return float(values[name])


def _metric_process_variable(values: dict[str, Any], name: str) -> str | None:
    if name in values:
        return name
    for fallback in PROCESS_METRIC_FALLBACKS.get(name, ()):
        if fallback in values:
            return fallback
    return None


def greenwald_fraction_from_process(values: dict[str, Any]) -> float:
    major_radius = _required_float(values, "rmajor")
    aspect = _required_float(values, "aspect")
    plasma_current_ma = _required_float(values, "plasma_current_MA")
    density = _required_float(values, "nd_plasma_electrons_vol_avg")
    minor_radius = major_radius / aspect
    greenwald_density = plasma_current_ma / (math.pi * minor_radius**2) * 1.0e20
    return density / greenwald_density


def build_reference_from_mfile(
    mfile_path: str | Path,
    *,
    scenario: str,
    input_file: str | Path | None = None,
    case_id: str | None = None,
    include_cost_metrics: bool = True,
) -> dict[str, Any]:
    """Create an in-memory PROCESS reference-case mapping from an MFILE."""
    values = parse_mfile(mfile_path)
    scenario_id = scenario.replace(" ", "_")
    process_version = str(values.get("procver", "local"))
    reference_case_id = case_id or f"process_{scenario_id}_{process_version}".replace(
        "+",
        "_",
    )

    design = {
        "major_radius_m": _required_float(values, "rmajor"),
        "aspect_ratio": _required_float(values, "aspect"),
        "elongation": _required_float(values, "kappa"),
        "triangularity": _required_float(values, "triang"),
        "toroidal_field_t": _required_float(values, "b_plasma_toroidal_on_axis"),
        "plasma_current_ma": _required_float(values, "plasma_current_MA"),
        "greenwald_fraction": greenwald_fraction_from_process(values),
        "target_temperature_keV": _required_float(
            values,
            "temp_plasma_electron_vol_avg_kev",
        ),
        "confinement_h98": _optional_float(values, "hfact", 1.0),
        "auxiliary_heating_mw": _optional_float(values, "p_hcd_primary_extra_heat_mw", 0.0),
        "current_drive_power_mw": _optional_float(
            values,
            "p_hcd_injected_current_total_mw",
            0.0,
        ),
        "inboard_first_wall_thickness_m": _optional_float(values, "dr_fw_inboard", 0.05),
        "outboard_first_wall_thickness_m": _optional_float(values, "dr_fw_outboard", 0.05),
        "inboard_blanket_thickness_m": _optional_float(values, "dr_blkt_inboard", 0.85),
        "outboard_blanket_thickness_m": _optional_float(values, "dr_blkt_outboard", 0.85),
        "inboard_shield_thickness_m": _optional_float(values, "dr_shld_inboard", 0.55),
        "outboard_shield_thickness_m": _optional_float(values, "dr_shld_outboard", 0.55),
        "tf_coil_thickness_m": _optional_float(values, "dr_tf_inboard", 1.0),
        "cs_thickness_m": _optional_float(values, "dr_cs", 0.8),
    }

    config = {
        "steady_state": False,
        "energy_closure_mode": "full_space",
        "plasma_profile_model": "process_pedestal"
        if int(_optional_float(values, "i_plasma_pedestal", 1.0)) == 1
        else "process_parabolic",
        "process_profile_alphan": _optional_float(values, "alphan", 0.9),
        "process_profile_alphat": _optional_float(values, "alphat", 1.4),
        "process_profile_tbeta": _optional_float(values, "tbeta", 2.0),
        "process_density_pedestal_radius": _optional_float(
            values,
            "radius_plasma_pedestal_density_norm",
            0.95,
        ),
        "process_temperature_pedestal_radius": _optional_float(
            values,
            "radius_plasma_pedestal_temp_norm",
            0.95,
        ),
        "process_density_pedestal_greenwald_fraction": _optional_float(
            values,
            "f_nd_plasma_pedestal_greenwald",
            0.8,
        ),
        "process_density_separatrix_greenwald_fraction": _optional_float(
            values,
            "f_nd_plasma_separatrix_greenwald",
            0.1,
        ),
        "process_temperature_pedestal_keV": _optional_float(
            values,
            "temp_plasma_pedestal_kev",
            4.5,
        ),
        "process_temperature_separatrix_keV": _optional_float(
            values,
            "temp_plasma_separatrix_kev",
            0.1,
        ),
    }
    technology = {
        "thermal_efficiency": _optional_float(values, "eta_turbine", 0.4),
        "net_electric_target_mw": _optional_float(
            values,
            "p_plant_electric_net_required_mw",
            400.0,
        ),
    }
    metrics = []
    for spec in PROCESS_METRIC_SPECS:
        if not include_cost_metrics and spec.name in {"capital_cost", "coe"}:
            continue
        process_variable = _metric_process_variable(values, spec.process_variable)
        if process_variable is None:
            continue
        metrics.append(
            {
                "name": spec.name,
                "process_variable": process_variable,
                "path": spec.path,
                "unit": spec.unit,
                "reference": spec.transform(float(values[process_variable])),
                "rtol": spec.rtol,
                "atol": spec.atol,
            }
        )

    source_input = str(input_file) if input_file is not None else str(values.get("fileprefix", ""))
    return {
        "case_id": reference_case_id,
        "description": f"Local PROCESS {scenario} run converted from MFILE output.",
        "source": {
            "code": "PROCESS",
            "scenario": scenario,
            "process_repository": "external/PROCESS",
            "input_file": source_input,
            "reference_mfile": str(mfile_path),
            "process_version": process_version,
            "run_date": values.get("date", ""),
            "ifail": values.get("ifail", ""),
            "sqsumsq": values.get("sqsumsq", ""),
            "notes": "Generated locally by tokasys.validation.process_runner.",
        },
        "tokasys_problem": {
            "design": design,
            "bounds": {"upper": {"greenwald_fraction": max(1.2, 1.2 * design["greenwald_fraction"])}},
            "technology": technology,
            "config": config,
        },
        "metrics": metrics,
    }


def generate_process_input(
    output_path: str | Path,
    *,
    base_input: str | Path = DEFAULT_PROCESS_BASE_INPUT,
    overlay: str | Path | None = ITER_BASELINE_OVERLAY,
    one_shot: bool = False,
) -> Path:
    """Generate a PROCESS IN.DAT by appending an override overlay to a base case."""
    output = Path(output_path)
    text = Path(base_input).read_text(encoding="utf-8")
    if overlay is not None:
        selected_overlay = Path(overlay)
        text += "\n\n*---------------TokaSys generated overrides---------------*\n"
        text += selected_overlay.read_text(encoding="utf-8")
    if one_shot:
        text = _make_one_shot_process_input(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def _make_one_shot_process_input(text: str) -> str:
    """Disable PROCESS equality closure variables for forward-model comparison."""
    lines = []
    replaced_neqns = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ixc ="):
            continue
        if stripped.startswith("neqns"):
            if not replaced_neqns:
                lines.append(
                    "neqns = 0 * TokaSys one-shot comparison: no equality closure solve",
                )
                replaced_neqns = True
            continue
        lines.append(line)
    if not replaced_neqns:
        lines.insert(
            0,
            "neqns = 0 * TokaSys one-shot comparison: no equality closure solve",
        )
    lines.append(
        "* NOTE: active ixc lines were removed so PROCESS evaluates the supplied scalars.",
    )
    return "\n".join(lines) + "\n"


def run_process(
    input_path: str | Path,
    *,
    workdir: str | Path,
    process_executable: str | Path = DEFAULT_PROCESS_EXECUTABLE,
    mfile_name: str = "MFILE.DAT",
) -> Path:
    """Run PROCESS on an IN.DAT and return the produced MFILE path."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    mfile_path = work / mfile_name
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(work / "mplconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    command = [
        str(process_executable),
        "-i",
        str(Path(input_path).resolve()),
        "-m",
        str(mfile_path.resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=work,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    (work / "process.log").write_text(
        completed.stdout + "\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PROCESS failed with exit code {completed.returncode}. "
            f"See {(work / 'process.log')}.",
        )
    return mfile_path


def write_reference_json(reference: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    return output


__all__ = [
    "DEFAULT_PROCESS_BASE_INPUT",
    "DEFAULT_PROCESS_EXECUTABLE",
    "ITER_BASELINE_OVERLAY",
    "PROCESS_SCENARIO_INPUTS",
    "build_reference_from_mfile",
    "generate_process_input",
    "greenwald_fraction_from_process",
    "parse_mfile",
    "run_process",
    "write_reference_json",
]
