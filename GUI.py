"""Desktop GUI for configuring, running, and saving a TokaSys optimization.

Run from the repository root with::

    PYTHONPATH=. python GUI.py

The GUI deliberately uses only Tkinter (plus Pillow, already used by the project
environment) so that it remains a single-file entry point.
"""

from __future__ import annotations

import json
import math
import queue
import shutil
import tempfile
import threading
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk

from tokasys import default_problem
from tokasys.core.types import ClosureVariables, DesignVariables, MarginState, ResidualState
from tokasys.models.objectives import DEFAULT_OBJECTIVE_WEIGHTS, OBJECTIVE_FUNCTIONS
from tokasys.solvers.config import apply_optimization_config, load_optimization_config
from tokasys.solvers.optimize import DesignSolution, solve_slsqp
from tokasys.solvers.reporting import equality_reports, inequality_reports, objective_reports


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "examples" / "optimization_reactor_config.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ENTRY_WIDTH = 7

OBJECTIVE_DISPLAY_NAMES = {
    "major_radius": "Major radius",
    "coe": "COE",
    "negative_net_power": "Net power",
}


OBJECTIVE_DESCRIPTIONS = {
    "major_radius": "Minimizes the plasma major radius R0 [m]. Its objective contribution is weight × R0.",
    "coe": "Minimizes the levelized cost of electricity [USD/MWh]. Its contribution is weight × COE.",
    "negative_net_power": "Maximizes net electric power [MW] through the contribution -weight × Pnet.",
}

DESIGN_DESCRIPTIONS = {
    "major_radius_m": "Plasma major radius R0 [m].",
    "aspect_ratio": "Aspect ratio A = R0/a, where a is the plasma minor radius.",
    "elongation": "Plasma cross-section elongation κ.",
    "triangularity": "Plasma cross-section triangularity δ.",
    "toroidal_field_t": "On-axis toroidal magnetic field B0 [T].",
    "plasma_current_ma": "Total plasma current Ip [MA].",
    "greenwald_fraction": "Operating density as a fraction of the Greenwald density limit.",
    "target_temperature_keV": "Volume-averaged target temperature used by the profile and closure models [keV].",
    "confinement_h98": "Energy-confinement multiplier relative to IPB98(y,2).",
    "auxiliary_heating_mw": "Auxiliary heating power injected into the plasma [MW].",
    "current_drive_power_mw": "Power injected for non-inductive current drive [MW].",
    "inboard_first_wall_thickness_m": "Inboard first-wall thickness [m].",
    "outboard_first_wall_thickness_m": "Outboard first-wall thickness [m].",
    "inboard_blanket_thickness_m": "Inboard blanket thickness [m].",
    "outboard_blanket_thickness_m": "Outboard blanket thickness [m].",
    "inboard_shield_thickness_m": "Inboard shield thickness [m].",
    "outboard_shield_thickness_m": "Outboard shield thickness [m].",
    "tf_coil_thickness_m": "Radial thickness of the TF coil [m].",
    "pf_coil_thickness_m": "Representative radial/vertical thickness of the PF coils [m].",
    "cs_thickness_m": "Central-solenoid radial build measured from the machine centerline [m].",
}

CLOSURE_DESCRIPTIONS = {
    "stored_energy_mj": "Closure variable for stored plasma energy [MJ].",
    "transport_loss_mw": "Closure variable for transport loss power [MW].",
    "bootstrap_current_ma": "Closure variable for bootstrap current [MA].",
    "current_drive_current_ma": "Closure variable for current-drive current [MA].",
    "net_electric_power_mw": "Closure variable for net electric power [MW].",
    "tf_peak_field_t": "Closure variable for peak TF field [T].",
    "tbr": "Closure variable for the tritium breeding ratio.",
    "divertor_heat_flux_mw_m2": "Closure variable for peak divertor heat flux [MW/m²].",
}

EQUALITY_DESCRIPTIONS = {
    "plasma_power_balance": "Balances plasma input power against radiation and transport losses.",
    "plasma_current_balance": "Matches total plasma current to the sum of bootstrap and driven currents.",
    "target_net_power": "Requires net electric power to equal the specified target [MW].",
    "stored_energy_closure": "Matches the stored-energy closure variable to the plasma-model value.",
    "transport_loss_closure": "Matches the transport-loss closure variable to the IPB98 model value.",
    "bootstrap_current_closure": "Matches the bootstrap-current closure variable to the model value.",
    "current_drive_current_closure": "Matches the current-drive closure variable to the model value.",
    "net_electric_power_closure": "Matches the net-power closure variable to the plant-model value.",
    "tf_peak_field_closure": "Matches the peak-TF-field closure variable to the magnet-model value.",
    "tbr_closure": "Matches the TBR closure variable to the nuclear-model value.",
    "divertor_heat_flux_closure": "Matches the divertor heat-flux closure variable to the exhaust-model value.",
}

INEQUALITY_DESCRIPTIONS = {
    "q95": "Requires q95 to remain at or above the minimum limit.",
    "beta_n": "Requires normalized beta to remain at or below the maximum limit.",
    "greenwald": "Requires the Greenwald fraction to remain at or below the maximum limit.",
    "fusion_gain": "Requires plasma gain Q to remain at or above the minimum limit.",
    "target_net_power": "Requires net electric power to remain at or above the target [MW].",
    "inboard_space": "Keeps the inner TF-coil surface outside the outer CS surface.",
    "tf_peak_field": "Requires peak TF field to remain below the limit [T].",
    "tf_current_density": "Requires TF engineering current density to remain below the material-model allowance.",
    "tf_stress": "Requires the TF structural stress to remain below the limit [Pa].",
    "tf_nuclear_heat": "Requires nuclear heating of cold TF structures to remain below the limit [MW].",
    "pf_equilibrium_ampere_turns": "Requires the PF equilibrium ampere-turn demand to remain below the limit [A-turn].",
    "pf_current_density": "Requires maximum PF engineering current density to remain below its material limit.",
    "pf_peak_field": "Requires the local peak PF field to remain below the limit [T].",
    "pf_stress": "Requires the PF structural-stress proxy to remain below the limit [Pa].",
    "cs_flux_swing": "Requires CS flux-swing capacity to cover startup and flat-top demand.",
    "cs_peak_field": "Requires peak CS field to remain below the limit [T].",
    "cs_current_density": "Requires CS current density to remain below its material limit.",
    "cs_stress": "Requires the CS structural stress to remain below the limit [Pa].",
    "tbr": "Requires the tritium breeding ratio to remain at or above the minimum limit.",
    "tf_fluence": "Requires lifetime TF neutron fluence to remain below the limit.",
    "tf_dpa": "Requires the TF displacement-damage proxy to remain below the limit.",
    "tritium_doubling_time": "Requires tritium inventory doubling time to remain below the limit [yr].",
    "first_wall_heat_load": "Requires first-wall surface heat load to remain below the limit [MW/m²].",
    "divertor_heat_flux": "Requires peak divertor heat flux to remain below the limit [MW/m²].",
    "divertor_target_temperature": "Requires divertor surface temperature to remain below the limit [°C].",
    "divertor_lifetime": "Requires divertor lifetime to remain at or above the minimum limit [FPY].",
    "recirculating_fraction": "Requires the recirculating-power fraction to remain below the maximum limit.",
}

# None means that the constraint has no independent user-set target/limit.
INEQUALITY_VALUE_KIND = {
    "q95": "limit",
    "beta_n": "limit",
    "greenwald": "limit",
    "fusion_gain": "limit",
    "target_net_power": "target",
    "inboard_space": None,
    "tf_peak_field": "limit",
    "tf_current_density": "limit",
    "tf_stress": "limit",
    "tf_nuclear_heat": "limit",
    "pf_equilibrium_ampere_turns": "limit",
    "pf_current_density": None,
    "pf_peak_field": "limit",
    "pf_stress": "limit",
    "cs_flux_swing": None,
    "cs_peak_field": "limit",
    "cs_current_density": None,
    "cs_stress": "limit",
    "tbr": "limit",
    "tf_fluence": "limit",
    "tf_dpa": "limit",
    "tritium_doubling_time": "limit",
    "first_wall_heat_load": "limit",
    "divertor_heat_flux": "limit",
    "divertor_target_temperature": "limit",
    "divertor_lifetime": "limit",
    "recirculating_fraction": "limit",
}

REACTOR_FIELD_SPECS = {
    "energy_closure_mode": ("choice", ("full_space", "self_consistent", "fixed_point"), "Selects the energy-closure solution method."),
    "energy_closure_iterations": ("int", None, "Number of internal fixed-point or self-consistent closure iterations."),
    "plasma_profile_model": ("mapped", {"tokasys": "tokasys", "parabolic": "process_parabolic", "pedestal": "process_pedestal"}, "Selects the plasma density and temperature profile model."),
    "process_profile_alphan": ("float", None, "Density-profile peaking exponent alphan."),
    "process_profile_alphat": ("float", None, "Temperature-profile peaking exponent alphat."),
    "process_profile_tbeta": ("float", None, "Secondary pedestal-temperature exponent tbeta."),
    "process_density_pedestal_radius": ("float", None, "Normalized radius at the top of the density pedestal."),
    "process_temperature_pedestal_radius": ("float", None, "Normalized radius at the top of the temperature pedestal."),
    "process_density_pedestal_greenwald_fraction": ("float", None, "Pedestal density as a fraction of the Greenwald density."),
    "process_density_separatrix_greenwald_fraction": ("float", None, "Separatrix density as a fraction of the Greenwald density."),
    "process_temperature_pedestal_keV": ("float", None, "Pedestal electron temperature [keV]."),
    "process_temperature_separatrix_keV": ("float", None, "Separatrix electron temperature [keV]."),
    "steady_state": ("bool", None, "Enables fully non-inductive steady-state operation."),
    "tf_superconductor_id": ("mapped", {"0 — REBCO": 0, "1 — Nb3Sn": 1}, "Selects the TF-coil superconductor model."),
    "pf_superconductor_id": ("mapped", {"0 — REBCO": 0, "1 — Nb3Sn": 1, "2 — NbTi": 2}, "Selects the PF-coil superconductor model."),
    "cs_superconductor_id": ("mapped", {"0 — REBCO": 0, "1 — Nb3Sn": 1, "2 — NbTi": 2}, "Selects the central-solenoid superconductor model."),
    "blanket_concept_id": ("mapped", {"0 — LiPb/WCLL": 0, "1 — FLiBe": 1, "2 — HCPB": 2}, "Selects the blanket material and concept model."),
    "cost_model_id": ("mapped", {"0 — legacy proxy": 0, "1 — capacity scaling (n=0.6)": 1, "2 — Jo et al. (2021)": 2}, "Selects the plant capital-cost model."),
    "divertor_heat_load_model_id": ("mapped", {"0 — fixed lambda": 0, "1 — Eich-like Bp scaling": 1}, "Selects the divertor heat-load-width model."),
}

REACTOR_DISPLAY_NAMES = {
    name: name.removeprefix("process_") for name in REACTOR_FIELD_SPECS
}

TECHNOLOGY_FIELD_SPECS = {
    "tf_coil_toroidal_pack_fraction": "Fraction of each toroidal pitch available to the TF coil at the inboard leg.",
    "cs_structural_fraction": "Fraction of the central-solenoid radial build assigned to structural material.",
}


def _display_number(value: Any) -> str:
    return f"{float(np.asarray(value)):.12g}"


def _to_jsonable(value: Any) -> Any:
    """Convert nested JAX/NumPy/NamedTuple values to JSON-compatible data."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "_asdict"):
        return {key: _to_jsonable(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        return scalar if isinstance(scalar, (str, bool, int, float)) else float(scalar)
    return array.tolist()


class ToolTip:
    """Small delayed tooltip for Tk and ttk widgets."""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        window = self._window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            window,
            text=self.text,
            justify="left",
            background="#fff8d8",
            relief="solid",
            borderwidth=1,
            padx=7,
            pady=5,
            wraplength=420,
            font=("TkDefaultFont", 10),
        )
        label.pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel()
        if self._window is not None:
            self._window.destroy()
            self._window = None


class OptimizationGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TokaSys Optimization")
        self.root.geometry("1180x900")
        self.root.minsize(1400, 950)

        self.objective_vars: dict[str, tuple[tk.BooleanVar, tk.StringVar]] = {}
        self.design_vars: dict[str, tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = {}
        self.net_power_target_var = tk.StringVar()
        self.equality_vars: dict[str, tuple[tk.BooleanVar, tk.StringVar | None]] = {}
        self.inequality_vars: dict[str, tuple[tk.BooleanVar, tk.StringVar | None]] = {}
        self.reactor_vars: dict[str, tk.Variable] = {}
        self.technology_vars: dict[str, tk.StringVar] = {}
        self.closure_vars: dict[str, tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = {}
        self.tolerance_vars = {
            "equality": tk.StringVar(),
            "inequality": tk.StringVar(),
        }
        self.numerics_vars = {"profile_num_points": tk.StringVar()}
        self.solver_vars = {
            "maxiter": tk.StringVar(value="400"),
            "ftol": tk.StringVar(value="1e-9"),
        }

        self.last_config: dict[str, Any] | None = None
        self.last_problem: Any = None
        self.last_solution: DesignSolution | None = None
        self.last_report: str | None = None
        self.last_plot_path: Path | None = None
        self._preview_temp: tempfile.TemporaryDirectory[str] | None = None
        self._worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._running = False

        self._build_layout()
        self.load_defaults(show_message=False)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_layout(self) -> None:
        shell = ttk.Frame(self.root)
        shell.pack(fill="both", expand=True)

        canvas = tk.Canvas(shell, highlightthickness=0)
        vertical = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        horizontal = ttk.Scrollbar(shell, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        shell.rowconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)

        content = ttk.Frame(canvas, padding=8)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))

        def fit_content(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=max(event.width, content.winfo_reqwidth()))

        canvas.bind("<Configure>", fit_content)
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))
        canvas.bind_all("<Shift-MouseWheel>", lambda event: canvas.xview_scroll(int(-event.delta / 120), "units"))

        # The first column holds the dense design table; columns 2 and 3 are
        # intentionally about two thirds of its width.
        content.columnconfigure(0, weight=9)
        content.columnconfigure(1, weight=6)
        content.columnconfigure(2, weight=6)

        left = ttk.Frame(content)
        middle = ttk.Frame(content)
        right = ttk.Frame(content)
        left.grid(row=0, column=0, padx=5, sticky="new")
        middle.grid(row=0, column=1, padx=5, sticky="new")
        right.grid(row=0, column=2, padx=5, sticky="new")

        self._build_objectives(left).pack(fill="x", pady=(0, 8))
        self._build_design_variables(left).pack(fill="x", pady=(0, 8))
        self._build_equalities(left).pack(fill="x")
        self._build_inequalities(middle).pack(fill="x")
        self._build_reactor_settings(right).pack(fill="x", pady=(0, 8))
        self._build_closure_variables(right).pack(fill="x")

        button_bar = ttk.Frame(self.root, padding=(12, 9))
        button_bar.pack(fill="x")
        for column in range(3):
            button_bar.columnconfigure(column, weight=1)
        self.default_button = ttk.Button(button_bar, text="Default", command=self.load_defaults)
        self.optimize_button = ttk.Button(button_bar, text="Optimize", command=self.optimize)
        self.save_button = ttk.Button(button_bar, text="Save", command=self.save, state="disabled")
        self.default_button.grid(row=0, column=0, padx=7, sticky="ew")
        self.optimize_button.grid(row=0, column=1, padx=7, sticky="ew")
        self.save_button.grid(row=0, column=2, padx=7, sticky="ew")
        ToolTip(self.default_button, "Reloads examples/optimization_reactor_config.json and fills omitted values from default_problem().")
        ToolTip(self.optimize_button, "Validates the current inputs and runs SLSQP in a background thread.")
        ToolTip(self.save_button, "Saves the last config, design, metrics, report, and diagnostic plot under outputs/.")

    @staticmethod
    def _section(parent: tk.Widget, title: str, description: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        ToolTip(frame, description)
        return frame

    @staticmethod
    def _label(parent: tk.Widget, text: str, description: str, row: int, column: int = 0) -> ttk.Label:
        label = ttk.Label(parent, text=text)
        label.grid(row=row, column=column, padx=3, pady=2, sticky="w")
        ToolTip(label, description)
        return label

    def _build_objectives(self, parent: tk.Widget) -> ttk.LabelFrame:
        frame = self._section(parent, "1. Objectives", "SLSQP minimizes the weighted sum of all enabled objective terms.")
        for column, name in enumerate(OBJECTIVE_FUNCTIONS):
            frame.columnconfigure(column, weight=1)
            enabled = tk.BooleanVar()
            weight = tk.StringVar()
            self.objective_vars[name] = (enabled, weight)
            check = ttk.Checkbutton(
                frame,
                text=OBJECTIVE_DISPLAY_NAMES[name],
                variable=enabled,
            )
            check.grid(row=0, column=column, padx=4, pady=(1, 3), sticky="w")
            weight_row = ttk.Frame(frame)
            weight_row.grid(row=1, column=column, padx=4, sticky="w")
            weight_label = ttk.Label(weight_row, text="Weight")
            weight_label.grid(row=0, column=0, padx=(0, 3))
            entry = ttk.Entry(weight_row, textvariable=weight, width=ENTRY_WIDTH)
            entry.grid(row=0, column=1)
            ToolTip(check, OBJECTIVE_DESCRIPTIONS[name])
            ToolTip(weight_label, f"Finite multiplier applied to the {name} objective term.")
            ToolTip(entry, f"Finite multiplier applied to the {name} objective term.")
        return frame

    def _build_design_variables(self, parent: tk.Widget) -> ttk.LabelFrame:
        frame = self._section(parent, "2. Design variables", "Sets the lower bound, upper bound, and initial point for every optimized design variable.")
        for column, text in enumerate(("Variable", "Lower", "Upper", "Initial")):
            ttk.Label(frame, text=text).grid(row=0, column=column, padx=2, sticky="w")
        frame.columnconfigure(0, weight=1)
        for row, name in enumerate(DesignVariables._fields, start=1):
            lower, upper, initial = tk.StringVar(), tk.StringVar(), tk.StringVar()
            self.design_vars[name] = (lower, upper, initial)
            self._label(frame, name, DESIGN_DESCRIPTIONS[name], row)
            for column, variable in enumerate((lower, upper, initial), start=1):
                entry = ttk.Entry(frame, textvariable=variable, width=ENTRY_WIDTH)
                entry.grid(row=row, column=column, padx=2, pady=1)
                ToolTip(entry, f"{('Lower bound', 'Upper bound', 'Initial value')[column - 1]} for {name}.")
        return frame

    def _build_equalities(self, parent: tk.Widget) -> ttk.LabelFrame:
        frame = self._section(parent, "3. Equality constraints", "Forces every enabled normalized equality residual to zero.")
        regular_names = [name for name in ResidualState._fields if name != "target_net_power"]
        for column in range(2):
            frame.columnconfigure(column, weight=1)
        for index, name in enumerate(regular_names):
            enabled = tk.BooleanVar()
            self.equality_vars[name] = (enabled, None)
            row, column = divmod(index, 2)
            check = ttk.Checkbutton(frame, text=name, variable=enabled)
            check.grid(row=row, column=column, padx=3, pady=2, sticky="w")
            ToolTip(check, EQUALITY_DESCRIPTIONS[name])

        target_name = "target_net_power"
        target_enabled = tk.BooleanVar()
        self.equality_vars[target_name] = (target_enabled, self.net_power_target_var)
        target_row = (len(regular_names) + 1) // 2
        target_box = ttk.Frame(frame)
        target_box.grid(row=target_row, column=0, columnspan=2, padx=3, pady=(4, 1), sticky="w")
        target_check = ttk.Checkbutton(target_box, text=target_name, variable=target_enabled)
        target_check.grid(row=0, column=0)
        ttk.Label(target_box, text="Target").grid(row=0, column=1, padx=(8, 3))
        target_entry = ttk.Entry(target_box, textvariable=self.net_power_target_var, width=ENTRY_WIDTH)
        target_entry.grid(row=0, column=2)
        ToolTip(target_check, EQUALITY_DESCRIPTIONS[target_name])
        ToolTip(target_entry, "Net electric power target that the equality must match [MW].")
        return frame

    def _build_inequalities(self, parent: tk.Widget) -> ttk.LabelFrame:
        frame = self._section(parent, "4. Inequality constraints", "Requires every enabled normalized inequality margin to be non-negative.")
        ttk.Label(frame, text="Use").grid(row=0, column=0)
        ttk.Label(frame, text="Constraint").grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="Target / limit").grid(row=0, column=2)
        frame.columnconfigure(1, weight=1)
        for row, name in enumerate(MarginState._fields, start=1):
            enabled = tk.BooleanVar()
            kind = INEQUALITY_VALUE_KIND[name]
            value = (
                self.net_power_target_var
                if name == "target_net_power"
                else tk.StringVar() if kind is not None else None
            )
            self.inequality_vars[name] = (enabled, value)
            check = ttk.Checkbutton(frame, variable=enabled)
            check.grid(row=row, column=0, padx=3)
            self._label(frame, name, INEQUALITY_DESCRIPTIONS[name], row, 1)
            if value is not None:
                entry = ttk.Entry(frame, textvariable=value, width=9)
                entry.grid(row=row, column=2, padx=3, pady=2)
                ToolTip(entry, f"The {kind} applied to {name}.")
            else:
                ttk.Label(frame, text="model-derived").grid(row=row, column=2, padx=3)
            ToolTip(check, INEQUALITY_DESCRIPTIONS[name])
        return frame

    def _build_reactor_settings(self, parent: tk.Widget) -> ttk.LabelFrame:
        frame = self._section(parent, "5. Reactor config, technology, tolerances", "Model selections, technology coefficients, numerical options, and solver controls held fixed during one optimization.")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(frame, text="Reactor config", font=("TkDefaultFont", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 3))
        row += 1
        for name, (kind, choices, description) in REACTOR_FIELD_SPECS.items():
            self._label(frame, REACTOR_DISPLAY_NAMES[name], description, row)
            if kind == "bool":
                variable: tk.Variable = tk.BooleanVar()
                widget: tk.Widget = ttk.Checkbutton(frame, variable=variable)
            elif kind in {"choice", "mapped"}:
                variable = tk.StringVar()
                values = tuple(choices) if kind == "mapped" else choices
                widget = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=19)
            else:
                variable = tk.StringVar()
                widget = ttk.Entry(frame, textvariable=variable, width=20)
            self.reactor_vars[name] = variable
            widget.grid(row=row, column=1, padx=3, pady=2, sticky="ew")
            ToolTip(widget, description)
            row += 1

        ttk.Separator(frame).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        ttk.Label(frame, text="Technology", font=("TkDefaultFont", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        for name, description in TECHNOLOGY_FIELD_SPECS.items():
            variable = self.technology_vars[name] = tk.StringVar()
            self._label(frame, name, description, row)
            entry = ttk.Entry(frame, textvariable=variable, width=20)
            entry.grid(row=row, column=1, padx=3, pady=2, sticky="ew")
            ToolTip(entry, description)
            row += 1

        ttk.Separator(frame).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        ttk.Label(frame, text="Tolerances & numerics", font=("TkDefaultFont", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        fixed_fields = (
            ("equality_tolerance", self.tolerance_vars["equality"], "Residual tolerance used to label an equality as satisfied in reports."),
            ("inequality_tolerance", self.tolerance_vars["inequality"], "Margin tolerance used to label an inequality as satisfied in reports."),
            ("profile_num_points", self.numerics_vars["profile_num_points"], "Number of radial grid points used for plasma-profile calculations."),
            ("SLSQP maxiter", self.solver_vars["maxiter"], "Maximum number of SLSQP iterations."),
            ("SLSQP ftol", self.solver_vars["ftol"], "SLSQP objective convergence tolerance."),
        )
        for name, variable, description in fixed_fields:
            self._label(frame, name, description, row)
            entry = ttk.Entry(frame, textvariable=variable, width=20)
            entry.grid(row=row, column=1, padx=3, pady=2, sticky="ew")
            ToolTip(entry, description)
            row += 1
        return frame

    def _build_closure_variables(self, parent: tk.Widget) -> ttk.LabelFrame:
        frame = self._section(parent, "6. Closure variables", "Sets bounds and initial values for the closure variables solved in the full-space formulation.")
        for column, text in enumerate(("Variable", "Lower", "Upper", "Initial")):
            ttk.Label(frame, text=text).grid(row=0, column=column, padx=2, sticky="w")
        frame.columnconfigure(0, weight=1)
        for row, name in enumerate(ClosureVariables._fields, start=1):
            lower, upper, initial = tk.StringVar(), tk.StringVar(), tk.StringVar()
            self.closure_vars[name] = (lower, upper, initial)
            self._label(frame, name, CLOSURE_DESCRIPTIONS[name], row)
            for column, variable in enumerate((lower, upper, initial), start=1):
                entry = ttk.Entry(frame, textvariable=variable, width=ENTRY_WIDTH)
                entry.grid(row=row, column=column, padx=2, pady=2)
                ToolTip(entry, f"{('Lower bound', 'Upper bound', 'Initial value')[column - 1]} for {name}.")
        return frame

    def load_defaults(self, show_message: bool = True) -> None:
        try:
            raw = load_optimization_config(DEFAULT_CONFIG_PATH)
            problem = apply_optimization_config(default_problem(), DEFAULT_CONFIG_PATH)
        except Exception as exc:
            messagebox.showerror("Default configuration error", str(exc), parent=self.root)
            return

        objective_section = raw.get("objectives", {})
        for name, (enabled_var, weight_var) in self.objective_vars.items():
            entry = objective_section.get(name, {})
            enabled_var.set(bool(entry.get("enabled", False)) if isinstance(entry, dict) else bool(entry))
            default_weight = DEFAULT_OBJECTIVE_WEIGHTS[name]
            weight_var.set(_display_number(entry.get("weight", default_weight) if isinstance(entry, dict) else default_weight))

        for name, (lower_var, upper_var, initial_var) in self.design_vars.items():
            lower_var.set(_display_number(getattr(problem.variable_bounds.lower.design, name)))
            upper_var.set(_display_number(getattr(problem.variable_bounds.upper.design, name)))
            initial_var.set(_display_number(getattr(problem.initial_variables.design, name)))

        # Explicit config values take precedence through apply_optimization_config;
        # omitted closure initial values are filled by default_problem().
        for name, (lower_var, upper_var, initial_var) in self.closure_vars.items():
            lower_var.set(_display_number(getattr(problem.variable_bounds.lower.closure, name)))
            upper_var.set(_display_number(getattr(problem.variable_bounds.upper.closure, name)))
            initial_var.set(_display_number(getattr(problem.initial_variables.closure, name)))

        for name, (enabled_var, target_var) in self.equality_vars.items():
            entry = raw.get("equalities", {}).get(name, {})
            enabled_var.set(bool(entry.get("enabled", False)) if isinstance(entry, dict) else bool(entry))
            if target_var is not None:
                target_var.set(_display_number(entry.get("target", problem.technology.net_electric_target_mw)))

        for name, (enabled_var, value_var) in self.inequality_vars.items():
            entry = raw.get("inequalities", {}).get(name, {})
            enabled_var.set(bool(entry.get("enabled", False)) if isinstance(entry, dict) else bool(entry))
            if value_var is not None:
                kind = INEQUALITY_VALUE_KIND[name]
                if kind not in entry:
                    raise ValueError(f"Default config is missing {kind!r} for inequality {name!r}.")
                value_var.set(_display_number(entry[kind]))

        raw_reactor = raw.get("reactor_config", {})
        for name, variable in self.reactor_vars.items():
            value = raw_reactor.get(name, getattr(problem.config, name))
            kind, choices, _description = REACTOR_FIELD_SPECS[name]
            if kind == "bool":
                variable.set(bool(value))
            elif kind == "mapped":
                display = next(label for label, mapped_value in choices.items() if mapped_value == value)
                variable.set(display)
            else:
                variable.set(str(value) if kind == "choice" else _display_number(value))

        raw_technology = raw.get("technology", {})
        for name, variable in self.technology_vars.items():
            variable.set(_display_number(raw_technology.get(name, getattr(problem.technology, name))))
        self.tolerance_vars["equality"].set(_display_number(problem.optimization.equality_tolerance))
        self.tolerance_vars["inequality"].set(_display_number(problem.optimization.inequality_tolerance))
        self.numerics_vars["profile_num_points"].set(str(problem.numerics.profile_num_points))
        self.solver_vars["maxiter"].set("400")
        self.solver_vars["ftol"].set("1e-9")

        self.last_config = None
        self.last_problem = None
        self.last_solution = None
        self.last_report = None
        self.last_plot_path = None
        self.save_button.configure(state="disabled")
        if show_message:
            print(f"Loaded defaults from {DEFAULT_CONFIG_PATH}", flush=True)

    @staticmethod
    def _finite_float(variable: tk.Variable, label: str) -> float:
        try:
            value = float(variable.get())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: a numeric value is required.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label}: value must be finite.")
        return value

    @staticmethod
    def _positive_int(variable: tk.Variable, label: str) -> int:
        try:
            value = int(variable.get())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: an integer is required.") from exc
        if value <= 0:
            raise ValueError(f"{label}: value must be positive.")
        return value

    def collect_config(self) -> tuple[dict[str, Any], int, float]:
        objectives: dict[str, Any] = {}
        for name, (enabled_var, weight_var) in self.objective_vars.items():
            objectives[name] = {
                "enabled": bool(enabled_var.get()),
                "weight": self._finite_float(weight_var, f"objectives.{name}.weight"),
            }
        if not any(entry["enabled"] for entry in objectives.values()):
            raise ValueError("At least one objective must be enabled.")

        design, bounds = {}, {}
        for name, (lower_var, upper_var, initial_var) in self.design_vars.items():
            lower = self._finite_float(lower_var, f"bounds.{name}.lower")
            upper = self._finite_float(upper_var, f"bounds.{name}.upper")
            initial = self._finite_float(initial_var, f"design.{name}")
            if upper <= lower:
                raise ValueError(f"{name}: upper bound must exceed lower bound.")
            if not lower <= initial <= upper:
                raise ValueError(f"{name}: initial value must lie inside [lower, upper].")
            design[name] = initial
            bounds[name] = {"lower": lower, "upper": upper}

        closure, closure_bounds = {}, {}
        for name, (lower_var, upper_var, initial_var) in self.closure_vars.items():
            lower = self._finite_float(lower_var, f"closure_bounds.{name}.lower")
            upper = self._finite_float(upper_var, f"closure_bounds.{name}.upper")
            initial = self._finite_float(initial_var, f"closure.{name}")
            if upper <= lower:
                raise ValueError(f"{name}: closure upper bound must exceed lower bound.")
            if not lower <= initial <= upper:
                raise ValueError(f"{name}: closure initial value must lie inside [lower, upper].")
            closure[name] = initial
            closure_bounds[name] = {"lower": lower, "upper": upper}

        equalities: dict[str, Any] = {}
        for name, (enabled_var, target_var) in self.equality_vars.items():
            entry: dict[str, Any] = {"enabled": bool(enabled_var.get())}
            if target_var is not None:
                entry["target"] = self._finite_float(target_var, f"equalities.{name}.target")
            equalities[name] = entry

        inequalities: dict[str, Any] = {}
        for name, (enabled_var, value_var) in self.inequality_vars.items():
            entry = {"enabled": bool(enabled_var.get())}
            if value_var is not None:
                kind = INEQUALITY_VALUE_KIND[name]
                entry[kind] = self._finite_float(value_var, f"inequalities.{name}.{kind}")
            inequalities[name] = entry

        reactor_config: dict[str, Any] = {}
        for name, variable in self.reactor_vars.items():
            kind, choices, _description = REACTOR_FIELD_SPECS[name]
            if kind == "bool":
                value: Any = bool(variable.get())
            elif kind == "mapped":
                selected = str(variable.get())
                if selected not in choices:
                    raise ValueError(f"reactor_config.{name}: select one of the listed values.")
                value = choices[selected]
            elif kind == "choice":
                value = str(variable.get())
                if value not in choices:
                    raise ValueError(f"reactor_config.{name}: invalid choice {value!r}.")
            elif kind == "int":
                value = self._positive_int(variable, f"reactor_config.{name}")
            else:
                value = self._finite_float(variable, f"reactor_config.{name}")
            reactor_config[name] = value

        technology = {
            name: self._finite_float(variable, f"technology.{name}")
            for name, variable in self.technology_vars.items()
        }
        equality_tolerance = self._finite_float(self.tolerance_vars["equality"], "tolerances.equality")
        inequality_tolerance = self._finite_float(self.tolerance_vars["inequality"], "tolerances.inequality")
        if equality_tolerance < 0.0 or inequality_tolerance < 0.0:
            raise ValueError("Constraint reporting tolerances must be non-negative.")
        profile_num_points = self._positive_int(self.numerics_vars["profile_num_points"], "numerics.profile_num_points")
        maxiter = self._positive_int(self.solver_vars["maxiter"], "SLSQP maxiter")
        ftol = self._finite_float(self.solver_vars["ftol"], "SLSQP ftol")
        if ftol <= 0.0:
            raise ValueError("SLSQP ftol must be positive.")

        config = {
            "objectives": objectives,
            "reactor_config": reactor_config,
            "technology": technology,
            "numerics": {"profile_num_points": profile_num_points},
            "design": design,
            "closure": closure,
            "bounds": bounds,
            "closure_bounds": closure_bounds,
            "equalities": equalities,
            "inequalities": inequalities,
            "tolerances": {
                "equality": equality_tolerance,
                "inequality": inequality_tolerance,
            },
        }
        return config, maxiter, ftol

    def optimize(self) -> None:
        if self._running:
            return
        try:
            config, maxiter, ftol = self.collect_config()
        except ValueError as exc:
            messagebox.showerror("Invalid optimization input", str(exc), parent=self.root)
            return

        self._running = True
        self.default_button.configure(state="disabled")
        self.optimize_button.configure(state="disabled", text="Optimizing…")
        self.save_button.configure(state="disabled")
        print("\n" + "=" * 78, flush=True)
        print("TokaSys GUI optimization started", flush=True)
        print(f"maxiter={maxiter}, ftol={ftol:g}", flush=True)
        print("SciPy/SLSQP progress follows below.", flush=True)

        worker = threading.Thread(
            target=self._optimization_worker,
            args=(config, maxiter, ftol),
            name="tokasys-optimization",
            daemon=True,
        )
        worker.start()
        self.root.after(100, self._poll_optimization_worker)

    def _optimization_worker(self, config: dict[str, Any], maxiter: int, ftol: float) -> None:
        preview_temp: tempfile.TemporaryDirectory[str] | None = None
        try:
            # Importing plotting initializes Matplotlib; keep that work out of GUI startup.
            from tokasys.diagnostics import plot_reactor_diagnostics

            preview_temp = tempfile.TemporaryDirectory(prefix="tokasys-gui-")
            temp_root = Path(preview_temp.name)
            config_path = temp_root / "optimization_config.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            problem = apply_optimization_config(default_problem(), config_path)
            solution = solve_slsqp(problem, maxiter=maxiter, ftol=ftol, disp=True)
            plot_path = plot_reactor_diagnostics(
                solution.design,
                problem.technology,
                solution.reactor,
                temp_root / "optimization_diagnostics.png",
                title="TokaSys Optimization Diagnostics",
            )
            report = self._format_report(problem, solution)
            print(report, flush=True)
            self._worker_queue.put(
                (
                    "success",
                    (config, problem, solution, report, Path(plot_path), preview_temp),
                )
            )
        except Exception as exc:
            traceback.print_exc()
            if preview_temp is not None:
                preview_temp.cleanup()
            self._worker_queue.put(("error", exc))

    def _poll_optimization_worker(self) -> None:
        """Transfer worker results to Tk's main thread without touching Tk in the worker."""
        try:
            outcome, payload = self._worker_queue.get_nowait()
        except queue.Empty:
            if self._running:
                self.root.after(100, self._poll_optimization_worker)
            return
        if outcome == "success":
            self._optimization_complete(*payload)
        else:
            self._optimization_failed(payload)

    @staticmethod
    def _format_report(problem: Any, solution: DesignSolution) -> str:
        result = solution.reactor
        scipy_result = solution.scipy_result
        lines = [
            "\n--- Optimization result ---",
            f"success: {bool(scipy_result.success)}",
            f"message: {scipy_result.message}",
            f"objective: {float(scipy_result.fun):.8g}",
            f"iterations: {getattr(scipy_result, 'nit', 'n/a')}",
            f"major radius: {float(solution.design.major_radius_m):.6g} m",
            f"aspect ratio: {float(solution.design.aspect_ratio):.6g}",
            f"toroidal field: {float(solution.design.toroidal_field_t):.6g} T",
            f"plasma current: {float(solution.design.plasma_current_ma):.6g} MA",
            f"fusion power: {float(result.plasma.fusion_power_mw):.6g} MW",
            f"net electric power: {float(result.power.net_electric_power_mw):.6g} MW",
            f"total capital cost: {float(result.economics.total_capital_cost_busd):.6g} BUSD",
            f"COE: {float(result.economics.coe_usd_mwh):.6g} USD/MWh",
            "\nObjectives:",
        ]
        for row in objective_reports(solution.variables, result, problem.config, problem.optimization):
            lines.append(
                f"  {row['name']}: actual={row['actual']:.6g} {row['unit']}, "
                f"weight={row['weight']:.6g}, contribution={row['objective_value']:.6g}"
            )
        lines.append("\nEquality constraints:")
        for row in equality_reports(solution.variables, result, problem.technology, problem.config, problem.optimization):
            status = "OK" if row["satisfied"] else "VIOLATED"
            lines.append(
                f"  [{status}] {row['name']}: {row['lhs']:.6g} {row['lhs_label']} vs "
                f"{row['rhs']:.6g} {row['rhs_label']}; residual={row['residual']:.3e}"
            )
        lines.append("\nInequality constraints:")
        for row in inequality_reports(solution.variables, result, problem.technology, problem.optimization):
            status = "OK" if row["satisfied"] else "VIOLATED"
            lines.append(
                f"  [{status}] {row['name']}: {row['actual']:.6g} {row['actual_unit']} "
                f"{row['sense']} {row['limit']:.6g} {row['limit_unit']}; margin={row['margin']:.3e}"
            )
        return "\n".join(lines)

    def _optimization_complete(
        self,
        config: dict[str, Any],
        problem: Any,
        solution: DesignSolution,
        report: str,
        plot_path: Path,
        preview_temp: tempfile.TemporaryDirectory[str],
    ) -> None:
        if self._preview_temp is not None:
            self._preview_temp.cleanup()
        self._preview_temp = preview_temp
        self.last_config = config
        self.last_problem = problem
        self.last_solution = solution
        self.last_report = report
        self.last_plot_path = plot_path
        self._running = False
        self.default_button.configure(state="normal")
        self.optimize_button.configure(state="normal", text="Optimize")
        self.save_button.configure(state="normal")
        self._show_diagnostic_plot(plot_path)
        if not solution.scipy_result.success:
            messagebox.showwarning(
                "Optimization finished without convergence",
                f"SLSQP returned: {solution.scipy_result.message}\n\nThe diagnostic plot and result are still available.",
                parent=self.root,
            )

    def _optimization_failed(self, error: Exception) -> None:
        self._running = False
        self.default_button.configure(state="normal")
        self.optimize_button.configure(state="normal", text="Optimize")
        self.save_button.configure(state="disabled")
        messagebox.showerror("Optimization failed", str(error), parent=self.root)

    def _show_diagnostic_plot(self, plot_path: Path) -> None:
        window = tk.Toplevel(self.root)
        window.title("TokaSys Optimization Diagnostics")
        screen_width = max(window.winfo_screenwidth() - 100, 700)
        screen_height = max(window.winfo_screenheight() - 160, 500)
        with Image.open(plot_path) as source:
            rendered = source.convert("RGB")
            rendered.thumbnail((screen_width, screen_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(rendered)
        label = ttk.Label(window, image=photo)
        label.image = photo
        label.pack(fill="both", expand=True)
        window.transient(self.root)
        window.lift()

    def save(self) -> None:
        if not all((self.last_config, self.last_problem, self.last_solution, self.last_report, self.last_plot_path)):
            messagebox.showinfo("Nothing to save", "Run Optimize before Save.", parent=self.root)
            return
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = OUTPUT_ROOT / f"optimization_{stamp}"
            suffix = 1
            while output_dir.exists():
                output_dir = OUTPUT_ROOT / f"optimization_{stamp}_{suffix:02d}"
                suffix += 1
            output_dir.mkdir(parents=True)

            solution = self.last_solution
            problem = self.last_problem
            result = solution.reactor
            objectives = objective_reports(solution.variables, result, problem.config, problem.optimization)
            equalities = equality_reports(solution.variables, result, problem.technology, problem.config, problem.optimization)
            inequalities = inequality_reports(solution.variables, result, problem.technology, problem.optimization)

            (output_dir / "optimization_config.json").write_text(
                json.dumps(self.last_config, indent=2), encoding="utf-8"
            )
            (output_dir / "optimized_design.json").write_text(
                json.dumps(
                    {
                        "design": _to_jsonable(solution.design),
                        "closure": _to_jsonable(solution.variables.closure),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "geometry": _to_jsonable(result.geometry),
                        "plasma": _to_jsonable(result.plasma),
                        "magnets": _to_jsonable(result.magnets),
                        "nuclear": _to_jsonable(result.nuclear),
                        "exhaust": _to_jsonable(result.exhaust),
                        "power": _to_jsonable(result.power),
                        "economics": _to_jsonable(result.economics),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (output_dir / "optimization_result.json").write_text(
                json.dumps(
                    {
                        "solver": {
                            "success": bool(solution.scipy_result.success),
                            "status": int(solution.scipy_result.status),
                            "message": str(solution.scipy_result.message),
                            "objective": float(solution.scipy_result.fun),
                            "iterations": int(getattr(solution.scipy_result, "nit", -1)),
                            "function_evaluations": int(getattr(solution.scipy_result, "nfev", -1)),
                        },
                        "objectives": _to_jsonable(objectives),
                        "equalities": _to_jsonable(equalities),
                        "inequalities": _to_jsonable(inequalities),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (output_dir / "optimization_report.txt").write_text(self.last_report + "\n", encoding="utf-8")
            shutil.copy2(self.last_plot_path, output_dir / "optimization_diagnostics.png")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return

        print(f"\nSaved optimization artifacts to {output_dir}", flush=True)
        messagebox.showinfo("Optimization saved", f"Saved to:\n{output_dir}", parent=self.root)

    def _close(self) -> None:
        if self._preview_temp is not None:
            self._preview_temp.cleanup()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("aqua" if root.tk.call("tk", "windowingsystem") == "aqua" else "clam")
    except tk.TclError:
        pass
    OptimizationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
