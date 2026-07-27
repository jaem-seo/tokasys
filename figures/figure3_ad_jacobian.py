"""Generate Figure 3: AD-based reactor-performance sensitivities.

This script uses the same ITER-scale baseline as ``examples/run_baseline.py``.
It evaluates the Jacobian of net electric power and cost of electricity (COE)
with respect to five selected design variables using JAX automatic
differentiation.

The plotted quantity is a dimensionless, bound-scaled sensitivity,

    S_ij = (x_j,max - x_j,min) / |y_i,0| * d y_i / d x_j,

where y_i,0 is the baseline output. This normalization allows parameters with
different physical units to be compared in a single bar chart. The script also
prints the raw Jacobian in physical units.

Place this file in the project root or in ``examples/`` and run, for example,

    python examples/figure3_ad_jacobian.py

The output files are written next to this script unless ``--output-dir`` is
specified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tokasys import default_problem, evaluate_reactor


jax.config.update("jax_enable_x64", True)


# Selected design parameters for Figure 3.
PARAMETERS = (
    ("major_radius_m", r"$R_0$"),
    ("toroidal_field_t", r"$B_t$"),
    ("plasma_current_ma", r"$I_p$"),
    #("auxiliary_heating_mw", r"$P_{\mathrm{aux}}$"),
    ("current_drive_power_mw", r"$P_{\mathrm{CD}}$"),
)

OUTPUT_NAMES = ("net_electric_power_mw", "coe_usd_mwh")
OUTPUT_LABELS = (
    r"$P_{\mathrm{net}}$",
    r"COE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the AD Jacobian of net power and COE at the baseline design."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG, PDF, and CSV outputs. Defaults to the script directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of the PNG output.",
    )
    return parser.parse_args()


def selected_vector(design) -> jnp.ndarray:
    """Pack the five selected physical design variables into one vector."""
    return jnp.asarray([getattr(design, name) for name, _ in PARAMETERS])


def replace_selected_variables(reference_design, values: jnp.ndarray):
    """Return a DesignVariables object with the selected fields replaced.

    The list construction is static during tracing, so this remains compatible
    with JAX transformations.
    """
    fields = list(reference_design)
    field_indices = {
        name: reference_design._fields.index(name) for name, _ in PARAMETERS
    }
    for value, (name, _) in zip(values, PARAMETERS):
        fields[field_indices[name]] = value
    return type(reference_design)(*fields)


def make_output_function(problem):
    """Create the two-output reactor evaluation used by ``jax.jacrev``."""
    reference_design = problem.initial_design

    def outputs(selected_values: jnp.ndarray) -> jnp.ndarray:
        design = replace_selected_variables(reference_design, selected_values)
        result = evaluate_reactor(
            design,
            problem.technology,
            problem.config,
            problem.numerics,
        )
        return jnp.asarray(
            [
                result.power.net_electric_power_mw,
                result.economics.coe_usd_mwh,
            ]
        )

    return outputs


def compute_jacobian(problem):
    """Evaluate baseline outputs and their full 2-by-5 AD Jacobian."""
    x0 = selected_vector(problem.initial_design)
    output_fn = make_output_function(problem)

    # jacrev returns both output gradients with respect to all selected inputs.
    output_and_jacobian = jax.jit(
        lambda values: (output_fn(values), jax.jacrev(output_fn)(values))
    )
    y0, jacobian = output_and_jacobian(x0)

    lower = selected_vector(problem.bounds.lower)
    upper = selected_vector(problem.bounds.upper)
    parameter_spans = upper - lower

    # Dimensionless bound-scaled Jacobian. The absolute value in the output
    # scale preserves the physical sign even for a potentially negative output.
    output_scale = jnp.maximum(jnp.abs(y0), 1.0e-12)
    #normalized_jacobian = jacobian * parameter_spans[None, :] / output_scale[:, None]
    normalized_jacobian = jacobian * x0 / output_scale[:, None]

    return (
        np.asarray(x0, dtype=float),
        np.asarray(y0, dtype=float),
        np.asarray(jacobian, dtype=float),
        np.asarray(normalized_jacobian, dtype=float),
        np.asarray(parameter_spans, dtype=float),
    )


def annotate_bars(ax, bars, values: np.ndarray) -> None:
    """Add compact values above or below bars."""
    largest = max(float(np.max(np.abs(values))), 1.0)
    offset = 0.025 * largest
    for bar, value in zip(bars, values):
        y = value + offset if value >= 0.0 else value - offset
        va = "bottom" if value >= 0.0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=8,
        )


def plot_figure(normalized_jacobian: np.ndarray, output_path: Path, dpi: int) -> None:
    labels = [label for _, label in PARAMETERS]
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True)

    panel_data = (
        (
            normalized_jacobian[0],
            r"Normalized Jacobian of $P_{\mathrm{net}}$",
            #"(a)",
            "",
        ),
        (
            normalized_jacobian[1],
            r"Normalized Jacobian of COE",
            #"(b)",
            "",
        ),
    )

    for ax, (values, ylabel, panel_label) in zip(axes, panel_data):
        bars = ax.bar(positions, values, width=0.68)
        ax.axhline(0.0, color='k', linestyle="--", linewidth=0.9)
        ax.set_xticks(positions, labels)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Sampled design parameters")
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.02,
            0.96,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        annotate_bars(ax, bars, values)

        # Add enough room for numerical labels without forcing symmetric limits.
        ymin, ymax = ax.get_ylim()
        padding = 0.10 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - padding, ymax + padding)

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_csv(
    csv_path: Path,
    x0: np.ndarray,
    y0: np.ndarray,
    raw_jacobian: np.ndarray,
    normalized_jacobian: np.ndarray,
    parameter_spans: np.ndarray,
) -> None:
    units = ("m", "T", "MA", "MW", "MW")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "parameter",
                "baseline_value",
                "parameter_unit",
                "bound_span",
                "dPnet_dx_raw",
                "dCOE_dx_raw",
                "Pnet_bound_scaled_sensitivity",
                "COE_bound_scaled_sensitivity",
            ]
        )
        for index, ((name, _), unit) in enumerate(zip(PARAMETERS, units)):
            writer.writerow(
                [
                    name,
                    x0[index],
                    unit,
                    parameter_spans[index],
                    raw_jacobian[0, index],
                    raw_jacobian[1, index],
                    normalized_jacobian[0, index],
                    normalized_jacobian[1, index],
                ]
            )

        writer.writerow([])
        writer.writerow(["baseline_net_electric_power_mw", y0[0]])
        writer.writerow(["baseline_coe_usd_mwh", y0[1]])


def print_results(
    x0: np.ndarray,
    y0: np.ndarray,
    raw_jacobian: np.ndarray,
    normalized_jacobian: np.ndarray,
) -> None:
    names = [name for name, _ in PARAMETERS]
    print("=== Baseline outputs ===")
    print(f"Net electric power : {y0[0]:.6f} MW")
    print(f"COE                : {y0[1]:.6f} USD/MWh")

    print("\n=== Selected baseline variables ===")
    for name, value in zip(names, x0):
        print(f"{name:<30s} {value: .8g}")

    print("\n=== Raw AD Jacobian ===")
    print("Rows: [P_net (MW), COE (USD/MWh)]")
    print("Cols:", names)
    print(raw_jacobian)

    print("\n=== Bound-scaled normalized Jacobian ===")
    print("S_ij = (x_j,max - x_j,min) / |y_i,0| * dy_i/dx_j")
    print(normalized_jacobian)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = default_problem()
    x0, y0, raw_jacobian, normalized_jacobian, spans = compute_jacobian(problem)

    output_stem = output_dir / "figure3_ad_jacobian"
    plot_figure(normalized_jacobian, output_stem, args.dpi)
    write_csv(
        output_stem.with_suffix(".csv"),
        x0,
        y0,
        raw_jacobian,
        normalized_jacobian,
        spans,
    )
    print_results(x0, y0, raw_jacobian, normalized_jacobian)

    print("\nSaved outputs:")
    print(f"  {output_stem.with_suffix('.png')}")
    print(f"  {output_stem.with_suffix('.pdf')}")
    print(f"  {output_stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
