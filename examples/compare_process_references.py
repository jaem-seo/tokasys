from __future__ import annotations

import argparse
from pathlib import Path

import jax

from tokasys.validation import (
    PROCESS_SCENARIO_INPUTS,
    build_reference_from_mfile,
    compare_reference_suite,
    format_reference_comparison,
    format_reference_suite,
    generate_process_input,
    run_process,
    write_reference_json,
)
from tokasys.validation.process_runner import DEFAULT_PROCESS_EXECUTABLE


jax.config.update("jax_enable_x64", True)


LOCAL_PROCESS_SCENARIOS = tuple(PROCESS_SCENARIO_INPUTS) + ("iter_baseline",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TokaSys evaluations against PROCESS reference cases.",
    )
    parser.add_argument(
        "references",
        nargs="*",
        type=Path,
        help="Optional reference-case JSON files. If omitted, bundled cases are used.",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=None,
        help="Directory containing PROCESS reference-case JSON files.",
    )
    parser.add_argument(
        "--full-space",
        action="store_true",
        help="Evaluate using full-space closure variables instead of design-only inputs.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print the full metric table for each case.",
    )
    parser.add_argument(
        "--plasma-profile-model",
        choices=("tokasys", "process_parabolic", "process_pedestal"),
        default=None,
        help="Override ReactorConfig.plasma_profile_model for all compared cases.",
    )
    parser.add_argument(
        "--scenario",
        choices=LOCAL_PROCESS_SCENARIOS,
        default=None,
        help="Generate and compare a local PROCESS scenario input.",
    )
    parser.add_argument(
        "--run-process",
        action="store_true",
        help="Run local PROCESS before comparing. Requires PROCESS to be installed.",
    )
    parser.add_argument(
        "--process-one-shot",
        action="store_true",
        help=(
            "Generate a PROCESS input with equality closure disabled "
            "(neqns=0 and no active ixc variables) before running PROCESS."
        ),
    )
    parser.add_argument(
        "--process-input",
        type=Path,
        default=None,
        help="PROCESS IN.DAT file to run. If omitted with --scenario iter_baseline, a generated input is used.",
    )
    parser.add_argument(
        "--process-mfile",
        type=Path,
        default=None,
        help="Existing PROCESS MFILE.DAT to convert and compare without running PROCESS.",
    )
    parser.add_argument(
        "--process-executable",
        type=Path,
        default=DEFAULT_PROCESS_EXECUTABLE,
        help="Path to the local process executable.",
    )
    parser.add_argument(
        "--process-workdir",
        type=Path,
        default=None,
        help="Directory for generated PROCESS inputs/outputs.",
    )
    parser.add_argument(
        "--write-reference-json",
        type=Path,
        default=None,
        help="Optional path to write the generated PROCESS reference JSON.",
    )
    parser.add_argument(
        "--include-process-cost",
        action="store_true",
        help="Include PROCESS cost/COE metrics for generated local references.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_reference = None
    if args.scenario is not None or args.run_process or args.process_mfile is not None:
        scenario = args.scenario or "local_process"
        workdir = args.process_workdir or Path("process_runs") / scenario
        if args.process_mfile is not None and not args.run_process:
            mfile = args.process_mfile
            process_input = args.process_input
        else:
            if args.process_input is None:
                if args.scenario == "iter_baseline":
                    process_input = generate_process_input(
                        workdir / "ITER_baseline.IN.DAT",
                        one_shot=args.process_one_shot,
                    )
                elif args.scenario in PROCESS_SCENARIO_INPUTS:
                    process_input = generate_process_input(
                        workdir / PROCESS_SCENARIO_INPUTS[args.scenario].name,
                        base_input=PROCESS_SCENARIO_INPUTS[args.scenario],
                        overlay=None,
                        one_shot=args.process_one_shot,
                    )
                else:
                    raise ValueError(
                        "--process-input is required unless --scenario is one of "
                        f"{LOCAL_PROCESS_SCENARIOS}."
                    )
            else:
                if args.process_one_shot:
                    process_input = generate_process_input(
                        workdir / args.process_input.name,
                        base_input=args.process_input,
                        overlay=None,
                        one_shot=True,
                    )
                else:
                    process_input = args.process_input
            mfile = run_process(
                process_input,
                workdir=workdir,
                process_executable=args.process_executable,
            )
        generated_reference = build_reference_from_mfile(
            mfile,
            scenario=scenario,
            input_file=process_input,
            include_cost_metrics=args.include_process_cost,
        )
        if args.write_reference_json is not None:
            write_reference_json(generated_reference, args.write_reference_json)

    paths = args.references if args.references else None
    config_overrides = {}
    if args.plasma_profile_model is not None:
        config_overrides["plasma_profile_model"] = args.plasma_profile_model
    if generated_reference is None:
        suite = compare_reference_suite(
            paths=paths,
            directory=args.directory,
            use_full_space=args.full_space,
            config_overrides=config_overrides or None,
        )
    else:
        suite = compare_reference_suite(
            references=[generated_reference],
            use_full_space=args.full_space,
            config_overrides=config_overrides or None,
        )
    print(format_reference_suite(suite))
    if args.details:
        for case_report in suite["cases"]:
            print()
            print(format_reference_comparison(case_report))


if __name__ == "__main__":
    main()
