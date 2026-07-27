import math

from tokasys.validation import (
    build_reference_from_mfile,
    bundled_reference_case_paths,
    build_process_reference_problem,
    compare_reference_suite,
    compare_to_reference,
    failed_metrics,
    format_reference_comparison,
    format_reference_suite,
    load_reference_cases,
    load_reference_case,
    parse_mfile,
)
from tokasys.validation.process_runner import generate_process_input


def test_process_reference_case_schema_loads():
    reference = load_reference_case()
    bundled_paths = bundled_reference_case_paths()
    references = load_reference_cases()

    assert reference["case_id"] == "process_large_tokamak_eval_dbd37a8f"
    assert bundled_paths
    assert references[0]["case_id"] == reference["case_id"]
    assert reference["source"]["code"] == "PROCESS"
    assert reference["source"]["scenario"] == "large_tokamak_eval"
    assert reference["source"]["reference_mfile"].endswith(
        "large_tokamak_eval_MFILE_dbd37a8ffdc88959c0fe604e2420f40d3d9833bd.DAT"
    )
    assert reference["metrics"]
    assert all("path" in metric for metric in reference["metrics"])
    assert all("reference" in metric for metric in reference["metrics"])
    assert all("process_variable" in metric for metric in reference["metrics"])
    net_metric = next(
        metric
        for metric in reference["metrics"]
        if metric["process_variable"] == "p_plant_electric_net_mw"
    )
    assert net_metric["path"] == "power.flat_top_net_electric_power_mw"


def test_process_reference_problem_uses_actual_case_mapping():
    reference = load_reference_case()
    problem = build_process_reference_problem(reference)

    assert float(problem.initial_design.major_radius_m) == 8.0
    assert float(problem.initial_design.aspect_ratio) == 3.0
    assert float(problem.initial_design.plasma_current_ma) == 16.091095408042267
    assert float(problem.technology.net_electric_target_mw) == 400.0
    assert float(problem.technology.thermal_efficiency) == 0.4
    assert problem.config.steady_state is False
    assert problem.config.energy_closure_mode == "full_space"


def test_generate_process_input_can_disable_process_closure(tmp_path):
    base = tmp_path / "base.IN.DAT"
    overlay = tmp_path / "overlay.IN.DAT"
    output = tmp_path / "generated.IN.DAT"
    base.write_text(
        "\n".join(
            [
                "ioptimz = -2",
                "neqns = 2",
                "icc = 1 * Beta",
                "icc = 2 * Global power balance",
                "ixc = 4 * temperature",
                "ixc = 6 * density",
                "rmajor = 6.2",
            ]
        ),
        encoding="utf-8",
    )
    overlay.write_text("q95 = 3.0\n", encoding="utf-8")

    generate_process_input(output, base_input=base, overlay=overlay, one_shot=True)

    generated = output.read_text(encoding="utf-8")
    assert "neqns = 0" in generated
    assert "ixc =" not in generated
    assert "icc = 1" in generated
    assert "q95 = 3.0" in generated


def test_process_reference_problem_accepts_closure_mapping():
    reference = load_reference_case()
    reference["tokasys_problem"]["closure"] = {
        "stored_energy_mj": 900.0,
        "transport_loss_mw": 310.0,
    }
    problem = build_process_reference_problem(reference)

    assert float(problem.initial_variables.closure.stored_energy_mj) == 900.0
    assert float(problem.initial_variables.closure.transport_loss_mw) == 310.0


def test_actual_process_reference_comparison_reports_metrics():
    report = compare_to_reference()
    full_space_report = compare_to_reference(use_full_space=True)
    suite = compare_reference_suite()

    assert report["case_id"] == "process_large_tokamak_eval_dbd37a8f"
    assert report["num_passed"] + report["num_failed"] == len(report["metrics"])
    assert report["num_passed"] >= 1
    assert full_space_report["case_id"] == report["case_id"]
    assert full_space_report["metrics"] != report["metrics"]
    assert failed_metrics(report) == [row for row in report["metrics"] if not row["passed"]]
    assert all(math.isfinite(row["value"]) for row in report["metrics"])
    assert all(math.isfinite(row["reference"]) for row in report["metrics"])
    assert "process_large_tokamak_eval_dbd37a8f" in format_reference_comparison(report)
    assert suite["num_cases"] >= 1
    assert suite["num_metrics"] >= len(report["metrics"])
    assert "PROCESS reference suite" in format_reference_suite(suite)


def test_large_tokamak_cost_metrics_are_within_ten_percent():
    report = compare_to_reference()
    metrics = {row["name"]: row for row in report["metrics"]}

    assert abs(metrics["capital_cost"]["relative_error"]) < 0.10
    assert abs(metrics["coe"]["relative_error"]) < 0.10


def test_process_reference_suite_accepts_multiple_cases():
    reference = load_reference_case()
    second = {
        **reference,
        "case_id": "process_large_tokamak_eval_copy",
        "description": "Copy used to test multi-case suite aggregation.",
    }
    suite = compare_reference_suite(references=[reference, second])

    assert suite["num_cases"] == 2
    assert len(suite["cases"]) == 2
    assert suite["cases"][0]["case_id"] == reference["case_id"]
    assert suite["cases"][1]["case_id"] == "process_large_tokamak_eval_copy"
    assert suite["num_metrics"] == 2 * len(reference["metrics"])


def test_low_aspect_ratio_power_balance_matches_process_accounting():
    low_reference = next(
        reference
        for reference in load_reference_cases()
        if reference["source"]["scenario"] == "low_aspect_ratio_DEMO"
    )
    report = compare_to_reference(reference=low_reference)
    metrics = {row["name"]: row for row in report["metrics"]}

    assert abs(metrics["gross_electric_power"]["relative_error"]) < 0.01
    assert abs(metrics["net_electric_power"]["relative_error"]) < 0.02
    assert abs(metrics["recirculating_fraction"]["relative_error"]) < 0.02


def test_local_process_mfile_can_build_reference(tmp_path):
    mfile = tmp_path / "MFILE.DAT"
    mfile.write_text(
        "\n".join(
            [
                'PROCESS_version__________________________________________________________ (procver)______________________ "local"',
                'PROCESS_run_title________________________________________________________ (runtitle)_____________________ "synthetic"',
                "Major_radius_(R0)_(m)___________________________________________________ (rmajor)_______________________ 6.20000000000000000e+00",
                "Aspect_ratio_(A)________________________________________________________ (aspect)_______________________ 3.10000000000000000e+00",
                "Elongation______________________________________________________________ (kappa)________________________ 1.70000000000000000e+00",
                "Triangularity___________________________________________________________ (triang)_______________________ 3.30000000000000000e-01",
                "Vacuum_toroidal_field_at_R0_(T)________________________________________ (b_plasma_toroidal_on_axis)____ 5.30000000000000000e+00",
                "Plasma_current_(Ip)_(MA)_______________________________________________ (plasma_current_MA)____________ 1.50000000000000000e+01",
                "Volume_averaged_electron_number_density________________________________ (nd_plasma_electrons_vol_avg)__ 8.00000000000000000e+19",
                "Volume_averaged_electron_temperature___________________________________ (temp_plasma_electron_vol_avg_kev)_ 1.00000000000000000e+01",
                "Safety_factor_at_95%_flux_surface______________________________________ (q95)__________________________ 3.00000000000000000e+00",
                "Volume_averaged_total_plasma_beta______________________________________ (beta_total_vol_avg)___________ 2.50000000000000014e-02",
                "Total_fusion_power_(MW)________________________________________________ (p_fusion_total_mw)____________ 5.00000000000000000e+02",
                "Auxiliary_heat_power_(MW)______________________________________________ (p_hcd_primary_extra_heat_mw)___ 5.00000000000000000e+01",
                "Current_drive_power_(MW)_______________________________________________ (p_hcd_injected_current_total_mw)_ 1.00000000000000000e+01",
                "TF_peak_toroidal_field_upper_limit_____________________________________ (ineq_value_con025)____________ 1.10000000000000000e+01",
                "Total_flux_consumption_for_plasma_current_ramp-up______________________ (vs_plasma_ramp_required)______ 1.50000000000000000e+02",
                "Inboard_first_wall_radial_thickness_(m)________________________________ (dr_fw_inboard)________________ 5.00000000000000028e-02",
                "Outboard_first_wall_radial_thickness_(m)_______________________________ (dr_fw_outboard)_______________ 5.00000000000000028e-02",
                "Inboard_blanket_radial_thickness_(m)___________________________________ (dr_blkt_inboard)______________ 8.50000000000000000e-01",
                "Outboard_blanket_radial_thickness_(m)__________________________________ (dr_blkt_outboard)_____________ 8.50000000000000000e-01",
                "Inner_radiation_shield_radial_thickness_(m)____________________________ (dr_shld_inboard)______________ 5.50000000000000044e-01",
                "Outer_radiation_shield_radial_thickness_(m)____________________________ (dr_shld_outboard)_____________ 5.50000000000000044e-01",
                "TF_coil_inboard_leg_radial_thickness_(m)_______________________________ (dr_tf_inboard)________________ 1.20000000000000000e+00",
                "Central_solenoid_radial_thickness_(m)__________________________________ (dr_cs)________________________ 8.00000000000000044e-01",
                "Thermal_to_electric_conversion_efficiency______________________________ (eta_turbine)__________________ 4.00000000000000022e-01",
                "Required_net_electric_power____________________________________________ (p_plant_electric_net_required_mw)_ 1.00000000000000000e+00",
            ]
        ),
        encoding="utf-8",
    )

    values = parse_mfile(mfile)
    reference = build_reference_from_mfile(
        mfile,
        scenario="synthetic",
        include_cost_metrics=False,
    )

    assert values["rmajor"] == 6.2
    assert reference["case_id"] == "process_synthetic_local"
    assert reference["tokasys_problem"]["design"]["major_radius_m"] == 6.2
    assert reference["tokasys_problem"]["config"]["plasma_profile_model"] == (
        "process_pedestal"
    )
    assert any(metric["name"] == "fusion_power" for metric in reference["metrics"])
