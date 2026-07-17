import json
import os
from pathlib import Path

from tigress_ncr_tools.check_suite import (
    PROBLEM,
    application_exit_code,
    discover_models,
    failure_reason,
    model_status,
    parse_pbs_json,
    parse_sacct,
)
from pathena.hst_reader import read_hst, restart_survivor_indices
from tigress_ncr_tools.plot_suite_hst import (
    histories,
    input_parameter,
    plot_dashboard,
    plot_sfr_grid,
    stellar_surface_density,
    status_color,
    time_range_mask,
    vertical_size,
)


def test_hst_suite_tools(tmp_path):
    assert time_range_mask([199.9, 200.0, 400.0, 600.0, 600.1]).tolist() == [
        False, True, True, True, False,
    ]

    model = tmp_path / "R8_8pc_NCR_Lxy1024_early"
    hst_dir = model / "hst"
    hst_dir.mkdir(parents=True)
    header = "# Athena history dump volume=1.0e+00\n# [1]=time [2]=sfr10 [3]=sfr40 [4]=sfr100 [5]=nmid [6]=Pth_mid [7]=Pturb_mid [8]=msp [9]=Lesc0 [10]=Ltot0 [11]=mass\n#\n"
    rows = (
        "0 1 2 3 4 5 6 7 8 16 2\n"
        "1 2 3 4 5 6 7 8 9 18 3\n"
        "2 30 3 4 5 6 7 8 9 18 4\n"
        "3 40 3 4 5 6 7 8 9 18 5\n"
        "2 300 3 4 5 6 7 8 9 18 40\n"
        "3 400 3 4 5 6 7 8 9 18 50\n"
        "4 500 3 4 5 6 7 8 9 18 60\n"
        "5 3\n"
    )
    path = hst_dir / "R8_8pc_NCR.hst"
    path.write_text(header + rows)
    (model / "model.pbs").write_text(
        "mpiexec athena problem/SurfS=42.5 problem/surf=10 "
        "domain1/x3min=-2048 domain1/x3max=2048\n"
    )
    assert input_parameter(model, "SurfS") == 42.5
    assert vertical_size(model) == 4096.0
    (tmp_path / "restart_files").mkdir()
    assert discover_models(tmp_path) == [model]
    assert stellar_surface_density(model, [1.0, 2.0], 0.5).tolist() == [
        43.0, 43.5,
    ]
    data = read_hst(path)
    assert data["time"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert data["sfr10"].tolist() == [1.0, 2.0, 300.0, 400.0, 500.0]
    assert data["vol"] == 1.0
    raw = read_hst(path, prune_restarts=False)
    assert raw["time"].tolist() == [0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
    assert restart_survivor_indices(
        [0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.5, 2.5]
    ).tolist() == [0, 1, 7, 8]

    job = parse_sacct(
        "123|R8_8pc_NCR_Lxy1024_early|COMPLETED|0:0|01:02:03\n"
    )[model.name]
    (model / "err.txt").write_text("### Fatal error: mass cannot be negative!\n")
    assert model_status(model, job)[0] == "FAILED"
    assert {"FAILED", "STALLED", "TIMEOUT"} <= PROBLEM
    assert status_color("FAILED") == "red"
    assert status_color("RUNNING") == "tab:blue"
    assert status_color("PENDING") == "tab:orange"
    assert status_color("COMPLETE") == "tab:green"

    slurm_out = model / "ncr-123.out"
    slurm_out.write_text("EXITCODE = 0\npost-processing failed\n")
    assert application_exit_code(model) == 0
    assert model_status(model, job, reason="", app_exit=0)[0] == "COMPLETE"

    running = parse_sacct(
        "124|R8_8pc_NCR_Lxy1024_early|RUNNING|0:0|03:00:00|2026-07-13T10:00:00\n"
    )[model.name]
    assert running["start"] is not None
    assert model_status(
        model, running, reason="", progress=running["start"] + 100.0,
        now=running["start"] + 10000.0,
        stale_seconds=7200.0,
    )[0] == "STALLED"
    assert model_status(
        model, running, reason="segmentation fault",
        progress=running["start"] + 9999.0,
        now=running["start"] + 10000.0,
    )[0] == "CRASHED"

    os.utime(model / "err.txt", (100.0, 100.0))
    assert failure_reason(model, since=200.0) == ""

    pbs_output = model / "model.o123"
    pbs_output.write_text("EXITCODE = 0\npost-processing failed\n")
    pbs = parse_pbs_json(json.dumps({
        "Jobs": {
            "123.pbs": {
                "Job_Name": model.name,
                "job_state": "F",
                "Exit_status": 1,
                "qtime": "Thu Jul 16 20:47:57 2026",
                "stime": "Thu Jul 16 20:48:00 2026",
                "resources_used": {"walltime": "00:01:00"},
                "Output_Path": f"host:{pbs_output}",
            }
        }
    }))[model.name]
    assert pbs["state"] == "FAILED"
    assert pbs["elapsed"] == "00:01:00"
    assert application_exit_code(model, job=pbs) == 0
    assert model_status(model, pbs, reason="", app_exit=0)[0] == "COMPLETE"

    output = tmp_path / "plots"
    output.mkdir()
    models = histories(tmp_path)
    plot_dashboard(models, output / "hst_summary.png")
    plot_sfr_grid(models, output / "hst_sfr_grid.png")
    assert (output / "hst_summary.png").is_file()
    assert (output / "hst_sfr_grid.png").is_file()
