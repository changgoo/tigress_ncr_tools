import os
from pathlib import Path

from tigress_ncr_tools.check_suite import PROBLEM, failure_reason, model_status, parse_sacct
from pathena.hst_reader import read_hst
from tigress_ncr_tools.plot_suite_hst import histories, plot_dashboard, plot_sfr_grid


def test_hst_suite_tools(tmp_path):
    model = tmp_path / "R8_8pc_NCR_row0002"
    hst_dir = model / "hst"
    hst_dir.mkdir(parents=True)
    header = "# Athena history dump volume=1.0e+00\n# [1]=time [2]=sfr10 [3]=sfr40 [4]=sfr100 [5]=nmid [6]=Pth_mid [7]=Pturb_mid [8]=msp [9]=Lesc0 [10]=Ltot0 [11]=mass\n#\n"
    rows = "0 1 2 3 4 5 6 7 8 16 2\n1 2 3 4 5 6 7 8 9 18 3\n2 3\n"
    path = hst_dir / "R8_8pc_NCR.hst"
    path.write_text(header + rows)
    data = read_hst(path)
    assert data["time"].tolist() == [0.0, 1.0]
    assert data["vol"] == 1.0

    job = parse_sacct("123|R8_8pc_NCR_row0002|COMPLETED|0:0|01:02:03\n")[model.name]
    (model / "err.txt").write_text("### Fatal error: mass cannot be negative!\n")
    assert model_status(model, job)[0] == "FAILED"
    assert {"FAILED", "STALLED", "TIMEOUT"} <= PROBLEM

    running = parse_sacct(
        "124|R8_8pc_NCR_row0002|RUNNING|0:0|03:00:00|2026-07-13T10:00:00\n"
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

    output = tmp_path / "plots"
    output.mkdir()
    models = histories(tmp_path)
    plot_dashboard(models, output / "hst_summary.png")
    plot_sfr_grid(models, output / "hst_sfr_grid.png")
    assert (output / "hst_summary.png").is_file()
    assert (output / "hst_sfr_grid.png").is_file()
