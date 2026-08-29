from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_tracer_correlations import (
    PDF_PHASE_SPECS,
    PDF_PHASE_VELOCITY_SPECS,
    PDF_METRICS,
    SPECTRUM_METRICS,
    TRACERS,
    load_tracer_summaries,
    parameter_correlation_table,
    phase_velocity_correlation_table,
    render_suite_tracer_correlations,
    tracer_correlation_table,
)


def _write_tracer_summaries(suite):
    models = ("model_b", "model_a", "model_c")
    sfr = (2.0e-3, 1.0e-3, 4.0e-3)
    for tracer_index, tracer in enumerate(TRACERS, start=1):
        rows = []
        for model_index, (model, mean_sfr) in enumerate(
            zip(models, sfr), start=1
        ):
            row = {
                "model": model,
                "mean_sfr10": mean_sfr,
                "stellar_surface_density": 10.0 * model_index,
                "stellar_scale_height": 100.0 * model_index,
                "omega": 0.01 * model_index,
                "qshear": 0.5 * model_index,
                "kappa": 0.015 * model_index,
                "stellar_midplane_density": 0.05 * model_index,
                "average_start": 200.0,
                "average_stop": 600.0,
            }
            for metric in (*PDF_METRICS, *SPECTRUM_METRICS):
                center = float(tracer_index * model_index)
                row[metric.median] = center
                row[metric.percentile16] = 0.8 * center
                row[metric.percentile84] = 1.2 * center
            rows.append(row)
        frame = pd.DataFrame(rows)
        for relative in (tracer.pdf_summary, tracer.spectrum_summary):
            path = suite / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
    phase_rows = []
    for phase, _ in PDF_PHASE_SPECS:
        for model_index, model in enumerate(models, start=1):
            row = {
                "model": model,
                "phase": phase,
                "average_start": 400.0,
                "average_stop": 600.0,
            }
            for velocity, _ in PDF_PHASE_VELOCITY_SPECS:
                row[f"{velocity}_time_median"] = float(model_index)
                row[f"{velocity}_time_percentile16"] = 0.8 * model_index
                row[f"{velocity}_time_percentile84"] = 1.2 * model_index
            phase_rows.append(row)
    phase_path = (
        suite
        / "phase_evolution_zprof"
        / "phase_correlation_model_summary.csv"
    )
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(phase_rows).to_csv(phase_path, index=False)


def test_tracer_summaries_align_and_correlate(tmp_path):
    _write_tracer_summaries(tmp_path)
    summaries = load_tracer_summaries(tmp_path, "pdf")
    assert summaries["hi"]["model"].tolist() == [
        "model_b",
        "model_a",
        "model_c",
    ]
    correlations = tracer_correlation_table(
        summaries, PDF_METRICS, "pdf"
    )
    np.testing.assert_allclose(correlations["spearman_rho"], 1.0)
    parameter_correlations = parameter_correlation_table(
        summaries, PDF_METRICS, "pdf"
    )
    assert len(parameter_correlations) == 42
    sfr_rows = parameter_correlations["parameter"] == "mean_sfr10"
    np.testing.assert_allclose(
        parameter_correlations.loc[~sfr_rows, "spearman_rho"], 1.0
    )
    np.testing.assert_allclose(
        parameter_correlations.loc[sfr_rows, "spearman_rho"], 0.5
    )
    phase_summary = pd.read_csv(
        tmp_path
        / "phase_evolution_zprof"
        / "phase_correlation_model_summary.csv"
    )
    phase_correlations = phase_velocity_correlation_table(
        summaries, phase_summary
    )
    assert len(phase_correlations) == 216
    np.testing.assert_allclose(phase_correlations["spearman_rho"], 1.0)


def test_render_suite_tracer_correlations(tmp_path):
    _write_tracer_summaries(tmp_path)
    output = tmp_path / "comparison"
    render_suite_tracer_correlations(tmp_path, output_dir=output, dpi=40)
    assert (output / "tracer_pdf_width_correlations.png").is_file()
    assert (output / "tracer_power_spectrum_correlations.png").is_file()
    assert (
        output / "tracer_pdf_width_parameter_correlation_matrix.png"
    ).is_file()
    assert (
        output / "tracer_power_spectrum_parameter_correlation_matrix.png"
    ).is_file()
    assert (
        output / "tracer_pdf_width_phase_velocity_correlation_matrix.png"
    ).is_file()
    table = pd.read_csv(output / "tracer_correlation_coefficients.csv")
    assert len(table) == 15
    np.testing.assert_allclose(table["spearman_rho"], 1.0)
    parameter_table = pd.read_csv(
        output / "tracer_parameter_correlation_coefficients.csv"
    )
    assert len(parameter_table) == 105
    sfr_rows = parameter_table["parameter"] == "mean_sfr10"
    np.testing.assert_allclose(parameter_table.loc[~sfr_rows, "spearman_rho"], 1.0)
    np.testing.assert_allclose(parameter_table.loc[sfr_rows, "spearman_rho"], 0.5)
    phase_table = pd.read_csv(
        output
        / "tracer_pdf_width_phase_velocity_correlation_coefficients.csv"
    )
    assert len(phase_table) == 216
    np.testing.assert_allclose(phase_table["spearman_rho"], 1.0)
