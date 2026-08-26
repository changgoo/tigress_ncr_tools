from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_tracer_correlations import (
    PDF_METRICS,
    SPECTRUM_METRICS,
    TRACERS,
    load_tracer_summaries,
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


def test_render_suite_tracer_correlations(tmp_path):
    _write_tracer_summaries(tmp_path)
    output = tmp_path / "comparison"
    render_suite_tracer_correlations(tmp_path, output_dir=output, dpi=40)
    assert (output / "tracer_pdf_width_correlations.png").is_file()
    assert (output / "tracer_power_spectrum_correlations.png").is_file()
    table = pd.read_csv(output / "tracer_correlation_coefficients.csv")
    assert len(table) == 15
    np.testing.assert_allclose(table["spearman_rho"], 1.0)
