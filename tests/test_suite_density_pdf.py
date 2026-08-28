from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_density_pdf import (
    attach_velocity_summary,
    frame_density_pdf,
    gaussian_fit_from_pdf,
    pdf_display_limits,
    plot_median_pdfs,
    plot_pdf_width_derived_velocity_correlations,
    plot_pdf_width_correlations,
    plot_pdf_width_velocity_correlations,
    plot_s_pdf_fit_grid,
)
from tigress_ncr_tools.plot_suite_hst_evolution import (
    SPEED_QUANTITIES,
    derived_velocity_quantities,
)


def test_frame_density_pdf_normalizes_and_measures_pixels_directly():
    sigma = np.asarray([[1.0, 1.0], [3.0, 3.0]])
    frame = {"theta": 0.0, "fields": {"nH": sigma}}
    delta_edges = np.linspace(-1.0, 1.0, 17)
    s_edges = np.linspace(-1.0, 1.0, 21)
    result = frame_density_pdf(frame, delta_edges, s_edges)
    ratio = sigma.ravel() / sigma.mean()
    np.testing.assert_allclose(
        np.sum(result["pdf_delta_area"] * np.diff(delta_edges)), 1.0
    )
    np.testing.assert_allclose(np.sum(result["pdf_s_area"] * np.diff(s_edges)), 1.0)
    assert result["std_delta"] == np.std(ratio - 1.0)
    assert result["std_s"] == np.std(np.log(ratio))


def test_frame_density_pdf_uses_requested_proj2d_field():
    frame = {
        "theta": 0.0,
        "fields": {
            "nH": np.ones((2, 2)),
            "nHI": np.asarray([[1.0, 1.0], [3.0, 3.0]]),
        },
    }
    result = frame_density_pdf(
        frame,
        np.linspace(-1.0, 1.0, 17),
        np.linspace(-1.0, 1.0, 21),
        field="nHI",
    )
    assert result["mean_sigma_code"] == 2.0
    assert result["std_delta"] == 0.5


def test_gaussian_fit_matches_binned_pdf_moments():
    edges = np.linspace(-4.0, 4.0, 161)
    centers = 0.5 * (edges[:-1] + edges[1:])
    expected_mean = -0.3
    expected_sigma = 0.7
    density = np.exp(-0.5 * ((centers - expected_mean) / expected_sigma) ** 2)
    density /= np.sqrt(2.0 * np.pi) * expected_sigma
    mean, sigma, fit = gaussian_fit_from_pdf(centers, edges, density)
    np.testing.assert_allclose(mean, expected_mean, atol=1.0e-5)
    np.testing.assert_allclose(sigma, expected_sigma, atol=1.0e-4)
    np.testing.assert_allclose(np.sum(fit * np.diff(edges)), 1.0, atol=1.0e-6)


def test_pdf_display_limits_exclude_empty_histogram_tails():
    centers = np.arange(-2.5, 3.0)
    density = np.asarray([[0.0, 2.0e-4, 1.0, 3.0e-4, 0.0, 0.0]])
    x_limits, y_limits = pdf_display_limits(centers, density)
    assert x_limits == pytest.approx((-2.0, 1.0))
    assert y_limits == pytest.approx((5.0e-5, 1.5))


def test_derived_velocity_quantities_are_formed_instantaneously():
    speeds = {
        "sigma_x1": np.asarray([3.0, 0.0]),
        "sigma_x2": np.asarray([4.0, 0.0]),
        "sigma_x3": np.asarray([12.0, 2.0]),
        "thermal": np.asarray([5.0, 1.0]),
        "alfven_x1": np.asarray([1.0, 0.0]),
        "alfven_x2": np.asarray([2.0, 0.0]),
        "alfven_x3": np.asarray([2.0, 0.0]),
    }
    derived = derived_velocity_quantities(speeds)
    np.testing.assert_allclose(derived["sigma_3d"], [13.0, 2.0])
    np.testing.assert_allclose(derived["alfven_3d"], [3.0, 0.0])
    np.testing.assert_allclose(derived["plasma_beta"][0], 50.0 / 9.0)
    np.testing.assert_allclose(derived["mach_3d"], [2.6, 2.0])
    assert np.isinf(derived["plasma_beta"][1])
    assert derived["mach_mhd"][1] == pytest.approx(2.0)
    assert derived["mach_mhd"][0] == pytest.approx(2.6 / np.sqrt(1.18))


def _plot_data():
    s_edges = np.linspace(-3.0, 2.0, 31)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    delta_edges = np.linspace(-1.0, 5.0, 31)
    delta_centers = 0.5 * (delta_edges[:-1] + delta_edges[1:])
    s_pdf = np.asarray(
        [
            np.exp(-0.5 * ((s_centers + 0.2) / 0.6) ** 2),
            np.exp(-0.5 * ((s_centers + 0.4) / 0.8) ** 2),
        ]
    )
    delta_pdf = np.asarray([np.exp(-delta_centers), np.exp(-0.7 * delta_centers)])
    return {
        "mean_sfr10": np.asarray([1.0e-2, 1.0e-3]),
        "s_centers": s_centers,
        "delta_centers": delta_centers,
        "pdf_s_time_median": s_pdf,
        "pdf_s_time_percentile16": 0.8 * s_pdf,
        "pdf_s_time_percentile84": 1.2 * s_pdf,
        "pdf_s_gaussian_fit": s_pdf,
        "pdf_delta_time_median": delta_pdf,
    }


def test_pdf_summary_and_gaussian_grid_figures(tmp_path):
    data = _plot_data()
    ranked = [(Path("high"), 1.0e-2), (Path("low"), 1.0e-3)]
    summary_output = tmp_path / "median.png"
    grid_output = tmp_path / "grid.png"
    plot_median_pdfs(data, ranked, summary_output, dpi=40)
    plot_s_pdf_fit_grid(data, ranked, grid_output, dpi=40)
    assert summary_output.stat().st_size > 0
    assert grid_output.stat().st_size > 0


def test_pdf_width_correlation_figure(tmp_path):
    summary = pd.DataFrame(
        {
            "mean_sfr10": [1.0e-3, 1.0e-2],
            "stellar_surface_density": [20.0, 80.0],
            "stellar_scale_height": [100.0, 400.0],
            "omega": [0.02, 0.04],
            "qshear": [0.7, 1.2],
            "kappa": [0.03, 0.06],
            "stellar_midplane_density": [0.02, 0.08],
            "std_delta_time_median": [1.2, 1.8],
            "std_delta_time_percentile16": [1.0, 1.5],
            "std_delta_time_percentile84": [1.5, 2.2],
            "std_s_time_median": [0.7, 1.0],
            "std_s_time_percentile16": [0.6, 0.8],
            "std_s_time_percentile84": [0.9, 1.3],
        }
    )
    output = tmp_path / "correlations.png"
    plot_pdf_width_correlations(summary, output, dpi=40)
    assert output.stat().st_size > 0


def test_velocity_summary_and_correlation_figure(monkeypatch, tmp_path):
    summary = pd.DataFrame(
        {
            "model": ["high", "low"],
            "mean_sfr10": [1.0e-2, 1.0e-3],
            "std_delta_time_median": [1.2, 1.8],
            "std_delta_time_percentile16": [1.0, 1.5],
            "std_delta_time_percentile84": [1.5, 2.2],
            "std_s_time_median": [0.7, 1.0],
            "std_s_time_percentile16": [0.6, 0.8],
            "std_s_time_percentile84": [0.9, 1.3],
        }
    )
    history = {
        "time": np.asarray([100.0, 200.0, 400.0, 600.0, 700.0]),
        "mass": np.ones(5),
    }
    for index, (_, _, field, _) in enumerate(SPEED_QUANTITIES, start=1):
        history[field] = np.full(5, float(index))
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_density_pdf.whole_history_file",
        lambda model: model,
    )
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_density_pdf.read_hst",
        lambda path, max_rows=None: history,
    )
    ranked = [(Path("high"), 1.0e-2), (Path("low"), 1.0e-3)]
    augmented = attach_velocity_summary(summary, ranked, bounds=(200.0, 600.0))
    assert augmented["sigma_x1_time_median"].iloc[0] == pytest.approx(np.sqrt(2.0))
    assert augmented["thermal_time_count"].iloc[0] == 3
    assert augmented["sigma_3d_time_median"].iloc[0] == pytest.approx(np.sqrt(12.0))
    assert augmented["alfven_3d_time_median"].iloc[0] == pytest.approx(6.0)
    assert augmented["plasma_beta_time_median"].iloc[0] == pytest.approx(2.0 / 9.0)
    assert augmented["mach_3d_time_median"].iloc[0] == pytest.approx(np.sqrt(3.0))
    assert augmented["mach_mhd_time_median"].iloc[0] == pytest.approx(
        np.sqrt(6.0 / 11.0)
    )
    output = tmp_path / "velocity_correlations.png"
    plot_pdf_width_velocity_correlations(augmented, output, dpi=40)
    assert output.stat().st_size > 0
    derived_output = tmp_path / "derived_velocity_correlations.png"
    plot_pdf_width_derived_velocity_correlations(augmented, derived_output, dpi=40)
    assert derived_output.stat().st_size > 0
