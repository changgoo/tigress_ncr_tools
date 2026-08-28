from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tigress_ncr_tools.correlation_parameters import parameter_axis_limits
from tigress_ncr_tools.plot_suite_density_spectrum import (
    _axial_angle_statistics,
    create_spectrum_figure,
    exclude_corrupted_archive_models,
    integral_scale,
    model_power2d_archive,
    overdensity_power,
    physical_time_mean_power_2d,
    plot_suite_time_mean_power_2d,
    plot_spectrum_diagnostic_relations,
    plot_spectrum_correlations,
    plot_time_mean_spectrum,
    spectrum_time_diagnostics,
    spectral_slope_alpha,
    time_mean_power,
)


def test_overdensity_power_remaps_then_uses_physical_shearing_wavevector():
    size = 64
    spacing = 1.0
    edges = np.arange(size + 1, dtype=float)
    centers = edges[:-1] + 0.5
    xx, yy = np.meshgrid(centers, centers)
    mx, my = 2, 4
    qshear, omega, time = 1.0, 0.01, 25.0
    shear = qshear * omega * time
    sigma = 2.0 + 0.5 * np.cos(2.0 * np.pi * ((mx + shear * my) * xx + my * yy) / size)
    frame = {
        "time": time,
        "theta": 0.0,
        "x_edges": edges,
        "y_edges": edges,
        "x_centers": centers,
        "y_centers": centers,
        "x_spacing": spacing,
        "y_spacing": spacing,
        "fields": {"nH": sigma},
    }
    k_edges = np.linspace(0.0, np.pi, 101)
    result = overdensity_power(frame, qshear, omega, k_edges=k_edges)
    expected_k = 2.0 * np.pi / size * np.sqrt((mx + shear * my) ** 2 + my**2)
    peak = int(np.nanargmax(result["power"]))
    assert k_edges[peak] <= expected_k < k_edges[peak + 1]
    assert result["shear"] == shear
    np.testing.assert_allclose(result["mean_sigma"], 2.0, rtol=0.0, atol=1.0e-14)
    assert not result["has_negative_sigma"]


def test_overdensity_power_uses_requested_proj2d_field():
    size = 16
    edges = np.arange(size + 1, dtype=float)
    centers = edges[:-1] + 0.5
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    frame = {
        "time": 0.0,
        "theta": 0.0,
        "x_edges": edges,
        "y_edges": edges,
        "x_centers": centers,
        "y_centers": centers,
        "x_spacing": 1.0,
        "y_spacing": 1.0,
        "fields": {
            "nH": np.ones((size, size)),
            "nHI": 2.0 + 0.5 * np.cos(2.0 * np.pi * xx / size),
        },
    }
    result = overdensity_power(
        frame, 1.0, 0.01, field="nHI", k_edges=np.linspace(0.0, np.pi, 33)
    )
    assert result["mean_sigma"] == pytest.approx(2.0)
    assert np.nanmax(result["power"]) > 0.0


def test_per_model_2d_archives_are_separate_by_quantity(tmp_path):
    gas = model_power2d_archive(tmp_path, quantity="gas")
    hi = model_power2d_archive(tmp_path, quantity="hi")
    em = model_power2d_archive(tmp_path, quantity="em")
    assert gas.parent.name == "density_power_2d"
    assert hi.parent.name == "hi_power_2d"
    assert em.parent.name == "em_power_2d"
    assert len({gas, hi, em}) == 3


def test_time_mean_power_selects_requested_interval():
    data = {
        "model": np.array(["a", "b"]),
        "time": np.array([[0.0, 200.0, 600.0], [0.0, 200.0, 600.0]]),
        "power_delta": np.array(
            [
                [[100.0, 100.0], [2.0, 4.0], [4.0, 8.0]],
                [[100.0, 100.0], [6.0, 12.0], [10.0, 20.0]],
            ]
        ),
    }
    result = time_mean_power(data, (200.0, 600.0))
    np.testing.assert_allclose(result, [[3.0, 6.0], [8.0, 16.0]])


def test_time_mean_2d_power_is_deposited_at_physical_sheared_kx():
    size = 8
    length = float(size)
    kx0 = 2.0 * np.pi * np.fft.fftfreq(size)
    ky = 2.0 * np.pi * np.fft.fftfreq(size)
    x_mode = kx0 * length / (2.0 * np.pi)
    y_mode = ky * length / (2.0 * np.pi)
    ix = int(np.flatnonzero(np.isclose(x_mode, 1.0))[0])
    iy = int(np.flatnonzero(np.isclose(y_mode, 1.0))[0])
    power = np.zeros((2, size, size))
    power[:, iy, ix] = 10.0
    mean, count, sorted_x, sorted_y = physical_time_mean_power_2d(
        power,
        kx0,
        ky,
        np.asarray([0.0, 1.0]),
        np.asarray([length, length]),
        np.asarray([True, True]),
        chunk_size=1,
    )
    output_y = int(np.flatnonzero(np.isclose(sorted_y, 1.0))[0])
    output_x1 = int(np.flatnonzero(np.isclose(sorted_x, 1.0))[0])
    output_x2 = int(np.flatnonzero(np.isclose(sorted_x, 2.0))[0])
    assert mean[output_y, output_x1] == pytest.approx(5.0)
    assert mean[output_y, output_x2] == pytest.approx(5.0)
    assert count[output_y, output_x1] == 2
    assert count[output_y, output_x2] == 2


def test_integral_scale_uses_shell_energy_and_bin_widths():
    edges = np.array([1.0, 2.0, 4.0])
    centers = np.sqrt(edges[:-1] * edges[1:])
    power = np.array([3.0, 1.0])
    energy = centers * power / (2.0 * np.pi)
    widths = np.diff(edges)
    expected = np.sum(energy * (2.0 * np.pi / centers) * widths) / np.sum(
        energy * widths
    )
    assert integral_scale(edges, power) == expected


def test_spectral_slope_uses_requested_wavelength_interval():
    k = np.geomspace(2.0 * np.pi / 400.0, 2.0 * np.pi / 32.0, 40)
    power = 7.0 * k**-2.5
    wavelength = 2.0 * np.pi / k
    outside = (wavelength <= 64.0) | (wavelength >= 256.0)
    power[outside] = 1.0e6 * k[outside] ** 4.0
    alpha, count = spectral_slope_alpha(k, power)
    assert count > 10
    assert alpha == pytest.approx(2.5)


def test_axial_angle_summary_respects_180_degree_wrap():
    angles = np.deg2rad([88.0, 89.0, -89.0, -88.0])
    center, coherence, _, low, high, count = _axial_angle_statistics(angles)
    assert abs(abs(np.rad2deg(center)) - 90.0) < 1.0e-10
    assert coherence > 0.99
    assert low <= center <= high
    assert count == 4


def test_diagnostics_are_measured_for_every_instantaneous_spectrum():
    edges = np.geomspace(2.0 * np.pi / 800.0, 2.0 * np.pi / 40.0, 41)
    k = np.sqrt(edges[:-1] * edges[1:])
    data = {
        "k_edges": edges,
        "k_centers": k,
        "pixel_size_pc": np.array([4.0]),
        "power_delta": np.array([[3.0 * k**-2.0, 5.0 * k**-3.0]]),
    }
    diagnostics = spectrum_time_diagnostics(data)
    assert diagnostics["integral_scale_time_pc"].shape == (1, 2)
    np.testing.assert_allclose(diagnostics["spectral_slope_alpha_time"], [[2.0, 3.0]])
    assert np.all(diagnostics["slope_fit_bin_count_time"] >= 3)


def test_repaired_row0000_is_preserved_by_default():
    data = {
        "model": np.array(["row0001", "R8_8pc_NCR_row0000", "row0002"]),
        "power_delta": np.arange(12).reshape(3, 2, 2),
        "qshear": np.array([0.5, 1.0, 1.5]),
        "k_centers": np.array([0.1, 0.2]),
    }
    filtered, removed = exclude_corrupted_archive_models(data)
    assert removed == ()
    assert filtered["model"].tolist() == [
        "row0001", "R8_8pc_NCR_row0000", "row0002",
    ]
    assert filtered["power_delta"].shape == (3, 2, 2)
    np.testing.assert_array_equal(filtered["k_centers"], data["k_centers"])


def test_explicit_archive_exclusion_filters_every_model_axis():
    data = {
        "model": np.array(["row0001", "bad-model", "row0002"]),
        "power_delta": np.arange(12).reshape(3, 2, 2),
        "qshear": np.array([0.5, 1.0, 1.5]),
        "k_centers": np.array([0.1, 0.2]),
    }
    filtered, removed = exclude_corrupted_archive_models(
        data, excluded={"bad-model"}
    )
    assert removed == ("bad-model",)
    assert filtered["model"].tolist() == ["row0001", "row0002"]
    assert filtered["power_delta"].shape == (2, 2, 2)
    np.testing.assert_array_equal(filtered["k_centers"], data["k_centers"])


def test_spectrum_diagnostic_relation_figure(tmp_path):
    summary = pd.DataFrame(
        {
            "mean_sfr10": [1.0e-3, 1.0e-2],
            "omega": [0.02, 0.04],
            "integral_scale_pc": [280.0, 390.0],
            "spectral_slope_alpha": [2.2, 2.4],
            "integral_scale_time_mean_pc": [275.0, 385.0],
            "integral_scale_time_std_pc": [25.0, 35.0],
            "spectral_slope_alpha_time_mean": [2.15, 2.35],
            "spectral_slope_alpha_time_std": [0.12, 0.18],
            "integral_scale_time_median_pc": [272.0, 382.0],
            "integral_scale_time_percentile16_pc": [250.0, 350.0],
            "integral_scale_time_percentile84_pc": [300.0, 420.0],
            "spectral_slope_alpha_time_median": [2.14, 2.34],
            "spectral_slope_alpha_time_percentile16": [2.02, 2.17],
            "spectral_slope_alpha_time_percentile84": [2.27, 2.51],
        }
    )
    output = tmp_path / "diagnostics.png"
    plot_spectrum_diagnostic_relations(
        summary,
        output,
        color_field="omega",
        cmap_name="viridis",
        color_scale="log",
        colorbar_label=r"$\Omega$",
        dpi=50,
    )
    assert output.stat().st_size > 0


def test_spectrum_correlation_figure(monkeypatch, tmp_path):
    summary = pd.DataFrame(
        {
            "mean_sfr10": [1.0e-3, 1.0e-2],
            "stellar_surface_density": [20.0, 80.0],
            "stellar_scale_height": [100.0, 400.0],
            "omega": [0.02, 0.04],
            "qshear": [0.7, 1.2],
            "kappa": [0.03, 0.06],
            "stellar_midplane_density": [0.02, 0.08],
            "integral_scale_time_median_pc": [270.0, 380.0],
            "integral_scale_time_percentile16_pc": [245.0, 345.0],
            "integral_scale_time_percentile84_pc": [300.0, 420.0],
            "spectral_slope_alpha_time_median": [2.1, 2.4],
            "spectral_slope_alpha_time_percentile16": [1.9, 2.2],
            "spectral_slope_alpha_time_percentile84": [2.3, 2.6],
            "anisotropy_band_amplitude_time_median": [0.2, 0.4],
            "anisotropy_band_amplitude_time_percentile16": [0.1, 0.3],
            "anisotropy_band_amplitude_time_percentile84": [0.3, 0.5],
            "anisotropy_band_angle_circular_mean_deg": [-20.0, 30.0],
            "anisotropy_band_angle_time_percentile16_deg": [-35.0, 15.0],
            "anisotropy_band_angle_time_percentile84_deg": [-5.0, 45.0],
        }
    )
    output = tmp_path / "correlations.png"
    figures = []
    with monkeypatch.context() as patch:
        patch.setattr(
            "tigress_ncr_tools.plot_suite_density_spectrum.plt.close", figures.append
        )
        plot_spectrum_correlations(summary, output, dpi=50)
    assert output.stat().st_size > 0
    np.testing.assert_allclose(
        figures[0].axes[0].get_xlim(),
        parameter_axis_limits(summary["stellar_surface_density"], "log"),
    )
    np.testing.assert_allclose(
        figures[0].axes[1].get_xlim(),
        parameter_axis_limits(summary["stellar_scale_height"], "log"),
    )
    plt.close(figures[0])


def test_time_mean_spectrum_figure(tmp_path):
    ranked = [(Path("high"), 1.0e-2), (Path("low"), 1.0e-3)]
    data = {
        "model": np.array(["high", "low"]),
        "mean_sfr10": np.array([1.0e-2, 1.0e-3]),
        "time": np.array([[200.0, 600.0], [200.0, 600.0]]),
        "k_centers": np.array([0.01, 0.02, 0.04]),
        "box_size_pc": np.asarray(1024.0),
        "power_delta": np.array(
            [
                [[10.0, 3.0, 1.0], [12.0, 4.0, 1.2]],
                [[5.0, 2.0, 0.8], [6.0, 2.5, 0.9]],
            ]
        ),
        "anisotropy_amplitude": np.array(
            [
                [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
            ]
        ),
    }
    output = tmp_path / "spectrum.png"
    plot_time_mean_spectrum(data, ranked, output, dpi=60)
    assert output.stat().st_size > 0


def test_time_mean_2d_spectrum_suite_figure(tmp_path):
    ranked = [(Path("high"), 1.0e-2), (Path("low"), 1.0e-3)]
    modes = np.arange(-4.0, 4.0)
    yy, xx = np.meshgrid(modes, modes, indexing="ij")
    base = 1.0 / (1.0 + xx**2 + 2.0 * yy**2)
    data = {
        "model": np.asarray(["high", "low"]),
        "time_bounds": np.asarray([200.0, 600.0]),
        "kx_mode": modes,
        "ky_mode": modes,
        "mean_power_2d": np.asarray([base, 0.5 * base]),
    }
    output = tmp_path / "power2d.png"
    plot_suite_time_mean_power_2d(data, ranked, output, mode_limit=3.0, dpi=40)
    assert output.stat().st_size > 0


def test_uniform_first_spectrum_can_initialize_log_figure(tmp_path):
    ranked = [(Path("model"), 1.0e-3)]
    from tigress_ncr_tools.plot_suite_hst_evolution import sfr_colormap

    cmap, norm = sfr_colormap([1.0e-3], "plasma", "log")
    fig, axes, _, _ = create_spectrum_figure(
        np.array([0.01, 0.02, 0.04]),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        ranked,
        title="uniform",
        cmap=cmap,
        norm=norm,
        power_limits=(1.0e-2, 1.0e2),
        anisotropy_limits=(0.0, 1.0),
        box_size=1024.0,
    )
    for axis in axes:
        np.testing.assert_allclose(
            axis.get_xlim(), np.array([0.01, 0.04]) * 1024.0 / (2.0 * np.pi)
        )
        assert axis.get_xlabel() == r"dimensionless wavenumber $kL/(2\pi)$"
        assert len(axis.child_axes) == 1
        assert "wavelength" in axis.child_axes[0].get_xlabel()
        np.testing.assert_allclose(axis.child_axes[0].get_xticks(), [256.0, 512.0])
    output = tmp_path / "uniform.png"
    fig.savefig(output, dpi=40)
    matplotlib.pyplot.close(fig)
    assert output.stat().st_size > 0
