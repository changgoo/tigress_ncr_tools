from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_density_spectrum import (
    create_spectrum_figure,
    exclude_corrupted_archive_models,
    integral_scale,
    overdensity_power,
    plot_spectrum_diagnostic_relations,
    plot_time_mean_spectrum,
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
    k = np.geomspace(2.0 * np.pi / 600.0, 2.0 * np.pi / 40.0, 40)
    power = 7.0 * k**-2.5
    alpha, count = spectral_slope_alpha(k, power, pixel_size=8.0, scale=400.0)
    assert count > 10
    assert alpha == pytest.approx(2.5)


def test_corrupted_row0000_is_removed_from_every_model_axis():
    data = {
        "model": np.array(["row0001", "R8_8pc_NCR_row0000", "row0002"]),
        "power_delta": np.arange(12).reshape(3, 2, 2),
        "qshear": np.array([0.5, 1.0, 1.5]),
        "k_centers": np.array([0.1, 0.2]),
    }
    filtered, removed = exclude_corrupted_archive_models(data)
    assert removed == ("R8_8pc_NCR_row0000",)
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
    }
    output = tmp_path / "spectrum.png"
    plot_time_mean_spectrum(data, ranked, output, dpi=60)
    assert output.stat().st_size > 0


def test_uniform_first_spectrum_can_initialize_log_figure(tmp_path):
    ranked = [(Path("model"), 1.0e-3)]
    from tigress_ncr_tools.plot_suite_hst_evolution import sfr_colormap

    cmap, norm = sfr_colormap([1.0e-3], "plasma", "log")
    fig, axes, _, _ = create_spectrum_figure(
        np.array([0.01, 0.02, 0.04]),
        np.zeros((1, 3)),
        ranked,
        title="uniform",
        cmap=cmap,
        norm=norm,
        power_limits=(1.0e-2, 1.0e2),
        dimensionless_limits=(1.0e-5, 1.0),
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
