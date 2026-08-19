from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_density_spectrum import (
    create_spectrum_figure,
    overdensity_power,
    plot_time_mean_spectrum,
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
    sigma = 2.0 + 0.5 * np.cos(
        2.0 * np.pi * ((mx + shear * my) * xx + my * yy) / size
    )
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
    result = overdensity_power(
        frame, qshear, omega, k_edges=k_edges
    )
    expected_k = 2.0 * np.pi / size * np.sqrt(
        (mx + shear * my) ** 2 + my**2
    )
    peak = int(np.nanargmax(result["power"]))
    assert k_edges[peak] <= expected_k < k_edges[peak + 1]
    assert result["shear"] == shear
    np.testing.assert_allclose(
        result["mean_sigma"], 2.0, rtol=0.0, atol=1.0e-14
    )
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


def test_time_mean_spectrum_figure(tmp_path):
    ranked = [(Path("high"), 1.0e-2), (Path("low"), 1.0e-3)]
    data = {
        "model": np.array(["high", "low"]),
        "mean_sfr10": np.array([1.0e-2, 1.0e-3]),
        "time": np.array([[200.0, 600.0], [200.0, 600.0]]),
        "k_centers": np.array([0.01, 0.02, 0.04]),
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
    )
    for axis in axes:
        np.testing.assert_allclose(axis.get_xlim(), (0.01, 0.04))
    output = tmp_path / "uniform.png"
    fig.savefig(output, dpi=40)
    matplotlib.pyplot.close(fig)
    assert output.stat().st_size > 0
