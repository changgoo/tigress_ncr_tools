import matplotlib
import numpy as np

matplotlib.use("Agg")

import tigress_ncr_tools.surface_density_stats as stats
from tigress_ncr_tools.surface_density_stats import (
    angle_averaged_power,
    centered_subregion,
    pdfs_fluctuations,
    pdfs_log10_sigma,
    plot_pdf_summary,
    plot_power_summary,
    read_shear_parameters,
    residual_shear,
    shear_remap_periodic,
    window_1d,
)


def test_read_shear_parameters_from_problem_section(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "athinput.test").write_text(
        "<job>\nOmega = 999\n"
        "<problem>\nqshear = 1.0 # comment\nOmega = 2.8d-2\n"
    )
    path, qshear, omega = read_shear_parameters(model)
    assert path.name == "athinput.test"
    assert qshear == 1.0
    assert omega == 0.028


def test_fourier_remap_recovers_periodic_shearing_wave():
    size = 64
    length = 1.0
    spacing = length / size
    x = (np.arange(size) + 0.5) * spacing - length / 2
    y = (np.arange(size) + 0.5) * spacing - length / 2
    xx, yy = np.meshgrid(x, y)
    mx, my = 2, 4
    remap_time, shear = residual_shear(
        time=12.5, qshear=1.0, omega=0.1, lx=length, ly=length
    )
    expected = np.cos(2 * np.pi * (mx * xx + my * yy))
    sheared = np.cos(
        2 * np.pi * ((mx + shear * my) * xx + my * yy)
    )
    recovered = shear_remap_periodic(sheared, x, spacing, shear)
    assert remap_time == 2.5
    assert shear == 0.25
    np.testing.assert_allclose(recovered, expected, atol=2.0e-14)


def test_power_uses_physical_shearing_wavevector():
    size = 64
    spacing = 1.0
    x = (np.arange(size) + 0.5) * spacing
    y = (np.arange(size) + 0.5) * spacing
    xx, yy = np.meshgrid(x, y)
    mx, my = 2, 4
    shear = 0.25
    fluctuation = np.cos(
        2 * np.pi * (mx * xx / size + my * yy / size)
    )
    edges = np.linspace(0.0, np.pi, 101)
    power, count = angle_averaged_power(
        fluctuation, spacing, spacing, shear, edges
    )
    expected_k = (
        2 * np.pi / size
        * np.sqrt((mx + shear * my) ** 2 + my**2)
    )
    peak = int(np.nanargmax(power))
    assert edges[peak] <= expected_k < edges[peak + 1]
    assert count[peak] > 0


def test_area_and_mass_weighted_sigma_delta_and_s_pdfs():
    sigma = np.asarray([[1.0, 1.0], [3.0, 3.0]])
    sigma_edges = np.linspace(-0.1, 0.6, 8)
    delta_edges = np.linspace(-0.6, 0.6, 7)
    s_edges = np.linspace(-0.8, 0.5, 14)
    area, mass = pdfs_log10_sigma(sigma, sigma_edges)
    result = pdfs_fluctuations(sigma, delta_edges, s_edges)
    np.testing.assert_allclose(np.sum(area * np.diff(sigma_edges)), 1.0)
    np.testing.assert_allclose(np.sum(mass * np.diff(sigma_edges)), 1.0)
    np.testing.assert_allclose(
        np.sum(result["pdf_delta_area"] * np.diff(delta_edges)), 1.0
    )
    np.testing.assert_allclose(
        np.sum(result["pdf_delta_mass"] * np.diff(delta_edges)), 1.0
    )
    np.testing.assert_allclose(
        np.sum(result["pdf_s_area"] * np.diff(s_edges)), 1.0
    )
    np.testing.assert_allclose(
        np.sum(result["pdf_s_mass"] * np.diff(s_edges)), 1.0
    )
    assert np.argmax(mass) == np.argmax(area) + 4


def test_centered_subregion_windows_and_padding():
    values = np.arange(64).reshape(8, 8)
    centers = np.arange(8) + 0.5
    local, local_x, local_y = centered_subregion(
        values, centers, centers, size=4.0
    )
    assert local.shape == (4, 4)
    np.testing.assert_array_equal(local_x, [2.5, 3.5, 4.5, 5.5])
    np.testing.assert_array_equal(local_y, local_x)
    assert np.all(window_1d(8, "none") == 1.0)
    assert window_1d(8, "hann")[0] == 0.0
    assert window_1d(8, "tukey", 0.5)[0] == 0.0
    power, count = angle_averaged_power(
        local, 1.0, 1.0, 0.1, np.geomspace(0.1, np.pi, 9),
        window="tukey", tukey_alpha=0.5, pad_factor=2.0,
    )
    assert power.shape == count.shape == (8,)


def test_summary_figures_include_all_pdf_and_power_panels(tmp_path):
    result = {
        "model": np.asarray("R8_8pc_NCR_Lxy1024_late"),
        "pdf_log10_sigma_centers": np.linspace(-2, 3, 4),
        "pdf_delta_centers": np.linspace(-1, 3, 4),
        "pdf_s_centers": np.linspace(-3, 3, 4),
        "k_centers": np.geomspace(0.01, 1, 4),
    }
    for key in (
        "pdf_log10_sigma_area", "pdf_log10_sigma_mass", "pdf_delta_area", "pdf_delta_mass",
        "pdf_s_area", "pdf_s_mass", "power_delta", "power_s",
    ):
        result[key] = np.ones((3, 4))
    pdf_path = tmp_path / "pdf.png"
    power_path = tmp_path / "power.png"
    plot_pdf_summary([result], pdf_path)
    plot_power_summary([result], power_path)
    assert pdf_path.stat().st_size > 0
    assert power_path.stat().st_size > 0


def test_analyze_model_saves_time_ordered_complete_archive(tmp_path, monkeypatch):
    model = tmp_path / "late_model"
    output_dir = model / "proj2d" / "theta0"
    output_dir.mkdir(parents=True)
    (model / "athinput.test").write_text(
        "<problem>\nqshear = 1.0\nOmega = 0.02\n"
    )
    paths = [tmp_path / "restart.9999.proj2d", tmp_path / "restart.0001.proj2d"]
    monkeypatch.setattr(
        stats, "projection_series",
        lambda *args, **kwargs: [(5.0, paths[0]), (6.0, paths[1])],
    )

    def fake_frame(path, fields):
        time = 5.0 if path == paths[0] else 6.0
        num = "9999" if path == paths[0] else "0001"
        size = 16
        edges = np.arange(size + 1, dtype=float)
        centers = edges[:-1] + 0.5
        xx, yy = np.meshgrid(centers, centers)
        nH = 10.0 + np.cos(2 * np.pi * (xx + yy) / size)
        return {
            "path": str(path),
            "time": time,
            "num": num,
            "theta": 0.0,
            "x_edges": edges,
            "y_edges": edges,
            "x_centers": centers,
            "y_centers": centers,
            "x_spacing": 1.0,
            "y_spacing": 1.0,
            "fields": {"nH": nH},
        }

    monkeypatch.setattr(stats, "read_proj2d", fake_frame)
    result = stats.analyze_model(
        model,
        np.linspace(-2, 3, 11),
        np.linspace(-1, 30, 11),
        np.linspace(-6, 4, 11),
        k_bins=4,
    )
    archive = output_dir / stats.STATISTICS_NAME
    assert archive.is_file()
    assert not (output_dir / f".{stats.STATISTICS_NAME}.tmp").exists()
    with np.load(archive) as saved:
        np.testing.assert_array_equal(saved["time"], [5.0, 6.0])
        np.testing.assert_array_equal(saved["num"], ["9999", "0001"])
        assert saved["pdf_delta_mass"].shape == (2, 10)
        assert saved["pdf_s_area"].shape == (2, 10)
        assert saved["power_delta"].shape == (2, 4)
        assert saved["power_s"].shape == (2, 4)
        assert saved["qshear"] == 1.0
        assert saved["omega_kms_per_pc"] == 0.02
    np.testing.assert_array_equal(result["time"], [5.0, 6.0])
