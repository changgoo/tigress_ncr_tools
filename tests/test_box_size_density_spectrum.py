import matplotlib
import numpy as np

matplotlib.use("Agg")

from tigress_ncr_tools.plot_box_size_density_spectrum import (
    SegmentedBoxModel,
    box_size_diagnostic_summary,
    common_k_edges_from_archives,
    discover_box_size_models,
    plot_box_size_diagnostics,
    plot_box_size_time_mean,
    plot_box_size_time_mean_power2d,
    segmented_projection_index,
)


def _make_segment(suite, name):
    segment = suite / name
    (segment / "hst").mkdir(parents=True)
    (segment / "hst" / "model.hst").touch()
    (segment / "proj2d" / "theta0").mkdir(parents=True)
    return segment


def test_discovers_and_orders_complete_segmented_box_models(tmp_path):
    for size in (4096, 1024, 2048):
        _make_segment(tmp_path, f"R8_8pc_NCR_Lxy{size}_early")
        _make_segment(tmp_path, f"R8_8pc_NCR_Lxy{size}_late")

    models = discover_box_size_models(tmp_path)

    assert [model.name for model in models] == [
        "R8_8pc_NCR_Lxy1024",
        "R8_8pc_NCR_Lxy2048",
        "R8_8pc_NCR_Lxy4096",
    ]
    assert [model.box_size_pc for model in models] == [1024.0, 2048.0, 4096.0]
    assert all(len(model.segments) == 2 for model in models)


def test_segmented_projection_index_joins_and_prefers_late(tmp_path):
    early = _make_segment(tmp_path, "R8_8pc_NCR_Lxy1024_early")
    late = _make_segment(tmp_path, "R8_8pc_NCR_Lxy1024_late")
    for segment, numbers in ((early, (0, 1)), (late, (1, 2))):
        for number in numbers:
            path = segment / "proj2d" / "theta0" / f"model.{number:04d}.theta0.proj2d"
            path.touch()
    model = SegmentedBoxModel("R8_8pc_NCR_Lxy1024", (early, late), 1024.0)

    index = segmented_projection_index(model)

    assert sorted(index) == [0, 1, 2]
    assert index[1].is_relative_to(late)


def test_common_k_bins_keep_largest_box_fundamental_and_common_nyquist():
    archives = [
        {
            "box_size_xy_pc": np.array([1024.0, 1024.0]),
            "pixel_size_xy_pc": np.array([8.0, 8.0]),
        },
        {
            "box_size_xy_pc": np.array([4096.0, 4096.0]),
            "pixel_size_xy_pc": np.array([16.0, 16.0]),
        },
    ]

    edges = common_k_edges_from_archives(archives, bins=12)

    assert edges.size == 13
    np.testing.assert_allclose(edges[0], 2.0 * np.pi / 4096.0)
    np.testing.assert_allclose(edges[-1], np.pi / 16.0)


def _synthetic_archive():
    edges = np.geomspace(2.0 * np.pi / 4096.0, np.pi / 8.0, 41)
    k = np.sqrt(edges[:-1] * edges[1:])
    time = np.array([[200.0, 600.0]] * 3)
    power = np.asarray(
        [
            [2.0 * k**-2.0, 2.5 * k**-2.0],
            [3.0 * k**-2.2, 3.5 * k**-2.2],
            [4.0 * k**-2.4, 4.5 * k**-2.4],
        ]
    )
    amplitude = np.broadcast_to(np.linspace(0.1, 0.4, k.size), power.shape).copy()
    band_amplitude = np.array([[0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
    band_angle = np.deg2rad(np.array([[-20.0, -10.0], [0.0, 10.0], [20.0, 30.0]]))
    return {
        "model": np.array(["L1024", "L2048", "L4096"]),
        "box_size_pc": np.array([1024.0, 2048.0, 4096.0]),
        "pixel_size_pc": np.array([8.0, 8.0, 8.0]),
        "mean_sfr10": np.array([0.003, 0.0032, 0.0031]),
        "time": time,
        "k_edges": edges,
        "k_centers": k,
        "power_delta": power,
        "anisotropy_amplitude": amplitude,
        "anisotropy_band_amplitude_time": band_amplitude,
        "anisotropy_band_angle_rad_time": band_angle,
    }


def test_box_size_summary_and_figures(tmp_path):
    data = _synthetic_archive()

    summary, diagnostics = box_size_diagnostic_summary(data)
    spectrum_output = tmp_path / "spectrum.png"
    diagnostic_output = tmp_path / "diagnostics.png"
    plot_box_size_time_mean(data, spectrum_output, dpi=40)
    plot_box_size_diagnostics(summary, diagnostic_output, dpi=40)

    assert diagnostics["integral_scale_time_pc"].shape == (3, 2)
    np.testing.assert_allclose(
        summary["spectral_slope_alpha_time_median"], [2.0, 2.2, 2.4]
    )
    assert spectrum_output.stat().st_size > 0
    assert diagnostic_output.stat().st_size > 0


def test_box_size_2d_figure_accepts_different_fourier_grid_shapes(tmp_path):
    results = []
    for size in (8, 16, 32):
        modes = np.arange(-size // 2, size // 2, dtype=float)
        yy, xx = np.meshgrid(modes, modes, indexing="ij")
        results.append(
            {
                "model": np.asarray(f"L{size}"),
                "time_bounds": np.asarray([200.0, 600.0]),
                "mean_power_2d": 1.0 / (1.0 + xx**2 + yy**2),
                "kx_mode": modes,
                "ky_mode": modes,
                "box_size_xy_pc": np.asarray([size, size], dtype=float),
            }
        )

    output = tmp_path / "power2d.png"
    plot_box_size_time_mean_power2d(results, output, mode_limit=3.0, dpi=40)

    assert output.stat().st_size > 0
