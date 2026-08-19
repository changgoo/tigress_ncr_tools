from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_evolution import (
    create_grid_figure,
    create_phase_grid_figure,
    hydrogen_phase_rgb,
    nearest_projection_paths,
    projection_path,
    short_model_name,
    time_average,
    update_grid_figure,
    update_phase_grid_figure,
    write_model_order,
)


def test_time_average_interpolates_requested_boundaries():
    time = np.array([0.0, 1.0, 3.0, 4.0])
    values = 2.0 * time
    assert time_average(time, values, (0.5, 3.5)) == pytest.approx(4.0)


def test_time_average_requires_full_coverage():
    with pytest.raises(ValueError, match="history covers"):
        time_average([1.0, 2.0], [3.0, 4.0], (0.0, 2.0))

def test_nearest_projection_paths_uses_time_not_output_number(monkeypatch):
    paths_by_time = {
        Path("a.0300.proj2d"): 299.0,
        Path("a.0301.proj2d"): 300.002,
        Path("b.0300.proj2d"): 298.0,
        Path("b.0302.proj2d"): 299.999,
    }
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_evolution.read_proj2d_metadata",
        lambda path: {"time": paths_by_time[path]},
    )
    indices = [
        {300: Path("a.0300.proj2d"), 301: Path("a.0301.proj2d")},
        {300: Path("b.0300.proj2d"), 302: Path("b.0302.proj2d")},
    ]
    paths, times, offsets = nearest_projection_paths(indices, 300.0)
    assert paths == [Path("a.0301.proj2d"), Path("b.0302.proj2d")]
    assert times.tolist() == [300.002, 299.999]
    assert offsets == [1, 2]
    with pytest.raises(ValueError, match="offset"):
        nearest_projection_paths(indices, 299.5)




def test_projection_path_and_short_name(tmp_path):
    model = tmp_path / "R8_8pc_NCR_row0010"
    directory = model / "proj2d" / "theta0"
    directory.mkdir(parents=True)
    expected = directory / "R8_8pc_NCR.0300.theta0.proj2d"
    expected.touch()
    assert projection_path(model, 300) == expected
    assert short_model_name(model) == "row0010"


def test_grid_figure_is_row_major_and_reusable(tmp_path):
    ranked = [
        (Path(f"R8_8pc_NCR_row{index:04d}"), float(4 - index))
        for index in range(4)
    ]
    maps = [np.full((3, 3), index + 1.0) for index in range(4)]
    fig, images, time_text = create_grid_figure(
        ranked, maps, 12.0, nrows=2, ncols=2
    )
    assert len(images) == 4
    assert "01 row0000" in fig.axes[0].texts[0].get_text()
    assert "04 row0003" in fig.axes[3].texts[0].get_text()
    update_grid_figure(images, time_text, maps[::-1], 13.0)
    assert np.all(images[0].get_array() == 4.0)
    assert "13.0" in time_text.get_text()
    fig.clear()

    order = tmp_path / "model_order.csv"
    write_model_order(ranked, order)
    lines = order.read_text().splitlines()
    assert lines[0] == "rank,model,mean_sfr10,time_start,time_stop"
    assert lines[1].startswith("1,R8_8pc_NCR_row0000,4,")


def test_hydrogen_phase_rgb_counts_nuclei_and_uses_fixed_stretch(monkeypatch):
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_evolution.star_particle_units",
        lambda: {"mass_msun": 1.0},
    )
    rgb = hydrogen_phase_rgb(
        np.array([[6.0]]),
        np.array([[2.0]]),
        np.array([[1.0]]),
        scale=2.0,
        asinh_q=1.0,
        hi_green_scale=0.65,
    )
    assert rgb.shape == (1, 1, 3)
    assert np.allclose(rgb[0, 0], (1.0, 0.65, 1.0))

    clipped = hydrogen_phase_rgb(
        np.array([[1.0]]),
        np.array([[2.0]]),
        np.array([[1.0]]),
        scale=2.0,
        asinh_q=1.0,
    )
    assert clipped[0, 0, 2] == 0.0
    with pytest.raises(ValueError, match="matching shapes"):
        hydrogen_phase_rgb(np.ones((1, 2)), np.ones((2, 1)), np.ones((1, 2)))


def test_phase_grid_figure_is_reusable():
    ranked = [
        (Path(f"R8_8pc_NCR_row{index:04d}"), float(4 - index))
        for index in range(4)
    ]
    maps = [np.full((3, 3, 3), index / 4.0) for index in range(4)]
    fig, images, time_text = create_phase_grid_figure(
        ranked, maps, 12.0, nrows=2, ncols=2, scale=25.0, asinh_q=10.0
    )
    assert len(images) == 4
    assert "01 row0000" in fig.axes[0].texts[0].get_text()
    assert any("2" in text.get_text() for text in fig.texts)
    update_phase_grid_figure(images, time_text, maps[::-1], 13.0)
    assert np.allclose(images[0].get_array(), 0.75)
    assert "13.0" in time_text.get_text()
    fig.clear()
