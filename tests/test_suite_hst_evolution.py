from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_hst_evolution import (
    characteristic_speed,
    discover_history_models,
    history_speeds,
    plot_velocity_evolution,
    sfr_colormap,
    write_model_colors,
)


def test_characteristic_speed_masks_invalid_values():
    result = characteristic_speed(
        np.array([8.0, -1.0, 2.0]),
        np.array([4.0, 1.0, 0.0]),
        factor=2.0,
    )
    assert result[0] == pytest.approx(2.0)
    assert np.isnan(result[1:]).all()
    with pytest.raises(ValueError, match="matching shapes"):
        characteristic_speed(np.ones(2), np.ones(3))


def test_history_speeds_uses_requested_energy_and_pressure_formulas():
    history = {
        "mass": np.array([2.0]),
        "x1KE": np.array([4.0]),
        "x2dke": np.array([9.0]),
        "x3KE": np.array([16.0]),
        "P": np.array([8.0]),
        "x1ME": np.array([1.0]),
        "x2ME": np.array([4.0]),
        "x3ME": np.array([9.0]),
    }
    speeds = history_speeds(history)
    assert speeds["sigma_x1"][0] == pytest.approx(2.0)
    assert speeds["sigma_x2"][0] == pytest.approx(3.0)
    assert speeds["sigma_x3"][0] == pytest.approx(4.0)
    assert speeds["thermal"][0] == pytest.approx(2.0)
    assert speeds["alfven_x1"][0] == pytest.approx(1.0)
    assert speeds["alfven_x2"][0] == pytest.approx(2.0)
    assert speeds["alfven_x3"][0] == pytest.approx(3.0)


def test_sfr_colormap_and_color_key(tmp_path):
    values = np.array([1.0e-3, 1.0e-2])
    cmap, norm = sfr_colormap(values, scale="log")
    assert norm(values[0]) == pytest.approx(0.0)
    assert norm(values[1]) == pytest.approx(1.0)
    ranked = [
        (Path("R8_8pc_NCR_row0001"), values[1]),
        (Path("R8_8pc_NCR_row0002"), values[0]),
    ]
    output = tmp_path / "colors.csv"
    write_model_colors(ranked, output, cmap, norm)
    lines = output.read_text().splitlines()
    assert lines[0].startswith("rank,model,mean_sfr10,color_hex")
    assert "R8_8pc_NCR_row0001" in lines[1]
    assert "#" in lines[1]
    with pytest.raises(ValueError, match="positive"):
        sfr_colormap([0.0, 1.0], scale="log")


def test_discover_history_models_requires_primary_and_whole_files(tmp_path):
    complete = tmp_path / "R8_8pc_NCR_row0001" / "hst"
    complete.mkdir(parents=True)
    (complete / "run.hst").touch()
    (complete / "run.whole.hst").touch()
    incomplete = tmp_path / "R8_8pc_NCR_row0002" / "hst"
    incomplete.mkdir(parents=True)
    (incomplete / "run.hst").touch()
    assert discover_history_models(tmp_path) == [complete.parent]


def test_velocity_plot_contains_colored_tracks(monkeypatch, tmp_path):
    history = {
        "time": np.array([0.0, 1.0, 2.0]),
        "mass": np.ones(3),
        "x1KE": np.array([2.0, 4.0, 8.0]),
        "x2dke": np.array([1.0, 2.0, 4.0]),
        "x3KE": np.array([3.0, 5.0, 7.0]),
        "P": np.array([4.0, 5.0, 6.0]),
        "x1ME": np.array([1.0, 2.0, 3.0]),
        "x2ME": np.array([2.0, 3.0, 4.0]),
        "x3ME": np.array([3.0, 4.0, 5.0]),
    }
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_hst_evolution.whole_history_file",
        lambda model: model,
    )
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_hst_evolution.read_hst",
        lambda path, max_rows=None: history,
    )
    output = tmp_path / "speeds.png"
    ranked = [(Path("high"), 1.0e-2), (Path("low"), 1.0e-3)]
    plot_velocity_evolution(
        ranked,
        output,
        time_bounds=(0.0, 2.0),
        history_samples=3,
        dpi=60,
    )
    image = matplotlib.image.imread(output)
    upper_left = image[: image.shape[0] // 2, : image.shape[1] // 4, :3]
    chroma = np.ptp(upper_left, axis=-1)
    assert np.count_nonzero(chroma > 0.08) > 20
