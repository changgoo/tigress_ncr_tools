from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_xz import (
    create_xz_grid_figure,
    discover_xz_models,
    nearest_xz_paths,
    load_xz_maps_with_vtk_fallback,
    nearest_vtk_snapshot,
    vtk_snapshot_index,
    xz_surface_density_from_vtk,
    update_xz_grid_figure,
    xz_number_index,
    xz_surface_density_map,
)


def test_xz_surface_density_uses_projected_cell_area(monkeypatch):
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_xz.star_particle_units",
        lambda: {"mass_msun": 2.0},
    )
    pdf = {
        "id": "x1-x3",
        "binx_name": "x1",
        "biny_name": "x3",
        "binx_log": 0,
        "biny_log": 0,
        "Nbinx": 2,
        "Nbiny": 3,
        "x_spacing": 2.0,
        "y_spacing": 4.0,
        "x_edges": np.array([-2.0, 0.0, 2.0]),
        "y_edges": np.array([-6.0, -2.0, 2.0, 6.0]),
        "weights": {"nH": np.full((3, 2), 8.0)},
    }
    data, extent = xz_surface_density_map(pdf)
    np.testing.assert_allclose(data, 2.0)
    assert extent == (-2.0, 2.0, -6.0, 6.0)

    bad = dict(pdf, biny_name="x2")
    with pytest.raises(ValueError, match="x1 and x3"):
        xz_surface_density_map(bad)


def test_xz_discovery_index_and_time_alignment(tmp_path, monkeypatch):
    model = tmp_path / "R8_8pc_NCR_row0000"
    (model / "hst").mkdir(parents=True)
    (model / "hst" / "model.hst").touch()
    directory = model / "pdf2d" / "x1-x3"
    directory.mkdir(parents=True)
    first = directory / "model.0300.x1-x3.pdf2d"
    second = directory / "model.0302.x1-x3.pdf2d"
    first.touch()
    second.touch()

    assert discover_xz_models(tmp_path) == [model]
    index = xz_number_index(model)
    assert index == {300: first, 302: second}

    times = {first: 299.0, second: 300.002}
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_xz.read_pdf2d_metadata",
        lambda path: {"time": times[path]},
    )
    paths, stored_times, offsets = nearest_xz_paths([index], 300.0)
    assert paths == [second]
    assert stored_times.tolist() == [300.002]
    assert offsets == [2]


def _write_vtk_density(path, density, origin, *, spacing=(2.0, 1.0, 2.0), time=100.0):
    shape = np.asarray(density.shape[::-1], dtype=int)
    dimensions = shape + 1
    header = (
        "# vtk DataFile Version 2.0\n"
        f"PRIMITIVE vars at time= {time:.6e}, level= 0, domain= 0\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {dimensions[0]} {dimensions[1]} {dimensions[2]}\n"
        f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
        f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n"
        f"CELL_DATA {density.size}\n"
        "SCALARS density float\n"
        "LOOKUP_TABLE default\n"
    ).encode("ascii")
    path.write_bytes(header + np.asarray(density, dtype=">f4").tobytes())


def test_vtk_xz_reconstruction_and_native_fallback(tmp_path, monkeypatch):
    model = tmp_path / "R8_8pc_NCR_row0000"
    snapshot = model / "vtk" / "0010"
    snapshot.mkdir(parents=True)
    _write_vtk_density(
        snapshot / "model-id0.0010.vtk",
        np.ones((1, 2, 2)),
        (0.0, 0.0, -1.0),
    )
    _write_vtk_density(
        snapshot / "model-id1.0010.vtk",
        np.full((1, 2, 2), 3.0),
        (0.0, 2.0, -1.0),
    )
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_xz.star_particle_units",
        lambda: {"mass_msun": 2.0},
    )

    index = vtk_snapshot_index(model)
    assert nearest_vtk_snapshot(index, 100.0) == (snapshot, 100.0)
    data, extent = xz_surface_density_from_vtk(snapshot)
    np.testing.assert_allclose(data, 16.0)
    assert extent == (0.0, 4.0, -1.0, 1.0)

    malformed = {
        "id": "x1-x3",
        "time": 100.0,
        "binx_name": "x1",
        "biny_name": "x3",
        "binx_log": 0,
        "biny_log": 0,
        "Nbinx": 2,
        "Nbiny": 1,
        "x_spacing": 2.0,
        "y_spacing": 2.0,
        "x_edges": np.array([0.0, 2.0, 4.0]),
        "y_edges": np.array([-1.0, 1.0]),
        "weights": {"nH": np.zeros((1, 2))},
    }
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_xz.read_pdf2d",
        lambda path, fields=None: malformed,
    )
    maps, fallback_extent, sources = load_xz_maps_with_vtk_fallback(
        [Path("malformed.pdf2d")],
        [index],
        100.0,
        reference_extent=extent,
    )
    np.testing.assert_allclose(maps[0], data)
    assert fallback_extent == extent
    assert sources[0]["source"] == "vtk-reconstructed"
    assert sources[0]["replaced_pdf2d"] == "malformed.pdf2d"


def test_xz_grid_preserves_physical_aspect_and_updates():
    ranked = [
        (Path(f"R8_8pc_NCR_row{index:04d}"), float(4 - index))
        for index in range(4)
    ]
    maps = [np.full((8, 2), index + 1.0) for index in range(4)]
    extent = (-1.0, 1.0, -4.0, 4.0)
    fig, images, time_text = create_xz_grid_figure(
        ranked,
        maps,
        extent,
        12.0,
        nrows=2,
        ncols=2,
    )
    assert len(images) == 4
    assert images[0].get_extent() == list(extent)
    assert fig.axes[0].get_aspect() == 1.0
    assert "01 row0000" in fig.axes[0].texts[0].get_text()

    update_xz_grid_figure(images, time_text, maps[::-1], 13.0)
    assert np.all(images[0].get_array() == 4.0)
    assert "13.0" in time_text.get_text()
    fig.clear()
