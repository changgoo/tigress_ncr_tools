import numpy as np

from tigress_ncr_tools.plot_hi_fraction_mask import (
    face_on_hi_columns_from_vtk,
    hi_mask_statistics,
    normalized_log_pdf,
)


def _write_vtk(path, density, xhi, origin, spacing=(1.0, 1.0, 2.0)):
    shape = np.asarray(density.shape[::-1], dtype=int)
    dimensions = shape + 1
    header = (
        "# vtk DataFile Version 2.0\n"
        "PRIMITIVE vars at time= 1.000000e+02, level= 0, domain= 0\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {dimensions[0]} {dimensions[1]} {dimensions[2]}\n"
        f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
        f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n"
        f"CELL_DATA {density.size}\n"
        "SCALARS density float\n"
        "LOOKUP_TABLE default\n"
    ).encode("ascii")
    middle = b"\nSCALARS xHI float\nLOOKUP_TABLE default\n"
    path.write_bytes(
        header
        + np.asarray(density, dtype=">f4").tobytes()
        + middle
        + np.asarray(xhi, dtype=">f4").tobytes()
    )


def test_face_on_hi_columns_apply_cell_mask(tmp_path):
    snapshot = tmp_path / "vtk" / "0001"
    snapshot.mkdir(parents=True)
    density = np.ones((2, 1, 2))
    _write_vtk(
        snapshot / "model.0001.vtk",
        density,
        np.array([[[0.005, 0.5]], [[0.02, 0.25]]]),
        (0.0, 0.0, 0.0),
    )
    _write_vtk(
        snapshot / "model-id1.0001.vtk",
        2.0 * density,
        np.array([[[0.0, 0.01]], [[0.5, 0.005]]]),
        (2.0, 0.0, 0.0),
    )

    result = face_on_hi_columns_from_vtk(snapshot, xhi_min=0.01)
    np.testing.assert_allclose(result["raw"], [[0.05, 1.5, 2.0, 0.06]])
    np.testing.assert_allclose(result["masked"], [[0.04, 1.5, 2.0, 0.04]])
    np.testing.assert_array_equal(result["coverage"], [[2, 2, 2, 2]])
    stats = hi_mask_statistics(result["raw"], result["masked"])
    assert np.isclose(stats["global_hi_removed_fraction"], 0.03 / 3.61)
    assert stats["zero_sightline_fraction_after_mask"] == 0.0


def test_normalized_log_pdf_retains_zero_probability():
    edges = np.array([-1.0, 0.0, 1.0])
    pdf = normalized_log_pdf(np.array([0.0, 1.0, 1.0]), edges)
    assert np.isclose(np.sum(pdf * np.diff(edges)), 2.0 / 3.0)
