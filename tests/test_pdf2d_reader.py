from pathlib import Path

import numpy as np
import pytest

from pathena.pdf2d_reader import read_pdf2d, read_pdf2d_metadata


def write_pdf2d(path):
    header = (
        "# vtk DataFile Version 2.0\n"
        "PDF2D x1-x3 at time= 3.250000e+00, axes=x1,x3, log=0,0\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        "DIMENSIONS 3 3 1\n"
        "ORIGIN -1 -2 0\n"
        "SPACING 1 2 1\n"
        "CELL_DATA 4\n"
    ).encode("ascii")
    first = np.arange(4).astype(">f4").tobytes()
    second = (np.arange(4) + 10.0).astype(">f4").tobytes()
    payload = (
        header
        + b"\nSCALARS first float 1\nLOOKUP_TABLE default\n"
        + first
        + b"\nSCALARS second float 1\nLOOKUP_TABLE default\n"
        + second
        + b"\n"
    )
    Path(path).write_bytes(payload)


def test_read_pdf2d_metadata_does_not_require_scalar_scan(tmp_path):
    path = tmp_path / "model.0003.x1-x3.pdf2d"
    write_pdf2d(path)
    metadata = read_pdf2d_metadata(path)
    assert metadata["id"] == "x1-x3"
    assert metadata["time"] == 3.25
    assert metadata["binx_name"] == "x1"
    assert metadata["biny_name"] == "x3"


def test_read_pdf2d_can_select_and_skip_weights(tmp_path):
    path = tmp_path / "model.0003.x1-x3.pdf2d"
    write_pdf2d(path)

    selected = read_pdf2d(path, fields="second")
    assert selected["weight_names"] == ["first", "second"]
    assert list(selected["weights"]) == ["second"]
    np.testing.assert_allclose(
        selected["weights"]["second"],
        [[10.0, 11.0], [12.0, 13.0]],
    )
    assert selected["x_edges"].tolist() == [-1.0, 0.0, 1.0]
    assert selected["y_edges"].tolist() == [-2.0, 0.0, 2.0]

    complete = read_pdf2d(path)
    assert list(complete["weights"]) == ["first", "second"]
    with pytest.raises(KeyError, match="missing"):
        read_pdf2d(path, fields="not_present")
