import os

import numpy as np

from pathena.slicevtk_reader import (
    index_slicevtk_series,
    read_slicevtk,
    read_slicevtk_metadata,
)


def write_field(stream, name, values, ncomp=1, vtk_type="float"):
    array = np.asarray(values)
    if ncomp > 1:
        array = array.reshape(-1, ncomp)
        ntuple = array.shape[0]
    else:
        array = array.reshape(-1)
        ntuple = array.size
    dtype = {"float": ">f4", "int": ">i4"}[vtk_type]
    stream.write(f"{name} {ncomp} {ntuple} {vtk_type}\n".encode("ascii"))
    stream.write(array.astype(dtype).tobytes())
    stream.write(b"\n")


def write_slicevtk(path, time):
    with path.open("wb") as stream:
        stream.write(b"# vtk DataFile Version 2.0\n")
        stream.write(
            f"SLICEVTK midplane time={time:.8f} out=slice\n".encode("ascii")
        )
        stream.write(b"BINARY\n")
        stream.write(b"DATASET FIELD\n")
        stream.write(b"FIELD FieldData 7\n")
        write_field(stream, "time", [time])
        write_field(stream, "x3_dims", [3, 3, 2], vtk_type="int")
        write_field(stream, "x3_origin", [-1.0, -1.0, -0.5])
        write_field(stream, "x3_spacing", [1.0, 1.0, 1.0])
        write_field(stream, "x3_slice_coord", [0.0, 0.5])
        write_field(stream, "x3_density", [1.0, 2.0, 3.0, 4.0])
        write_field(
            stream,
            "x3_velocity",
            np.arange(12, dtype=float).reshape(4, 3),
            ncomp=3,
        )


def test_read_slicevtk_metadata_skips_field_payloads(tmp_path):
    path = tmp_path / "R8_8pc_NCR.0042.midplane.slice.vtk"
    write_slicevtk(path, 12.5)

    metadata = read_slicevtk_metadata(path)
    assert metadata["num"] == "0042"
    assert metadata["id"] == "midplane"
    assert metadata["time"] == 12.5
    assert metadata["plane_names"] == ["x3"]
    assert metadata["field_names"] == {"x3": ["density", "velocity"]}
    assert metadata["arrays"][-1] == {
        "name": "x3_velocity",
        "ncomp": 3,
        "ntuple": 4,
        "dtype": "float",
    }

    frame = read_slicevtk(path)
    assert frame["path"] == str(path)
    assert frame["planes"]["x3"]["fields"]["density"].shape == (2, 2)
    assert frame["planes"]["x3"]["fields"]["velocity"].shape == (2, 2, 3)


def test_index_slicevtk_series_sorts_time_and_deduplicates_restarts(tmp_path):
    early = tmp_path / "R8.0001.midplane.slice.vtk"
    replacement = tmp_path / "R8.0099.midplane.slice.vtk"
    late = tmp_path / "R8.0002.midplane.slice.vtk"
    write_slicevtk(late, 2.0)
    write_slicevtk(early, 1.0)
    write_slicevtk(replacement, 1.005)
    os.utime(early, ns=(1_000_000_000, 1_000_000_000))
    os.utime(replacement, ns=(2_000_000_000, 2_000_000_000))

    records = index_slicevtk_series(
        tmp_path / "*.slice.vtk", time_tolerance=0.01
    )
    assert [record["time"] for record in records] == [1.005, 2.0]
    assert [record["num"] for record in records] == ["0099", "0002"]


def test_index_slicevtk_series_empty_and_tolerance_validation(tmp_path):
    assert index_slicevtk_series(tmp_path / "*.slice.vtk") == []
    try:
        index_slicevtk_series(tmp_path / "*.slice.vtk", time_tolerance=-1.0)
    except ValueError as error:
        assert "time_tolerance" in str(error)
    else:
        raise AssertionError("negative time tolerance should fail")
