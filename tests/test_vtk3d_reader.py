import tarfile

import numpy as np
import pytest

from pathena.vtk3d_reader import (
    discover_vtk_pieces,
    estimate_volume_bytes,
    inspect_vtk_volume,
    read_vtk_piece_field,
    read_vtk_piece_metadata,
    read_vtk_volume,
)


def write_piece(path, origin, fields, shape_xyz=(2, 2, 2), time=3.0):
    nx, ny, nz = shape_xyz
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"# vtk DataFile Version 2.0\n")
        stream.write(
            f"PRIMITIVE vars at time= {time:.8e}, level= 0, domain= 0\n".encode()
        )
        stream.write(b"BINARY\n")
        stream.write(b"DATASET STRUCTURED_POINTS\n")
        stream.write(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n".encode())
        stream.write(
            f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n".encode()
        )
        stream.write(b"SPACING 1 1 1\n")
        stream.write(f"CELL_DATA {nx * ny * nz}\n".encode())
        for name, values in fields.items():
            values = np.asarray(values)
            if values.ndim == 4:
                assert values.shape == (nz, ny, nx, 3)
                stream.write(f"VECTORS {name} float\n".encode())
            else:
                assert values.shape == (nz, ny, nx)
                stream.write(f"SCALARS {name} float\n".encode())
                stream.write(b"LOOKUP_TABLE default\n")
            stream.write(values.astype(">f4").tobytes())
            stream.write(b"\n")


def piece_fields(x_offset, shape_xyz=(2, 2, 2)):
    nx, ny, nz = shape_xyz
    z, y, x = np.indices((nz, ny, nx))
    density = x_offset + x + 10.0 * y + 100.0 * z
    velocity = np.stack((density, density + 1.0, density + 2.0), axis=-1)
    return {
        "density": density,
        "pressure": density + 1000.0,
        "specific_scalar_3": np.full_like(density, 0.2),
        "specific_scalar_4": np.full_like(density, 0.1),
        "velocity": velocity,
    }


def two_piece_paths(tmp_path, second_origin=(2, 0, 0), second_time=3.0):
    first = tmp_path / "R8.0000.vtk"
    second = tmp_path / "R8-id1.0000.vtk"
    write_piece(first, (0, 0, 0), piece_fields(0.0))
    write_piece(
        second,
        second_origin,
        piece_fields(float(second_origin[0])),
        time=second_time,
    )
    return first, second


def test_piece_metadata_and_selected_field_read(tmp_path):
    path, _ = two_piece_paths(tmp_path)
    piece = read_vtk_piece_metadata(path)
    assert piece.time == 3.0
    assert piece.shape_xyz == (2, 2, 2)
    assert piece.origin == (0.0, 0.0, 0.0)
    assert piece.right_edge == (2.0, 2.0, 2.0)
    assert sorted(piece.fields) == [
        "density", "pressure", "specific_scalar_3",
        "specific_scalar_4", "velocity",
    ]
    assert piece.fields["velocity"].ncomp == 3
    density = read_vtk_piece_field(piece, "density")
    velocity = read_vtk_piece_field(piece, "velocity")
    assert density.shape == (2, 2, 2)
    assert velocity.shape == (2, 2, 2, 3)
    assert np.allclose(density, piece_fields(0.0)["density"])
    assert np.allclose(velocity, piece_fields(0.0)["velocity"])


def test_volume_layout_memory_and_assembly(tmp_path):
    paths = two_piece_paths(tmp_path)
    layout = inspect_vtk_volume(paths)
    assert layout.left_edge == (0.0, 0.0, 0.0)
    assert layout.right_edge == (4.0, 2.0, 2.0)
    assert layout.shape_xyz == (4, 2, 2)
    assert len(layout.placements) == 2
    assert estimate_volume_bytes(
        layout, ["density", "velocity"], dtype=np.float32
    ) == 4 * 2 * 2 * 4 * 4

    volume = read_vtk_volume(paths, ["density", "velocity"])
    assert volume["fields"]["density"].shape == (2, 2, 4)
    assert volume["fields"]["velocity"].shape == (2, 2, 4, 3)
    assert np.allclose(volume["fields"]["density"][:, :, :2],
                       piece_fields(0.0)["density"])
    assert np.allclose(volume["fields"]["density"][:, :, 2:],
                       piece_fields(2.0)["density"])
    assert np.allclose(volume["x_edges"], np.arange(5.0))
    assert np.allclose(volume["z_centers"], [0.5, 1.5])
    assert volume["required_bytes"] == 256

    with pytest.raises(MemoryError, match="exceeding"):
        read_vtk_volume(paths, ["density", "velocity"], max_bytes=255)
    with pytest.raises(KeyError, match="missing"):
        estimate_volume_bytes(layout, ["temperature"])


def test_layout_rejects_gaps_overlaps_and_time_mismatch(tmp_path):
    gap_dir = tmp_path / "gap"
    with pytest.raises(ValueError, match="gaps"):
        inspect_vtk_volume(two_piece_paths(gap_dir, second_origin=(3, 0, 0)))

    overlap_dir = tmp_path / "overlap"
    with pytest.raises(ValueError, match="overlap"):
        inspect_vtk_volume(two_piece_paths(overlap_dir, second_origin=(1, 0, 0)))

    time_dir = tmp_path / "time"
    with pytest.raises(ValueError, match="time mismatch"):
        inspect_vtk_volume(
            two_piece_paths(time_dir, second_time=4.0), time_tolerance=0.01
        )


def test_discover_new_output_directory_orders_rank_zero_first(tmp_path):
    output = tmp_path / "vtk" / "0007"
    fields = piece_fields(0.0)
    write_piece(output / "R8-id2.0007.vtk", (4, 0, 0), fields)
    write_piece(output / "R8.0007.vtk", (0, 0, 0), fields)
    write_piece(output / "R8-id1.0007.vtk", (2, 0, 0), fields)
    paths = discover_vtk_pieces(tmp_path, "R8", 7)
    assert [path.name for path in paths] == [
        "R8.0007.vtk", "R8-id1.0007.vtk", "R8-id2.0007.vtk"
    ]


def test_discover_and_assemble_archived_snapshot_without_extraction(tmp_path):
    output = tmp_path / "vtk" / "0008"
    first = output / "R8.0008.vtk"
    second = output / "R8-id1.0008.vtk"
    write_piece(first, (0, 0, 0), piece_fields(0.0))
    write_piece(second, (2, 0, 0), piece_fields(2.0))
    archive_path = tmp_path / "vtk" / "R8.0008.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(first, arcname=f"0008/{first.name}")
        archive.add(second, arcname=f"0008/{second.name}")
    first.unlink()
    second.unlink()
    output.rmdir()

    members = discover_vtk_pieces(tmp_path, "R8", 8)
    assert [member.name for member in members] == [
        "R8.0008.vtk", "R8-id1.0008.vtk"
    ]
    piece = read_vtk_piece_metadata(members[0])
    assert "R8.0008.tar::0008/R8.0008.vtk" in piece.path
    assert np.allclose(
        read_vtk_piece_field(piece, "density"), piece_fields(0.0)["density"]
    )

    volume = read_vtk_volume(members, ["density", "velocity"])
    assert volume["fields"]["density"].shape == (2, 2, 4)
    assert np.allclose(
        volume["fields"]["density"][:, :, :2], piece_fields(0.0)["density"]
    )
    assert np.allclose(
        volume["fields"]["density"][:, :, 2:], piece_fields(2.0)["density"]
    )
