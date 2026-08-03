"""Lightweight selected-field reader for distributed Athena legacy VTK data."""

import os
import re
import tarfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .vtk_helper import axis_arrays, read_ascii_line, read_vtk_magic


_TIME_RE = re.compile(r"time=\s*(?P<time>[-+0-9.eE]+)")
_RANK_RE = re.compile(r"-id(?P<rank>\d+)\.")
_VTK_DTYPES = {
    "float": np.dtype(">f4"),
    "double": np.dtype(">f8"),
    "int": np.dtype(">i4"),
    "unsigned_int": np.dtype(">u4"),
    "long": np.dtype(">i8"),
    "unsigned_long": np.dtype(">u8"),
}


@dataclass(frozen=True)
class VtkFieldRecord:
    """Location and representation of one field inside one VTK piece."""

    name: str
    ncomp: int
    vtk_dtype: str
    data_offset: int
    nvalues: int
    nbytes: int


@dataclass(frozen=True)
class VtkArchiveMember:
    """A seekable VTK file stored inside an uncompressed snapshot tar."""

    archive_path: str
    member_name: str
    size: int

    @property
    def name(self):
        return Path(self.member_name).name

    def __str__(self):
        return f"{self.archive_path}::{self.member_name}"


@dataclass(frozen=True)
class VtkPiece:
    """Metadata for one rank-local structured-points VTK file."""

    path: str
    time: float
    point_dims: tuple
    shape_xyz: tuple
    origin: tuple
    spacing: tuple
    right_edge: tuple
    ncells: int
    fields: dict
    source: object


@dataclass(frozen=True)
class VtkPlacement:
    """One piece plus its integer global cell-index bounds."""

    piece: VtkPiece
    start_xyz: tuple
    stop_xyz: tuple


@dataclass(frozen=True)
class VtkVolumeLayout:
    """Validated global geometry for a collection of rank pieces."""

    time: float
    left_edge: tuple
    right_edge: tuple
    spacing: tuple
    shape_xyz: tuple
    placements: tuple
    field_names: tuple


def _field_value_count(name, ncomp, ncells, shape_xyz):
    nx, ny, nz = shape_xyz
    if name == "face_centered_B1":
        return (nx + 1) * ny * nz
    if name == "face_centered_B2":
        return nx * (ny + 1) * nz
    if name == "face_centered_B3":
        return nx * ny * (nz + 1)
    return ncomp * ncells


def _consume_binary_tail(stream):
    tail = stream.read(1)
    if tail not in (b"", b"\n"):
        stream.seek(-1, os.SEEK_CUR)


@contextmanager
def _open_vtk_source(source, archive=None):
    if isinstance(source, VtkArchiveMember):
        if archive is None:
            with tarfile.open(source.archive_path, "r") as opened:
                stream = opened.extractfile(source.member_name)
                if stream is None:
                    raise ValueError(f"cannot read archive member {source}")
                with stream:
                    yield stream
        else:
            stream = archive.extractfile(source.member_name)
            if stream is None:
                raise ValueError(f"cannot read archive member {source}")
            with stream:
                yield stream
    else:
        with Path(source).open("rb") as stream:
            yield stream


@contextmanager
def _open_source_archives(sources):
    archive_paths = sorted({
        source.archive_path
        for source in sources
        if isinstance(source, VtkArchiveMember)
    })
    with ExitStack() as stack:
        archives = {
            path: stack.enter_context(tarfile.open(path, "r"))
            for path in archive_paths
        }
        yield archives


def read_vtk_piece_metadata(path, _archive=None):
    """Read one Athena VTK piece's geometry and field map without payloads."""
    source = path
    if isinstance(source, VtkArchiveMember):
        file_size = source.size
        display_path = str(source)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        file_size = path.stat().st_size
        display_path = str(path)
        source = path
    with _open_vtk_source(source, archive=_archive) as stream:
        read_vtk_magic(stream, display_path)
        description = read_ascii_line(stream).strip()
        time_match = _TIME_RE.search(description)
        if time_match is None:
            raise ValueError(
                f"{display_path}: cannot parse time from {description!r}"
            )
        if read_ascii_line(stream).strip() != "BINARY":
            raise ValueError(f"{display_path}: expected BINARY VTK data")
        dataset = read_ascii_line(stream).strip()
        if dataset != "DATASET STRUCTURED_POINTS":
            raise ValueError(
                f"{display_path}: expected DATASET STRUCTURED_POINTS, got {dataset!r}"
            )

        point_dims = None
        origin = None
        spacing = None
        ncells = None
        while ncells is None:
            line = read_ascii_line(stream)
            if not line:
                raise ValueError(f"{display_path}: EOF before CELL_DATA")
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "DIMENSIONS":
                point_dims = tuple(int(value) for value in tokens[1:4])
            elif tokens[0] == "ORIGIN":
                origin = tuple(float(value) for value in tokens[1:4])
            elif tokens[0] == "SPACING":
                spacing = tuple(float(value) for value in tokens[1:4])
            elif tokens[0] == "CELL_DATA":
                ncells = int(tokens[1])
        if point_dims is None or origin is None or spacing is None:
            raise ValueError(
                f"{display_path}: incomplete structured-points geometry"
            )
        shape_xyz = tuple(max(value - 1, 1) for value in point_dims)
        if int(np.prod(shape_xyz)) != ncells:
            raise ValueError(
                f"{display_path}: CELL_DATA {ncells} does not match dimensions "
                f"{shape_xyz}"
            )
        if any(value <= 0.0 for value in spacing):
            raise ValueError(
                f"{display_path}: spacing must be positive, got {spacing}"
            )

        fields = {}
        while True:
            line = read_ascii_line(stream)
            if not line:
                break
            tokens = line.strip().split()
            if not tokens:
                continue
            kind = tokens[0]
            if kind not in ("SCALARS", "VECTORS") or len(tokens) < 3:
                raise ValueError(
                    f"{display_path}: malformed field header {line!r}"
                )
            name, vtk_dtype = tokens[1], tokens[2]
            if vtk_dtype not in _VTK_DTYPES:
                raise ValueError(
                    f"{display_path}: field {name} has unsupported dtype {vtk_dtype!r}"
                )
            if kind == "SCALARS":
                ncomp = int(tokens[3]) if len(tokens) > 3 else 1
                lookup = read_ascii_line(stream).strip()
                if not lookup.startswith("LOOKUP_TABLE"):
                    raise ValueError(
                        f"{display_path}: field {name} is missing LOOKUP_TABLE"
                    )
            else:
                ncomp = 3
            data_offset = stream.tell()
            nvalues = _field_value_count(name, ncomp, ncells, shape_xyz)
            nbytes = nvalues * _VTK_DTYPES[vtk_dtype].itemsize
            if data_offset + nbytes > file_size:
                raise ValueError(
                    f"{display_path}: short payload for field {name}; "
                    f"expected {nbytes} bytes"
                )
            fields[name] = VtkFieldRecord(
                name=name,
                ncomp=ncomp,
                vtk_dtype=vtk_dtype,
                data_offset=data_offset,
                nvalues=nvalues,
                nbytes=nbytes,
            )
            stream.seek(nbytes, os.SEEK_CUR)
            _consume_binary_tail(stream)

    right_edge = tuple(
        left + count * delta
        for left, count, delta in zip(origin, shape_xyz, spacing)
    )
    return VtkPiece(
        path=display_path,
        time=float(time_match.group("time")),
        point_dims=point_dims,
        shape_xyz=shape_xyz,
        origin=origin,
        spacing=spacing,
        right_edge=right_edge,
        ncells=ncells,
        fields=fields,
        source=source,
    )


def _piece_rank(path):
    name = path.name if isinstance(path, VtkArchiveMember) else Path(path).name
    match = _RANK_RE.search(name)
    return int(match.group("rank")) if match else 0


def discover_vtk_pieces(run_dir, problem_id, num):
    """Discover one output's pieces in new-output, MPI, or flat layouts."""
    run_dir = Path(run_dir).expanduser()
    number = str(num)
    if number.isdigit():
        number = f"{int(number):04d}"
    output_dir = run_dir / "vtk" / number
    patterns = [
        output_dir / f"{problem_id}*.{number}.vtk",
        run_dir / "id0" / f"{problem_id}.{number}.vtk",
        run_dir / f"{problem_id}.{number}.vtk",
    ]
    for pattern in patterns:
        paths = list(pattern.parent.glob(pattern.name))
        if not paths:
            continue
        if pattern.parent.name == "id0":
            paths.extend(
                path
                for directory in run_dir.glob("id[1-9]*")
                for path in directory.glob(f"{problem_id}-id*.{number}.vtk")
            )
        return sorted(set(paths), key=lambda path: (_piece_rank(path), str(path)))
    archives = (
        run_dir / "vtk" / f"{problem_id}.{number}.tar",
        run_dir / f"{problem_id}.{number}.tar",
    )
    archive_path = next((path for path in archives if path.is_file()), None)
    if archive_path is not None:
        pattern = re.compile(
            rf"^(?:.*/)?{re.escape(problem_id)}"
            rf"(?:-id\d+)?\.{re.escape(number)}\.vtk$"
        )
        with tarfile.open(archive_path, "r") as archive:
            members = [
                VtkArchiveMember(
                    str(archive_path), member.name, member.size
                )
                for member in archive.getmembers()
                if member.isfile() and pattern.fullmatch(member.name)
            ]
        if not members:
            raise ValueError(f"archive has no matching VTK members: {archive_path}")
        return sorted(members, key=lambda item: (_piece_rank(item), item.member_name))
    raise FileNotFoundError(
        f"cannot find VTK output {problem_id}.{number} under {run_dir}"
    )


def _boxes_overlap(first, second):
    return all(
        a0 < b1 and b0 < a1
        for a0, a1, b0, b1 in zip(
            first.start_xyz, first.stop_xyz,
            second.start_xyz, second.stop_xyz,
        )
    )


def inspect_vtk_volume(paths, time_tolerance=0.01):
    """Inspect and validate a set of pieces as a complete uniform volume."""
    if time_tolerance < 0.0:
        raise ValueError("time_tolerance must be non-negative")
    paths = list(paths)
    with _open_source_archives(paths) as archives:
        pieces = [
            read_vtk_piece_metadata(
                path,
                _archive=(
                    archives[path.archive_path]
                    if isinstance(path, VtkArchiveMember)
                    else None
                ),
            )
            for path in paths
        ]
    if not pieces:
        raise ValueError("at least one VTK piece is required")
    reference_spacing = np.asarray(pieces[0].spacing)
    reference_time = pieces[0].time
    for piece in pieces[1:]:
        if not np.allclose(piece.spacing, reference_spacing, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"piece spacing mismatch: {piece.path} has {piece.spacing}, "
                f"expected {tuple(reference_spacing)}"
            )
        if abs(piece.time - reference_time) > time_tolerance:
            raise ValueError(
                f"piece time mismatch: {piece.path} has {piece.time:g}, "
                f"expected {reference_time:g}"
            )
    left_edge = np.min(np.asarray([piece.origin for piece in pieces]), axis=0)
    right_edge = np.max(
        np.asarray([piece.right_edge for piece in pieces]), axis=0
    )
    shape_float = (right_edge - left_edge) / reference_spacing
    shape_xyz = np.rint(shape_float).astype(int)
    if not np.allclose(shape_float, shape_xyz, rtol=0.0, atol=1e-8):
        raise ValueError("global VTK bounds do not align to common spacing")

    placements = []
    for piece in pieces:
        start_float = (
            np.asarray(piece.origin) - left_edge
        ) / reference_spacing
        start = np.rint(start_float).astype(int)
        if not np.allclose(start_float, start, rtol=0.0, atol=1e-8):
            raise ValueError(f"piece origin is off-grid: {piece.path}")
        stop = start + np.asarray(piece.shape_xyz)
        placements.append(VtkPlacement(
            piece=piece,
            start_xyz=tuple(int(value) for value in start),
            stop_xyz=tuple(int(value) for value in stop),
        ))

    for index, first in enumerate(placements):
        for second in placements[index + 1:]:
            if _boxes_overlap(first, second):
                raise ValueError(
                    f"VTK pieces overlap: {first.piece.path} and {second.piece.path}"
                )
    covered_cells = sum(piece.ncells for piece in pieces)
    expected_cells = int(np.prod(shape_xyz))
    if covered_cells != expected_cells:
        raise ValueError(
            f"VTK pieces leave gaps: cover {covered_cells} of "
            f"{expected_cells} global cells"
        )

    common_fields = set(pieces[0].fields)
    for piece in pieces[1:]:
        common_fields.intersection_update(piece.fields)
    for name in common_fields:
        signature = (
            pieces[0].fields[name].ncomp,
            pieces[0].fields[name].vtk_dtype,
        )
        for piece in pieces[1:]:
            record = piece.fields[name]
            if (record.ncomp, record.vtk_dtype) != signature:
                raise ValueError(
                    f"field representation mismatch for {name!r} in {piece.path}"
                )
    return VtkVolumeLayout(
        time=float(np.mean([piece.time for piece in pieces])),
        left_edge=tuple(float(value) for value in left_edge),
        right_edge=tuple(float(value) for value in right_edge),
        spacing=tuple(float(value) for value in reference_spacing),
        shape_xyz=tuple(int(value) for value in shape_xyz),
        placements=tuple(placements),
        field_names=tuple(sorted(common_fields)),
    )


def estimate_volume_bytes(layout, field_names, dtype=np.float32):
    """Estimate assembled array bytes for selected common fields."""
    dtype = np.dtype(dtype)
    total_cells = int(np.prod(layout.shape_xyz))
    total = 0
    first_piece = layout.placements[0].piece
    for name in field_names:
        if name not in layout.field_names:
            raise KeyError(f"volume is missing field {name!r}")
        total += total_cells * first_piece.fields[name].ncomp * dtype.itemsize
    return total


def read_vtk_piece_field(piece, name, dtype=np.float32, _archive=None):
    """Read one cell-centered scalar or vector field from one piece."""
    if name not in piece.fields:
        raise KeyError(f"{piece.path} is missing field {name!r}")
    record = piece.fields[name]
    if name.startswith("face_centered_B"):
        raise ValueError("face-centered fields are not supported for assembly")
    with _open_vtk_source(piece.source, archive=_archive) as stream:
        stream.seek(record.data_offset)
        raw = stream.read(record.nbytes)
    if len(raw) != record.nbytes:
        raise ValueError(f"{piece.path}: short read for field {name!r}")
    values = np.frombuffer(raw, dtype=_VTK_DTYPES[record.vtk_dtype])
    shape_zyx = tuple(reversed(piece.shape_xyz))
    shape = shape_zyx if record.ncomp == 1 else shape_zyx + (record.ncomp,)
    return values.reshape(shape).astype(dtype, copy=False)


def read_vtk_volume(paths, field_names, *, dtype=np.float32,
                    time_tolerance=0.01, max_bytes=None):
    """Assemble selected fields from rank pieces into native-order volumes."""
    if isinstance(field_names, str):
        field_names = [field_names]
    field_names = list(dict.fromkeys(field_names))
    if not field_names:
        raise ValueError("at least one field must be selected")
    layout = inspect_vtk_volume(paths, time_tolerance=time_tolerance)
    required_bytes = estimate_volume_bytes(layout, field_names, dtype=dtype)
    if max_bytes is not None and required_bytes > max_bytes:
        raise MemoryError(
            f"selected volume requires {required_bytes} bytes, exceeding "
            f"limit {max_bytes}"
        )
    shape_zyx = tuple(reversed(layout.shape_xyz))
    first_piece = layout.placements[0].piece
    arrays = {}
    for name in field_names:
        ncomp = first_piece.fields[name].ncomp
        shape = shape_zyx if ncomp == 1 else shape_zyx + (ncomp,)
        arrays[name] = np.empty(shape, dtype=dtype)

    sources = [placement.piece.source for placement in layout.placements]
    with _open_source_archives(sources) as archives:
        for placement in layout.placements:
            x0, y0, z0 = placement.start_xyz
            x1, y1, z1 = placement.stop_xyz
            target = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            source = placement.piece.source
            archive = (
                archives[source.archive_path]
                if isinstance(source, VtkArchiveMember)
                else None
            )
            for name in field_names:
                arrays[name][target] = read_vtk_piece_field(
                    placement.piece, name, dtype=dtype, _archive=archive
                )

    axes = {}
    for name, left, delta, count in zip(
        ("x", "y", "z"),
        layout.left_edge,
        layout.spacing,
        layout.shape_xyz,
    ):
        edges, centers = axis_arrays(left, delta, count)
        axes[name + "_edges"] = edges
        axes[name + "_centers"] = centers
    return {
        "time": layout.time,
        "left_edge": np.asarray(layout.left_edge),
        "right_edge": np.asarray(layout.right_edge),
        "spacing": np.asarray(layout.spacing),
        "shape_xyz": np.asarray(layout.shape_xyz),
        "fields": arrays,
        "layout": layout,
        "required_bytes": required_bytes,
        **axes,
    }
