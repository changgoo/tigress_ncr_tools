"""Small shared helpers for binary VTK legacy reader modules."""

import numpy as np

_VTK_DTYPE_MAP = {
    "float": ">f4",
    "double": ">f8",
    "int": ">i4",
    "unsigned_int": ">u4",
    "long": ">i8",
    "unsigned_long": ">u8",
}


def read_ascii_line(fp):
    """Read one ASCII line from a binary VTK file handle."""
    return fp.readline().decode("ascii", errors="replace")


def axis_arrays(origin, spacing, n):
    """Return cell-edge and cell-center arrays for uniform VTK axes."""
    edges = origin + spacing * np.arange(n + 1, dtype=np.float64)
    centers = origin + spacing * (np.arange(n, dtype=np.float64) + 0.5)
    return edges, centers


def read_vtk_magic(fp, path):
    """Read and validate the first VTK legacy header line."""
    line = read_ascii_line(fp)
    if "vtk DataFile Version" not in line:
        raise ValueError(f"{path}: not a VTK legacy file (line 1: {line!r})")
    return line


def read_structured_points_header(fp, path):
    """Read VTK STRUCTURED_POINTS geometry through CELL_DATA.

    Returns
    -------
    tuple
        ``(header_lines, dims, origin, spacing, cell_data)`` where ``dims`` is
        the raw VTK point dimensions, not cell counts.
    """
    header_lines = []
    dims = None
    origin = np.zeros(3, dtype=np.float64)
    spacing = np.ones(3, dtype=np.float64)
    cell_data = None
    while cell_data is None:
        line = read_ascii_line(fp)
        if not line:
            raise ValueError(f"{path}: unexpected EOF before CELL_DATA")
        header_lines.append(line.rstrip())
        stripped = line.strip()
        if stripped.startswith("DIMENSIONS"):
            dims = np.array([int(x) for x in stripped.split()[1:4]], dtype=np.int64)
        elif stripped.startswith("ORIGIN"):
            origin = np.array([float(x) for x in stripped.split()[1:4]], dtype=np.float64)
        elif stripped.startswith("SPACING"):
            spacing = np.array([float(x) for x in stripped.split()[1:4]], dtype=np.float64)
        elif stripped.startswith("CELL_DATA"):
            cell_data = int(stripped.split()[1])
    if dims is None:
        raise ValueError(f"{path}: DIMENSIONS not found in STRUCTURED_POINTS header")
    return header_lines, dims, origin, spacing, cell_data


def read_vtk_scalar_2d(fp, path, name, nx, ny):
    """Read one big-endian float scalar array with shape ``(ny, nx)``."""
    nbytes = nx * ny * 4
    raw = fp.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(
            f"{path}: short read for field {name}: got {len(raw)} bytes, expected {nbytes}"
        )
    return np.frombuffer(raw, dtype=">f4").reshape(ny, nx).astype(np.float64)


def read_vtk_field_array(fp, path):
    """Read one VTK legacy FIELD array record.

    Returns ``None`` at EOF. Otherwise returns
    ``(name, ncomp, ntuple, dtype_name, array)``. Scalar arrays are returned as
    shape ``(ntuple,)`` and multi-component arrays as ``(ntuple, ncomp)``.
    """
    while True:
        line = read_ascii_line(fp)
        if not line:
            return None
        if line.strip():
            break

    tok = line.strip().split()
    if len(tok) != 4:
        raise ValueError(f"{path}: malformed FIELD array header {line!r}")
    name, ncomp_s, ntuple_s, dtype_name = tok
    ncomp = int(ncomp_s)
    ntuple = int(ntuple_s)
    if dtype_name not in _VTK_DTYPE_MAP:
        raise ValueError(f"{path}: unsupported VTK FIELD dtype {dtype_name!r}")

    dtype = np.dtype(_VTK_DTYPE_MAP[dtype_name])
    nvals = ncomp * ntuple
    nbytes = nvals * dtype.itemsize
    raw = fp.read(nbytes)
    if len(raw) != nbytes:
        raise ValueError(
            f"{path}: short read for FIELD array {name}: got {len(raw)} bytes, expected {nbytes}"
        )
    arr = np.frombuffer(raw, dtype=dtype, count=nvals)
    tail = fp.read(1)
    if tail not in (b"", b"\n"):
        fp.seek(-1, os.SEEK_CUR)
    arr = arr.astype(np.float64 if dtype.kind == "f" else np.int64, copy=False)
    if ncomp > 1:
        arr = arr.reshape(ntuple, ncomp)
    return name, ncomp, ntuple, dtype_name, arr
