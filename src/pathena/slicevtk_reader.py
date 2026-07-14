"""Reader for slicevtk output files written by Athena-TIGRESS-NCR.

The slicevtk format is a binary VTK legacy ``DATASET FIELD`` file containing
one or more gathered coordinate-plane slices. Each plane stores metadata arrays
and one FIELD array per raw primitive/conserved variable.

Typical use::

    from pathena.slicevtk_reader import read_slicevtk

    slc = read_slicevtk("slice/slice-x0/R8_8pc_NCR.0001.slice-x0.slice.vtk")
    slc["time"]
    slc["planes"]["x3"]["fields"]["density"]  # shape (Ny, Nx)
    slc["planes"]["x3"]["fields"]["velocity"] # shape (Ny, Nx, 3)
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from .vtk_helper import (
    axis_arrays,
    read_ascii_line,
    read_vtk_field_array,
    read_vtk_magic,
)

_HDR_RE = re.compile(
    r"^SLICEVTK\s+(?P<id>\S+)\s+"
    r"time=(?P<time>[-+0-9.eE]+)\s+"
    r"out=(?P<out>\S+)"
)
_PLANE_RE = re.compile(r"^(x[123])_(.+)$")

_META_NAMES = {"dims", "origin", "spacing", "slice_coord"}

def _plane_axes(plane):
    if plane == "x1":
        return "x2", "x3", 1, 2
    if plane == "x2":
        return "x1", "x3", 0, 2
    if plane == "x3":
        return "x1", "x2", 0, 1
    raise ValueError(f"unknown plane {plane!r}")


def _finalize_planes(raw_planes):
    planes = {}
    for plane, d in raw_planes.items():
        xaxis, yaxis, ax0, ax1 = _plane_axes(plane)
        dims = np.asarray(d.get("dims"), dtype=np.int64)
        origin = np.asarray(d.get("origin"), dtype=np.float64)
        spacing = np.asarray(d.get("spacing"), dtype=np.float64)
        slice_coord = np.asarray(d.get("slice_coord"), dtype=np.float64)
        if dims.size != 3 or origin.size != 3 or spacing.size != 3:
            raise ValueError(f"plane {plane}: missing or malformed geometry metadata")

        nx = int(dims[ax0] - 1)
        ny = int(dims[ax1] - 1)
        if nx <= 0 or ny <= 0:
            raise ValueError(f"plane {plane}: invalid dimensions {dims}")

        x_edges, x_centers = axis_arrays(origin[ax0], spacing[ax0], nx)
        y_edges, y_centers = axis_arrays(origin[ax1], spacing[ax1], ny)

        fields = {}
        field_names = []
        for name, arr in d.get("fields", {}).items():
            a = np.asarray(arr)
            if a.ndim == 1:
                if a.size != nx * ny:
                    raise ValueError(
                        f"plane {plane} field {name}: size {a.size}, expected {nx*ny}"
                    )
                a = a.reshape(ny, nx)
            else:
                if a.shape[0] != nx * ny:
                    raise ValueError(
                        f"plane {plane} field {name}: tuples {a.shape[0]}, expected {nx*ny}"
                    )
                a = a.reshape(ny, nx, a.shape[1])
            fields[name] = a
            field_names.append(name)

        planes[plane] = {
            "normal": plane,
            "xaxis": xaxis,
            "yaxis": yaxis,
            "dims": dims,
            "origin": origin,
            "spacing": spacing,
            "requested_coord": float(slice_coord[0]) if slice_coord.size else np.nan,
            "selected_coord": float(slice_coord[1]) if slice_coord.size > 1 else np.nan,
            "Nx": nx,
            "Ny": ny,
            "x_edges": x_edges,
            "x_centers": x_centers,
            "y_edges": y_edges,
            "y_centers": y_centers,
            "field_names": field_names,
            "fields": fields,
        }
    return planes


def read_slicevtk(path):
    """Read one slicevtk file.

    Parameters
    ----------
    path : str
        Path to ``*.slice.vtk``.

    Returns
    -------
    dict
        Keys include ``id``, ``time``, ``out``, ``plane_names``, and ``planes``.
        ``planes`` is a dict keyed by normal direction ``x1``, ``x2``, ``x3``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as fp:
        header_lines = []
        line = read_vtk_magic(fp, path)
        header_lines.append(line.rstrip())

        meta_line = read_ascii_line(fp).rstrip()
        header_lines.append(meta_line)
        m = _HDR_RE.match(meta_line)
        if not m:
            raise ValueError(f"{path}: cannot parse SLICEVTK metadata line {meta_line!r}")

        binary_line = read_ascii_line(fp).strip()
        dataset_line = read_ascii_line(fp).strip()
        field_line = read_ascii_line(fp).strip()
        header_lines.extend([binary_line, dataset_line, field_line])
        if binary_line != "BINARY":
            raise ValueError(f"{path}: expected BINARY, got {binary_line!r}")
        if dataset_line != "DATASET FIELD":
            raise ValueError(f"{path}: expected DATASET FIELD, got {dataset_line!r}")
        ftok = field_line.split()
        if len(ftok) != 3 or ftok[0] != "FIELD":
            raise ValueError(f"{path}: malformed FIELD header {field_line!r}")
        narrays = int(ftok[2])

        raw_planes = {}
        arrays = {}
        out = {
            "id": m.group("id"),
            "time": float(m.group("time")),
            "out": m.group("out"),
            "header_lines": header_lines,
            "arrays": arrays,
        }

        for _ in range(narrays):
            rec = read_vtk_field_array(fp, path)
            if rec is None:
                raise ValueError(f"{path}: EOF before reading all FIELD arrays")
            name, ncomp, ntuple, dtype_name, arr = rec
            arrays[name] = {
                "ncomp": ncomp,
                "ntuple": ntuple,
                "dtype": dtype_name,
                "data": arr,
            }
            if name == "time":
                out["time_array"] = arr
                continue
            pm = _PLANE_RE.match(name)
            if not pm:
                continue
            plane, suffix = pm.groups()
            pd = raw_planes.setdefault(plane, {"fields": {}})
            if suffix in _META_NAMES:
                pd[suffix] = np.ravel(arr)
            else:
                pd["fields"][suffix] = arr

        out["planes"] = _finalize_planes(raw_planes)
        out["plane_names"] = sorted(out["planes"], key=lambda p: int(p[1]))
    return out


def read_slicevtk_series(pattern):
    """Read every file matching a glob pattern, sorted by filename."""
    return [read_slicevtk(p) for p in sorted(glob.glob(pattern))]


def read_all_slicevtks(basedir, problem_id, verbose=True):
    """Read every slicevtk id directory under ``basedir/slice``.

    Returns a dict ``{slice_id: list[frame_dict]}``.
    """
    slice_root = os.path.join(basedir, "slice")
    slice_ids = sorted(os.listdir(slice_root))
    out = {}
    for slice_id in slice_ids:
        if verbose:
            print(f"Searching for id={slice_id}....................")
        pattern = os.path.join(
            slice_root, slice_id, f"{problem_id}.????.{slice_id}.slice.vtk"
        )
        frames = read_slicevtk_series(pattern)
        if verbose:
            print(f"  found and read {len(frames)} slicevtk files")
        out[slice_id] = frames
    return out


def print_metadata(slc):
    """Print a compact summary of a slicevtk frame."""
    print(f"id={slc['id']!r}  time={slc['time']:g}  out={slc['out']}")
    for pname in slc["plane_names"]:
        p = slc["planes"][pname]
        print(
            f"  {pname}: axes=({p['xaxis']},{p['yaxis']}) "
            f"shape=({p['Ny']},{p['Nx']}) "
            f"requested={p['requested_coord']:g} selected={p['selected_coord']:g}"
        )
        print("    fields: " + ", ".join(p["field_names"]))


def plot_plane(slc, plane="x3", fields=None, ncol=3, cmap=None):
    """Plot selected fields for one plane using ``pcolormesh``.

    Vector fields are plotted component-by-component with suffixes ``_0``,
    ``_1``, and ``_2``.
    """
    p = slc["planes"][plane]
    if fields is None:
        fields = p["field_names"]

    panels = []
    for name in fields:
        arr = p["fields"][name]
        if arr.ndim == 3:
            for c in range(arr.shape[2]):
                panels.append((f"{name}_{c+1}", arr[..., c]))
        else:
            panels.append((name, arr))

    ncol = min(ncol, len(panels))
    nrow = len(panels)//ncol + (len(panels) % ncol > 0)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol*4, nrow*3), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, arr) in zip(axes, panels):
        im = ax.pcolormesh(p["x_edges"], p["y_edges"], arr, shading="auto", cmap=cmap)
        ax.set_title(name)
        ax.set_xlabel(p["xaxis"])
        ax.set_ylabel(p["yaxis"])
        fig.colorbar(im, ax=ax)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    return fig, axes
