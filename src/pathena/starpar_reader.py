"""Reader and plotting helpers for Athena star-particle VTK files.

The binary reader here handles the star-particle fields needed by the plotting
helpers and uses the same file-discovery conventions as
:mod:`pathena.proj2d_reader` (``<basedir>/starpar/<id>.<num>.starpar.vtk``)
and add helpers for overplotting particles on projected 2D images.

Typical use:

    from pathena.starpar_reader import (
        read_starpar, read_all_particles,
        plot_particles_on_projection,
    )

    parts = read_all_particles(basedir, "R8_8pc_NCR")
    # parts is a dict keyed by 4-digit num string; each value carries
    # {'path', 'num', 'time', 'particles'} where 'particles' is a DataFrame.
"""

import glob
import os
import re
import struct

import numpy as np
import pandas as pd

from .projection import basis_vectors
from .units import DEFAULT_MUH, star_particle_units


_NUM_RE = re.compile(r"\.(\d{4,})\.")


def _shear_wrap(positions, bounds, shear_offset=0.0):
    """Wrap x1 by Lx and shift x2 by m*shear_offset so all positions live in
    the base cell. ``positions`` is an (N, 3) array; ``bounds`` is the 2- or
    3-tuple accepted by :func:`pathena.proj2d_reader.plot_domain_footprint`.
    """
    (x1lo, x1hi), (x2lo, x2hi) = bounds[0], bounds[1]
    Lx = x1hi - x1lo
    Ly = x2hi - x2lo
    p = np.array(positions, dtype=float, copy=True)
    m = np.floor((p[:, 0] - x1lo)/Lx)
    p[:, 0] -= m*Lx
    p[:, 1] = x2lo + np.mod(p[:, 1] + m*shear_offset - x2lo, Ly)
    return p


def _parse_starvtk_line(line, star):
    parts = line.strip().split()
    if not parts:
        return
    if b"vtk" in parts:
        star["vtk_version"] = parts[-1]
    elif b"time=" in parts:
        time_index = parts.index(b"time=")
        star["time"] = float(parts[time_index + 1].rstrip(b","))
    elif b"CELL_DATA" in parts:
        star["ncells"] = int(parts[-1])
    elif b"NSTARS" in parts:
        star["nstar"] = int(parts[1])
    elif b"POINTS" in parts:
        star["nstar"] = int(parts[1])
        star["ncells"] = int(parts[1])
    elif b"SCALARS" in parts:
        star["read_field"] = parts[1].decode("utf-8")
        star["read_type"] = "scalar"
    elif b"VECTORS" in parts:
        star["read_field"] = parts[1].decode("utf-8")
        star["read_type"] = "vector"


def _starvtk_field_map(path, data_offset, ncells):
    field_map = {}
    with open(path, "rb") as fp:
        fp.seek(0, os.SEEK_END)
        eof = fp.tell()
        offset = data_offset
        fp.seek(offset)

        while offset < eof:
            line = fp.readline()
            if not line:
                break
            parts = line.strip().split()
            if not parts:
                offset = fp.tell()
                continue
            field = parts[1].decode("utf-8")
            entry = {"read_table": False, "offset": offset}

            if b"SCALARS" in line:
                fp.readline()
                entry["read_table"] = True
                entry["nvar"] = 1
            elif b"VECTORS" in line:
                entry["nvar"] = 3
            else:
                raise TypeError(f"{path}: unknown VTK field type {parts[0]!r}")

            entry["ndata"] = entry["nvar"] * ncells
            if parts[2] == b"int":
                dtype = "i"
            elif parts[2] == b"float":
                dtype = "f"
            elif parts[2] == b"double":
                dtype = "d"
            else:
                raise TypeError(f"{path}: unsupported VTK field dtype {parts[2]!r}")
            entry["dtype"] = dtype
            entry["dsize"] = entry["ndata"] * struct.calcsize(dtype)
            fp.seek(entry["dsize"], os.SEEK_CUR)
            offset = fp.tell()
            maybe_blank = fp.readline()
            if len(maybe_blank) > 1:
                fp.seek(offset)
            else:
                offset = fp.tell()
            field_map[field] = entry
    return field_map


def _read_starvtk_field(fp, field_map):
    fp.seek(field_map["offset"])
    fp.readline()
    if field_map["read_table"]:
        fp.readline()
    data = fp.read(field_map["dsize"])
    return np.asarray(struct.unpack(">" + str(field_map["ndata"]) + field_map["dtype"], data))


def read_starvtk(path, time_out=False):
    """Read an Athena star-particle VTK file into a pandas DataFrame."""
    star = {"filename": path, "read_field": None, "read_type": None}
    with open(path, "rb") as fp:
        while star["read_field"] is None:
            star["data_offset"] = fp.tell()
            line = fp.readline()
            if not line:
                raise ValueError(f"{path}: reached EOF before star-particle fields")
            _parse_starvtk_line(line, star)

        nstar = int(star["nstar"])
        fields = _starvtk_field_map(path, star["data_offset"], int(star["ncells"]))
        ids = _read_starvtk_field(fp, fields["star_particle_id"])
        mass = _read_starvtk_field(fp, fields["star_particle_mass"])
        age = _read_starvtk_field(fp, fields["star_particle_age"])
        pos = _read_starvtk_field(fp, fields["star_particle_position"]).reshape(nstar, 3)
        vel = _read_starvtk_field(fp, fields["star_particle_velocity"]).reshape(nstar, 3)

    df = pd.DataFrame({
        "id": ids,
        "mass": mass,
        "age": age,
        "v1": vel[:, 0],
        "v2": vel[:, 1],
        "v3": vel[:, 2],
        "x1": pos[:, 0],
        "x2": pos[:, 1],
        "x3": pos[:, 2],
        "time": float(star["time"]),
    })
    if time_out:
        return float(star["time"]), df
    return df


def read_starpar(path):
    """Read one starpar VTK file and wrap it with proj2d-style metadata.

    Parameters
    ----------
    path : str
        Path to the ``.starpar.vtk`` file.

    Returns
    -------
    dict with keys:

        path : str
        num  : 4-digit snapshot num parsed from the filename (or None)
        time : simulation time
        particles : pandas.DataFrame with the star-particle table
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    t, df = read_starvtk(path, time_out=True)
    m = _NUM_RE.search(os.path.basename(path))
    return {
        "path": path,
        "num": m.group(1) if m else None,
        "time": float(t),
        "particles": df,
    }


def read_starpar_series(pattern):
    """Read every starpar file matching a glob pattern, sorted by filename."""
    return [read_starpar(p) for p in sorted(glob.glob(pattern))]


def read_all_particles(basedir, problem_id, ext="vtk", verbose=True):
    """Read every starpar file for a run.

    Mirrors :func:`pathena.proj2d_reader.read_all_proj2ds`. Searches, in
    priority order:

    1. ``<basedir>/starpar/<problem_id>.????.starpar.<ext>`` (NEW_OUTPUT_DIRECTORY)
    2. ``<basedir>/id0/<problem_id>.????.starpar.<ext>`` (MPI layout)
    3. ``<basedir>/<problem_id>.????.starpar.<ext>`` (flat serial layout)

    The first layout that yields any files wins.

    Returns
    -------
    list[dict]
        One frame dict per snapshot, sorted by filename (i.e. by ``num``).
        Each element is what :func:`read_starpar` returns.
    """
    patterns = [
        os.path.join(basedir, "starpar",
                     f"{problem_id}.????.starpar.{ext}"),
        os.path.join(basedir, "id0",
                     f"{problem_id}.????.starpar.{ext}"),
        os.path.join(basedir,
                     f"{problem_id}.????.starpar.{ext}"),
    ]
    paths = []
    for pat in patterns:
        found = sorted(glob.glob(pat))
        if found:
            paths = found
            if verbose:
                print(f"Reading starpar files matching {pat}")
            break
    if not paths and verbose:
        print(f"No starpar files found under {basedir}")

    frames = [read_starpar(p) for p in paths]
    if verbose:
        print(f"  read {len(frames)} starpar files")
    return frames


def load_starpar_for_frame(basedir, problem_id, proj_frame, ext="vtk"):
    """Return the starpar DataFrame matching a proj2d frame's ``num``.

    Searches ``<basedir>/starpar/``, then ``<basedir>/id0/``, then
    ``<basedir>/``. Returns ``None`` if no file is found or the num is
    unknown.
    """
    num = proj_frame.get("num")
    if not num:
        return None
    candidates = [
        os.path.join(basedir, "starpar",
                     f"{problem_id}.{num}.starpar.{ext}"),
        os.path.join(basedir, "id0",
                     f"{problem_id}.{num}.starpar.{ext}"),
        os.path.join(basedir,
                     f"{problem_id}.{num}.starpar.{ext}"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None
    return read_starvtk(path)


def _resolve_particles(particles):
    """Return a DataFrame from a DataFrame / starpar frame dict / dict of arrays.

    Returns ``None`` for empty inputs. ``(N, 3)`` NumPy arrays are wrapped
    into a minimal DataFrame with x1/x2/x3 only (no mass/age/id — those
    particles will be treated as sources/runaways-of-unknown-kind).
    """
    if particles is None:
        return None
    if isinstance(particles, dict) and "particles" in particles:
        return _resolve_particles(particles["particles"])
    if hasattr(particles, "columns"):
        return particles
    if isinstance(particles, dict):
        return pd.DataFrame(particles)
    arr = np.asarray(particles, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("particles array must have shape (N, 3)")
    return pd.DataFrame({"x1": arr[:, 0], "x2": arr[:, 1], "x3": arr[:, 2]})


def plot_particles_on_projection(ax, particles, theta, phi, degrees=True,
                                 bounds=None, wrap=0, shear_offset=0.0,
                                 x_col="x1", y_col="x2", z_col="x3",
                                 mass_col="mass", age_col="age", id_col="id",
                                 norm_factor=4.0, agemax=40.0,
                                 runaway=True, source=True,
                                 cmap="cool_r", muH=1.4271,
                                 cluster_kwargs=None,
                                 runaway_kwargs=None,
                                 source_kwargs=None,
                                 **kwargs):
    """Overplot star particles on a proj2d image plane, ``scatter_sp``-style.

    Projects each particle's 3D position onto the image plane with the same
    basis as ``output_proj2d.c`` and scatters onto ``ax``. Particles are split
    into three categories:

    * **Clusters** (``mass > 0``): area = ``sqrt(mass [Msun]) / norm_factor``,
      color = ``age`` in Myr via ``cmap`` (default ``"cool_r"``), clipped to
      ``[0, agemax]``. Only particles younger than ``agemax`` are drawn.
    * **Runaway non-sources** (``mass == 0`` and ``id >= 0``): black ``o``
      markers of fixed size ``10/norm_factor``. Drawn only if ``runaway``.
    * **Runaway sources** (``mass == 0`` and ``id < 0``): red ``*`` markers of
      fixed size ``10/norm_factor``. Drawn only if ``source``.

    Category-specific ``ax.scatter`` overrides go through
    ``cluster_kwargs``/``runaway_kwargs``/``source_kwargs``. Any remaining
    ``**kwargs`` (e.g. ``zorder``, ``alpha``) are applied to all three
    categories.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes drawn in image-plane coordinates.
    particles : DataFrame, dict, starpar frame, or (N, 3) array
        Star-particle table (columns ``x1,x2,x3,mass,age,id``) or bare
        positions. A starpar frame dict from :func:`read_starpar` is accepted.
    theta, phi : float
        LOS angles matching the projection.
    degrees : bool
        If True (default), theta and phi are in degrees.
    bounds : sequence, optional
        Domain bounds for shear-periodic wrapping. Same format as
        :func:`pathena.proj2d_reader.plot_domain_footprint`.
    wrap : int
        If >0 and ``bounds`` is given, also plot particle images at
        ``m Lx`` shifts (m = -wrap..+wrap excluding 0).
    shear_offset : float
        Shear offset ``deltay`` for wrapping.
    x_col, y_col, z_col, mass_col, age_col, id_col : str
        Column names.
    norm_factor : float
        Scales cluster and runaway marker areas (bigger => smaller markers).
    agemax : float
        Age (Myr) cutoff for cluster display and color-normalization ``vmax``.
    runaway, source : bool
        Toggle the two runaway categories.
    cmap : str or Colormap
        Colormap used for cluster age.
    muH : float
        Hydrogen mass per H nucleus in proton masses for code-unit -> Msun/Myr
        conversion.
    cluster_kwargs, runaway_kwargs, source_kwargs : dict, optional
        Per-category kwargs passed to ``ax.scatter``.
    **kwargs :
        Shared ``ax.scatter`` overrides (e.g. ``alpha``, ``zorder``).

    Returns
    -------
    dict
        ``{"cluster": [...], "runaway": [...], "source": [...]}``: lists of
        scatter artists (one per base + wrap copy, per category).
    """
    df = _resolve_particles(particles)
    if df is None or len(df) == 0:
        return {"cluster": [], "runaway": [], "source": []}

    pos = np.column_stack([
        np.asarray(df[x_col], dtype=float),
        np.asarray(df[y_col], dtype=float),
        np.asarray(df[z_col], dtype=float),
    ])

    if bounds is not None:
        pos_base = _shear_wrap(pos, bounds, shear_offset=shear_offset)
    else:
        pos_base = pos

    _, ex, ey = basis_vectors(theta, phi, degrees=degrees)

    if bounds is not None:
        (x1lo, x1hi), (x2lo, x2hi) = bounds[0], bounds[1]
        cx = 0.5*(x1lo + x1hi)
        cy = 0.5*(x2lo + x2hi)
        cz = (0.5*(bounds[2][0] + bounds[2][1])
              if len(bounds) == 3 else 0.0)
    else:
        cx = cy = cz = 0.0
    center = np.array([cx, cy, cz])

    def _project(pos_arr, shift_x1, shift_x2):
        p = pos_arr.copy()
        p[:, 0] += shift_x1
        p[:, 1] += shift_x2
        rel = p - center
        return rel @ ex, rel @ ey

    unit = star_particle_units(muH=muH)
    Msun = unit["mass_msun"]
    Myr = unit["time_myr"]

    has_mass = mass_col in df.columns
    has_id = id_col in df.columns
    if has_mass:
        mass = np.asarray(df[mass_col], dtype=float)
        is_cluster = mass > 0.0
    else:
        mass = np.zeros(len(df))
        is_cluster = np.zeros(len(df), dtype=bool)
    if has_id:
        pid = np.asarray(df[id_col], dtype=float)
        is_source = (~is_cluster) & (pid < 0)
    else:
        pid = np.zeros(len(df))
        is_source = np.zeros(len(df), dtype=bool)
    is_runaway_ns = (~is_cluster) & (~is_source)

    # Cluster selection: age < agemax and mass > 0
    if age_col in df.columns:
        age_myr = np.asarray(df[age_col], dtype=float)*Myr
    else:
        age_myr = np.zeros(len(df))
    cluster_sel = is_cluster & (age_myr < agemax)

    artists = {"cluster": [], "runaway": [], "source": []}

    def _draw_cluster(shift_x1, shift_x2, alpha_scale):
        if not cluster_sel.any():
            return
        u, v = _project(pos_base[cluster_sel], shift_x1, shift_x2)
        sizes = np.sqrt(mass[cluster_sel]*Msun)/norm_factor
        colors = age_myr[cluster_sel]
        base = dict(marker="o", cmap=cmap, vmin=0.0, vmax=agemax, alpha=0.7,
                    zorder=5)
        base.update(kwargs)
        if cluster_kwargs:
            base.update(cluster_kwargs)
        base["alpha"] = base.get("alpha", 0.7)*alpha_scale
        art = ax.scatter(u, v, s=sizes, c=colors, **base)
        artists["cluster"].append(art)

    def _draw_runaway_ns(shift_x1, shift_x2, alpha_scale):
        if not runaway or not is_runaway_ns.any():
            return
        u, v = _project(pos_base[is_runaway_ns], shift_x1, shift_x2)
        base = dict(marker="o", color="k", alpha=1.0, zorder=5)
        base.update(kwargs)
        if runaway_kwargs:
            base.update(runaway_kwargs)
        base["alpha"] = base.get("alpha", 1.0)*alpha_scale
        art = ax.scatter(u, v, s=10.0/norm_factor, **base)
        artists["runaway"].append(art)

    def _draw_source(shift_x1, shift_x2, alpha_scale):
        if not source or not is_source.any():
            return
        u, v = _project(pos_base[is_source], shift_x1, shift_x2)
        base = dict(marker="*", color="r", alpha=1.0, zorder=5)
        base.update(kwargs)
        if source_kwargs:
            base.update(source_kwargs)
        base["alpha"] = base.get("alpha", 1.0)*alpha_scale
        art = ax.scatter(u, v, s=10.0/norm_factor, **base)
        artists["source"].append(art)

    def _draw_all(shift_x1, shift_x2, alpha_scale):
        _draw_cluster(shift_x1, shift_x2, alpha_scale)
        _draw_runaway_ns(shift_x1, shift_x2, alpha_scale)
        _draw_source(shift_x1, shift_x2, alpha_scale)

    _draw_all(0.0, 0.0, 1.0)
    if wrap > 0:
        if bounds is None:
            raise ValueError("wrap>0 requires bounds")
        (x1lo, x1hi) = bounds[0]
        Lx = x1hi - x1lo
        for m in range(-wrap, wrap + 1):
            if m == 0:
                continue
            _draw_all(m*Lx, m*shear_offset, 0.5)

    return artists
