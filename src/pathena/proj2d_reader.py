"""Reader for proj2d output files written by Athena-TIGRESS-NCR.

The proj2d format is a binary VTK legacy STRUCTURED_POINTS file containing
a 2D line-of-sight projection with one SCALARS record per projected field.
The second header line carries proj2d-specific metadata, for example:

    PROJ2D theta30-phi25 time=1.000000e+00 theta=3.000000e+01 phi=2.500000e+01 dl_factor=1.000000e+00

See src/output_proj2d.c for the writer.

Typical use:

    from pathena.proj2d_reader import read_proj2d

    prj = read_proj2d("proj2d/theta30-phi25/R8_8pc_NCR.0001.theta30-phi25.proj2d")
    prj["time"]       # simulation time
    prj["theta"]      # LOS angle from +x3, degrees
    prj["phi"]        # LOS azimuth in x1-x2 plane, degrees
    prj["x_edges"]    # image x edges
    prj["y_edges"]    # image y edges
    prj["fields"]     # dict {field_name: ndarray of shape (Nbiny, Nbinx)}
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from .projection import basis_vectors
from .units import PC_CGS, tigress_units
from .vtk_helper import (
    axis_arrays,
    read_ascii_line,
    read_structured_points_header,
    read_vtk_magic,
    read_vtk_scalar_2d,
)
from .starpar_reader import (
    plot_particles_on_projection,
    load_starpar_for_frame as _load_starpar_for_frame,
)

_HDR_RE = re.compile(
    r"^PROJ2D\s+(?P<id>\S+)\s+"
    r"time=(?P<time>[-+0-9.eE]+)\s+"
    r"theta=(?P<theta>[-+0-9.eE]+)\s+"
    r"phi=(?P<phi>[-+0-9.eE]+)\s+"
    r"dl_factor=(?P<dl_factor>[-+0-9.eE]+)"
)



def read_proj2d(path):
    """Read one proj2d VTK legacy file.

    Parameters
    ----------
    path : str
        Path to the .proj2d file.

    Returns
    -------
    dict with keys:

        time, id, theta, phi, dl_factor
        Nbinx, Nbiny
        x_origin, y_origin, x_spacing, y_spacing
        x_edges, x_centers, y_edges, y_centers
        field_names : list of str preserving file order
        fields      : dict[name -> ndarray of shape (Nbiny, Nbinx)]
        header_lines: raw ASCII header lines, for debugging
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as fp:
        header_lines = []

        ln = read_vtk_magic(fp, path)
        header_lines.append(ln.rstrip())

        meta_ln = read_ascii_line(fp).rstrip()
        header_lines.append(meta_ln)
        m = _HDR_RE.match(meta_ln)
        if not m:
            raise ValueError(
                f"{path}: cannot parse PROJ2D metadata line: {meta_ln!r}"
            )
        num_match = re.search(r"\.(\d{4,})\.", os.path.basename(path))
        out = {
            "path": path,
            "num": num_match.group(1) if num_match else None,
            "id": m.group("id"),
            "time": float(m.group("time")),
            "theta": float(m.group("theta")),
            "phi": float(m.group("phi")),
            "dl_factor": float(m.group("dl_factor")),
        }

        extra_header, dims, origin, spacing, _ = read_structured_points_header(fp, path)
        header_lines.extend(extra_header)
        Nx = int(dims[0] - 1)
        Ny = int(dims[1] - 1)
        x_orig = float(origin[0])
        y_orig = float(origin[1])
        dx = float(spacing[0])
        dy = float(spacing[1])

        xe, xc = axis_arrays(x_orig, dx, Nx)
        ye, yc = axis_arrays(y_orig, dy, Ny)
        out.update({
            "Nbinx": Nx,
            "Nbiny": Ny,
            "x_origin": x_orig,
            "y_origin": y_orig,
            "x_spacing": dx,
            "y_spacing": dy,
            "x_edges": xe,
            "x_centers": xc,
            "y_edges": ye,
            "y_centers": yc,
        })

        field_names = []
        fields = {}
        while True:
            ln = read_ascii_line(fp)
            if not ln:
                break
            stripped = ln.strip()
            if not stripped:
                continue
            if not stripped.startswith("SCALARS"):
                continue
            tok = stripped.split()
            name = tok[1]
            lookup_ln = read_ascii_line(fp).strip()
            if not lookup_ln.startswith("LOOKUP_TABLE"):
                raise ValueError(
                    f"{path}: expected LOOKUP_TABLE after SCALARS {name}, "
                    f"got {lookup_ln!r}"
                )
            arr = read_vtk_scalar_2d(fp, path, name, Nx, Ny)
            field_names.append(name)
            fields[name] = arr

        out["field_names"] = field_names
        out["fields"] = fields
        out["header_lines"] = header_lines

    return out


def read_proj2d_series(pattern):
    """Read every file matching a glob pattern, sorted by filename."""
    paths = sorted(glob.glob(pattern))
    return [read_proj2d(p) for p in paths]


def read_all_proj2ds(basedir, problem_id, verbose=True, index_by="list"):
    """Read every proj2d id directory under ``basedir/proj2d``.

    Returns a dict ``{proj_id: list[frame_dict]}`` by default. Set
    ``index_by="num"`` to return ``{proj_id: {output_number: frame_dict}}``.
    """
    if index_by not in ("list", "num"):
        raise ValueError("index_by must be 'list' or 'num'")
    proj_root = os.path.join(basedir, "proj2d")
    proj_ids = sorted(os.listdir(proj_root))
    proj_dict = {}
    for proj_id in proj_ids:
        if verbose:
            print(f"Searching for id={proj_id}....................")
        pattern = os.path.join(
            proj_root, proj_id, f"{problem_id}.????.{proj_id}.proj2d"
        )
        frames = read_proj2d_series(pattern)
        if verbose:
            print(f"  found and read {len(frames)} proj2d files")
        if index_by == "num":
            proj_dict[proj_id] = {int(frame["num"]): frame for frame in frames}
        else:
            proj_dict[proj_id] = frames
    return proj_dict


def check_proj2d_jumps(proj_dict, field="nH", ratio=1.5):
    """Return abrupt mean-field jumps in a proj2d series.

    ``proj_dict`` may be the default list output from :func:`read_all_proj2ds`
    or the ``index_by="num"`` form. Each returned tuple is
    ``(proj_id, previous_num, current_num, previous_mean, current_mean)``.
    """
    jumps = []
    for proj_id, frames in proj_dict.items():
        if isinstance(frames, dict):
            series = [frames[num] for num in sorted(frames)]
        else:
            series = frames
        previous = None
        for frame in series:
            if field not in frame["fields"]:
                continue
            current = (int(frame["num"]), float(np.mean(frame["fields"][field])))
            if previous is not None:
                prev_num, prev_mean = previous
                curr_num, curr_mean = current
                if prev_mean != 0.0:
                    jump_ratio = curr_mean/prev_mean
                    if jump_ratio > ratio or jump_ratio < 1.0/ratio:
                        jumps.append((proj_id, prev_num, curr_num, prev_mean, curr_mean))
            previous = current
    return jumps


def print_metadata(d):
    print(
        f"id={d['id']!r}  time={d['time']:g}  "
        f"theta={d['theta']:g}  phi={d['phi']:g}  dl_factor={d['dl_factor']:g}"
    )
    print(f"shape: Nbinx={d['Nbinx']}  Nbiny={d['Nbiny']}")
    print(f"x range: {d['x_edges'][0]:g} -> {d['x_edges'][-1]:g}")
    print(f"y range: {d['y_edges'][0]:g} -> {d['y_edges'][-1]:g}")
    print(f"fields ({len(d['field_names'])}):")
    for name in d["field_names"]:
        a = d["fields"][name]
        nz = int(np.count_nonzero(a))
        nan = int(np.isnan(a).sum())
        print(f"  {name:22s}  sum={a.sum():.4e}  nonzero={nz}/{a.size}  nans={nan}")


def plot_domain_footprint(ax, theta, phi, bounds, degrees=True,
                          wrap=0, shear_offset=0.0, **kwargs):
    """Overlay the projected base-domain wireframe on a proj2d image.

    Uses the same basis convention as ``output_proj2d.c`` and
    :func:`pathena.projection.project_shearing_periodic`:

        e_imgx = (-sin phi, cos phi, 0)
        e_imgy = (-cos theta cos phi, -cos theta sin phi, sin theta)

    The 12 edges of the axis-aligned box ``bounds`` are drawn on ``ax`` in
    image-plane coordinates. Set ``wrap > 0`` to also overlay shear-periodic
    copies of the box (``m Lx`` shifts in x1 combined with ``m*shear_offset``
    shifts in x2 for ``m = -wrap..+wrap``, excluding zero), which is useful
    when the image footprint extends beyond the base cell.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes drawn in image-plane coordinates. Typically the result of
        ``ax.pcolormesh(proj["x_edges"], proj["y_edges"], ...)``.
    theta, phi : float
        LOS angles matching the projection.
    bounds : sequence
        Either ``((x1lo, x1hi), (x2lo, x2hi))`` for the x1-x2 base rectangle
        (drawn at z = 0, i.e., 4 corners and 4 edges — sufficient when you
        only want to see the horizontal footprint and its shear-periodic
        copies), or ``((x1lo, x1hi), (x2lo, x2hi), (x3lo, x3hi))`` for the
        full 3D wireframe (8 corners, 12 edges). The x3 range only affects
        the drawing through ``ey[2] = sin(theta)``, so it is meaningful only
        for oblique views.
    degrees : bool
        If True (default), theta and phi are in degrees.
    wrap : int
        Number of shear-periodic copies to draw on each side of the base cell.
        ``0`` (default) draws only the base cell.
    shear_offset : float
        The x2 shift ``deltay`` per unit ``m`` = ``fmod(qshear*Omega_0*Lx*t, Ly)``.
        Only used when ``wrap > 0``.
    **kwargs :
        Passed to ``ax.plot`` for the base-cell edges. Defaults:
        ``color='k'``, ``lw=0.8``, ``ls='-'``. Wrapped copies are drawn with
        the same style but ``alpha`` scaled by 0.5.

    Returns
    -------
    list of matplotlib.lines.Line2D
        The line segments added.
    """
    _, ex, ey = basis_vectors(theta, phi, degrees=degrees)

    if len(bounds) == 2:
        (x1lo, x1hi), (x2lo, x2hi) = bounds
        x3lo, x3hi = 0.0, 0.0
        z_signs = (0,)
    elif len(bounds) == 3:
        (x1lo, x1hi), (x2lo, x2hi), (x3lo, x3hi) = bounds
        z_signs = (-1, +1)
    else:
        raise ValueError("bounds must have 2 or 3 (lo, hi) pairs")

    center = np.array([0.5*(x1lo + x1hi), 0.5*(x2lo + x2hi), 0.5*(x3lo + x3hi)])
    Lx = x1hi - x1lo
    Ly = x2hi - x2lo

    corners = np.array([
        (sx*(x1hi - center[0]), sy*(x2hi - center[1]), sz*(x3hi - center[2]))
        for sx in (-1, +1) for sy in (-1, +1) for sz in z_signs
    ])
    # Box edges: pairs of corners differing in exactly one bit (of the used
    # coordinates). 4 edges for 2D base, 12 edges for full 3D box.
    n_corners = len(corners)
    edges = [(i, j) for i in range(n_corners) for j in range(i+1, n_corners)
             if bin(i ^ j).count("1") == 1]

    base_kwargs = dict(color="r", lw=0.8, ls="-")
    base_kwargs.update(kwargs)
    base_alpha = base_kwargs.pop("alpha", 1.0)

    lines = []

    def _draw(shift, alpha):
        pts = corners + shift
        u = pts @ ex
        v = pts @ ey
        for (i, j) in edges:
            line, = ax.plot([u[i], u[j]], [v[i], v[j]], alpha=alpha,
                            **base_kwargs)
            lines.append(line)

    _draw(np.zeros(3), base_alpha)
    for m in range(-wrap, wrap + 1):
        if m == 0:
            continue
        shift = np.array([m*Lx, m*shear_offset, 0.0])
        _draw(shift, base_alpha*0.5)

    return lines


def _field_norm(data, norm):
    """Resolve a norm option for one projected field array."""
    if norm == "auto":
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return Normalize(vmin=0.0, vmax=1.0)
        positive = finite[finite > 0.0]
        if positive.size and positive.size == finite.size:
            return LogNorm(vmin=positive.min(), vmax=positive.max())
        return Normalize(vmin=finite.min(), vmax=finite.max())
    if norm == "linear":
        return Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    if norm == "log":
        positive = data[np.isfinite(data) & (data > 0.0)]
        if positive.size == 0:
            raise ValueError("log normalization requires at least one positive value")
        return LogNorm(vmin=positive.min(), vmax=positive.max())
    if norm == "symlog":
        vmax = np.nanmax(np.abs(data))
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        return SymLogNorm(linthresh=max(vmax*1.0e-3, 1.0e-30),
                          vmin=-vmax, vmax=vmax)
    return norm


def plot_field_all_projections(proj_dict, field, frame=-1, ncol=5, norm="auto",
                               cmap=None, share_norm=False, deprojection=False,
                               missing="raise",
                               footprint=False, bounds=None,
                               footprint_kwargs=None,
                               particles=None, particle_kwargs=None,
                               field_label=None, data_scale=1.0):
    """Plot one field from every projection id in a ``proj_dict``.

    Parameters
    ----------
    proj_dict : dict
        Return value from :func:`read_all_proj2ds`, i.e.
        ``{proj_id: list[frame_dict]}``.
    field : str
        Field name to plot from each proj2d frame.
    frame : int
        Frame list index to use for every projection id. Defaults to the last
        frame. This is an index into each list, not the Athena output number.
    ncol : int
        Number of subplot columns.
    norm : {'auto', 'linear', 'log', 'symlog'} or matplotlib norm
        Color normalization. If ``share_norm=True``, the same resolved norm is
        used for all panels.
    cmap : str or Colormap, optional
        Matplotlib colormap.
    share_norm : bool
        If true, derive one normalization from all selected projection data.
    missing : {'raise', 'skip', 'blank'}
        Behavior when a projection has no selected frame or lacks ``field``.
    footprint : bool
        If True, overlay the projected base-domain footprint on each panel
        via :func:`plot_domain_footprint`.
    bounds : sequence, optional
        Domain bounds in code units — either ``((x1lo, x1hi), (x2lo, x2hi))``
        for a 2D horizontal footprint or ``((x1lo, x1hi), (x2lo, x2hi),
        (x3lo, x3hi))`` for the full 3D wireframe. If omitted while
        ``footprint=True``, each panel's own image edges are reused as the
        x1/x2 bounds (i.e. the image footprint is assumed to match the
        original domain XY extent).
    footprint_kwargs : dict, optional
        Extra kwargs forwarded to :func:`plot_domain_footprint` (e.g.
        ``{"wrap": 2, "shear_offset": 500.0, "color": "tab:cyan"}``).
    particles : DataFrame, list, dict, callable, or "auto", optional
        Star (or generic) particles to overplot via
        :func:`pathena.starpar_reader.plot_particles_on_projection`. Accepted
        forms:

        * A single DataFrame or starpar frame dict — used for every panel.
        * A ``list`` of starpar frame dicts (e.g. from
          :func:`pathena.starpar_reader.read_all_particles`) — indexed by the
          same ``frame`` parameter used to pick a proj2d frame per panel.
        * A ``dict[proj_id -> DataFrame or frame dict]`` — panel-specific.
        * A callable ``fn(proj_id, proj_frame) -> DataFrame or None`` — invoked
          per panel; ``None`` skips overlay for that panel.
        * The string ``"auto"`` — together with ``basedir`` and ``problem_id``
          entries in ``particle_kwargs``, auto-loads the starpar VTK matching
          each proj2d frame's ``num``.
    particle_kwargs : dict, optional
        Extra kwargs for :func:`plot_particles_on_projection`. Special keys
        ``basedir``, ``problem_id`` are consumed for auto-loading.
    field_label : str, optional
        Colorbar label. Defaults to ``field``.
    data_scale : float, optional
        Multiplicative factor applied after optional deprojection.

    Returns
    -------
    fig, axes
        Matplotlib figure and a 1D array of axes.
    """
    if missing not in ("raise", "skip", "blank"):
        raise ValueError("missing must be 'raise', 'skip', or 'blank'")
    if not proj_dict:
        raise ValueError("proj_dict is empty")
    footprint_kwargs = dict(footprint_kwargs) if footprint_kwargs else {}
    particle_kwargs = dict(particle_kwargs) if particle_kwargs else {}
    _particle_basedir = particle_kwargs.pop("basedir", None)
    _particle_problem_id = particle_kwargs.pop("problem_id", None)

    panels = []
    for proj_id in sorted(proj_dict):
        frames = proj_dict[proj_id]
        if not frames:
            if missing == "raise":
                raise ValueError(f"projection {proj_id!r} has no frames")
            if missing == "blank":
                panels.append((proj_id, None, None))
            continue
        try:
            proj = frames[frame]
        except IndexError:
            if missing == "raise":
                raise IndexError(
                    f"projection {proj_id!r} has no frame index {frame}"
                )
            if missing == "blank":
                panels.append((proj_id, None, None))
            continue
        if field not in proj["fields"]:
            if missing == "raise":
                available = ", ".join(proj["field_names"])
                raise KeyError(
                    f"projection {proj_id!r} is missing field {field!r}; "
                    f"available: {available}"
                )
            if missing == "blank":
                panels.append((proj_id, proj, None))
            continue
        data = proj["fields"][field]
        if deprojection:
            data = data * np.cos(np.deg2rad(proj["theta"]))
        if data_scale != 1.0:
            data = data * data_scale
        panels.append((proj_id, proj, data))

    if not panels:
        raise ValueError(f"no projections to plot for field {field!r}")

    data_arrays = [data for _, _, data in panels if data is not None]
    shared_norm = None
    if share_norm and data_arrays:
        shared_norm = _field_norm(np.concatenate([a.ravel() for a in data_arrays]), norm)

    ncol = max(1, min(ncol, len(panels)))
    nrow = len(panels)//ncol + (len(panels) % ncol > 0)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol*4, nrow*3), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (proj_id, proj, data) in zip(axes, panels):
        if proj is None or data is None:
            ax.set_visible(False)
            continue
        data_norm = shared_norm if shared_norm is not None else _field_norm(data, norm)
        im = ax.pcolormesh(proj["x_edges"], proj["y_edges"], data,
                           norm=data_norm, cmap=cmap)
        panel_bounds = bounds
        if panel_bounds is None:
            panel_bounds = (
                (float(proj["x_edges"][0]), float(proj["x_edges"][-1])),
                (float(proj["y_edges"][0]), float(proj["y_edges"][-1])),
            )
        if footprint:
            plot_domain_footprint(ax, proj["theta"], proj["phi"], panel_bounds,
                                  **footprint_kwargs)
        if particles is not None:
            if isinstance(particles, str) and particles == "auto":
                if _particle_basedir is None or _particle_problem_id is None:
                    raise ValueError(
                        "particles='auto' requires particle_kwargs["
                        "'basedir'] and particle_kwargs['problem_id']"
                    )
                sp_df = _load_starpar_for_frame(_particle_basedir,
                                                _particle_problem_id, proj)
            elif callable(particles):
                sp_df = particles(proj_id, proj)
            elif isinstance(particles, dict):
                sp_df = particles.get(proj_id, particles)
            elif isinstance(particles, list):
                try:
                    sp_df = particles[frame]
                except IndexError:
                    sp_df = None
            else:
                sp_df = particles
            if sp_df is not None and len(sp_df) > 0:
                plot_particles_on_projection(
                    ax, sp_df, proj["theta"], proj["phi"],
                    bounds=panel_bounds, **particle_kwargs
                )
                ax.set_xlim(panel_bounds[0])
                ax.set_ylim(panel_bounds[1])
        ax.set_aspect("equal")
        ax.set_xlabel("image x")
        ax.set_ylabel("image y")
        ax.set_title(
            f"theta={proj['theta']:g}  phi={proj['phi']:g}"
        )
        fig.colorbar(im, ax=ax, label=field_label or field)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    return fig, axes


def plot_field_pdf_all_projections(proj_dict, field, frame=-1,
                                   bins=50, range=None, log=True, density=True,
                                   cdf=False,
                                   deprojection=False, data_scale=1.0,
                                   field_label=None, ax=None,
                                   missing="raise", **step_kwargs):
    """Overlay per-id PDFs (or CDFs) of one proj2d field on a single axes.

    Iterates ``sorted(proj_dict)`` and draws one step-histogram per projection
    id (e.g. ``theta0``, ``theta15``, ...), so shape changes with viewing
    angle are directly visible.

    Parameters
    ----------
    proj_dict : dict
        Return value from :func:`read_all_proj2ds`, i.e.
        ``{proj_id: list[frame_dict]}``.
    field : str
        Field name to histogram from each proj2d frame.
    frame : int
        Frame list index into each id's frame list. Defaults to the last
        frame. Matches :func:`plot_field_all_projections`.
    bins : int or array_like
        Number of bins (default 50) or explicit edge array. When an int and
        ``log=True``, edges are ``np.logspace(log10 xmin, log10 xmax, bins+1)``
        computed from the pooled positive data across all ids.
    range : (float, float), optional
        Override the auto x-range for the shared edges. When ``log=True`` both
        values must be positive.
    log : bool
        If True (default), use log-spaced edges and log x-axis. Requires
        positive data.
    density : bool
        Passed to :func:`numpy.histogram`. Default True (area under each curve
        = 1) — direct shape comparison across ids. Ignored when ``cdf=True``.
    cdf : bool
        If True, plot the empirical cumulative distribution function instead
        of the PDF. Each line is normalized to [0, 1] (``density`` is ignored)
        and drawn as a piecewise-linear curve through the bin edges.
    deprojection : bool
        If True, multiply the field data by ``cos(theta)`` before
        histogramming. Same convention as
        :func:`plot_field_all_projections`.
    data_scale : float
        Multiplicative factor applied after ``deprojection``. For column
        density in cm^-2 from an ``nH``-like field, pass
        ``data_scale=pathena.units.PC_CGS`` — the writer accumulates
        ``value * dl_eff`` in code units (cm^-3 * pc), and ``PC_CGS``
        converts the length factor to cm.
    field_label : str, optional
        X-axis label. Defaults to ``field``.
    ax : matplotlib.axes.Axes, optional
        Draw on the given axes. Otherwise a new figure and axes are created.
    missing : {'raise', 'skip'}
        Behavior when an id has no frame at ``frame`` or lacks ``field``.
    **step_kwargs
        Forwarded to ``ax.step`` (PDF) or ``ax.plot`` (CDF) — e.g. ``lw``,
        ``alpha``.

    Returns
    -------
    fig, ax
        Matplotlib figure and the axes drawn on.
    """
    if missing not in ("raise", "skip"):
        raise ValueError("missing must be 'raise' or 'skip'")
    if not proj_dict:
        raise ValueError("proj_dict is empty")

    entries = []
    for proj_id in sorted(proj_dict):
        frames = proj_dict[proj_id]
        if not frames:
            if missing == "raise":
                raise ValueError(f"projection {proj_id!r} has no frames")
            continue
        try:
            proj = frames[frame]
        except IndexError:
            if missing == "raise":
                raise IndexError(
                    f"projection {proj_id!r} has no frame index {frame}"
                )
            continue
        if field not in proj["fields"]:
            if missing == "raise":
                available = ", ".join(proj["field_names"])
                raise KeyError(
                    f"projection {proj_id!r} is missing field {field!r}; "
                    f"available: {available}"
                )
            continue
        data = proj["fields"][field]
        if deprojection:
            data = data * np.cos(np.deg2rad(proj["theta"]))
        if data_scale != 1.0:
            data = data * data_scale
        entries.append((proj_id, proj, np.asarray(data).ravel()))

    if not entries:
        raise ValueError(f"no projections to plot for field {field!r}")

    if hasattr(bins, "__len__"):
        edges = np.asarray(bins, dtype=float)
    else:
        pooled = np.concatenate([d for _, _, d in entries])
        pooled = pooled[np.isfinite(pooled)]
        if log:
            pooled = pooled[pooled > 0.0]
            if pooled.size == 0:
                raise ValueError(
                    f"log=True requires positive values in field {field!r}"
                )
        if range is not None:
            vmin, vmax = float(range[0]), float(range[1])
        elif pooled.size == 0:
            raise ValueError(f"field {field!r} has no finite values")
        else:
            vmin, vmax = float(pooled.min()), float(pooled.max())
        if vmax <= vmin:
            raise ValueError(
                f"invalid range for field {field!r}: vmin={vmin} vmax={vmax}"
            )
        if log:
            if vmin <= 0.0:
                raise ValueError(
                    f"log=True requires positive range; got vmin={vmin}"
                )
            edges = np.logspace(np.log10(vmin), np.log10(vmax), int(bins) + 1)
        else:
            edges = np.linspace(vmin, vmax, int(bins) + 1)

    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.figure

    for proj_id, proj, flat in entries:
        line_label = (
            f"theta={proj['theta']:g}, phi={proj['phi']:g}"
        )
        if cdf:
            counts, _ = np.histogram(flat, bins=edges, density=False)
            total = counts.sum()
            if total == 0:
                continue
            cdf_at_edges = np.concatenate(
                [[0.0], np.cumsum(counts).astype(float) / total]
            )
            ax.plot(edges, cdf_at_edges, label=line_label, **step_kwargs)
        else:
            counts, _ = np.histogram(flat, bins=edges, density=density)
            ax.step(edges[:-1], counts, where="post", label=line_label,
                    **step_kwargs)

    if log:
        ax.set_xscale("log")
    ax.set_xlabel(field_label or field)
    if cdf:
        ax.set_ylabel("CDF")
    else:
        ax.set_ylabel("PDF" if density else "count")
    ax.legend()

    return fig, ax


def plot_all(proj, ncol=4, norm="auto", cmap=None):
    """Plot all fields in a proj2d frame with pcolormesh.

    Parameters
    ----------
    proj : dict
        Return value from :func:`read_proj2d`.
    ncol : int
        Number of subplot columns.
    norm : {'auto', 'linear', 'log', 'symlog'} or matplotlib norm
        Color normalization. ``'auto'`` uses log for positive fields and
        linear otherwise.
    cmap : str or Colormap, optional
        Matplotlib colormap.
    """
    names = proj["field_names"]
    if not names:
        raise ValueError("proj2d frame has no fields")
    ncol = min(ncol, len(names))
    nrow = len(names)//ncol + (len(names) % ncol > 0)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol*4, nrow*3), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        data = proj["fields"][name]
        data_norm = norm
        if norm == "auto":
            finite = data[np.isfinite(data)]
            positive = finite[finite > 0.0]
            if positive.size and positive.size == finite.size:
                data_norm = LogNorm(vmin=positive.min(), vmax=positive.max())
            else:
                data_norm = Normalize(vmin=finite.min(), vmax=finite.max())
        elif norm == "linear":
            data_norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        elif norm == "log":
            positive = data[np.isfinite(data) & (data > 0.0)]
            data_norm = LogNorm(vmin=positive.min(), vmax=positive.max())
        elif norm == "symlog":
            vmax = np.nanmax(np.abs(data))
            data_norm = SymLogNorm(linthresh=max(vmax*1.0e-3, 1.0e-30),
                                   vmin=-vmax, vmax=vmax)

        im = ax.pcolormesh(proj["x_edges"], proj["y_edges"], data,
                           norm=data_norm, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("image x")
        ax.set_ylabel("image y")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, label=name)

    for ax in axes[len(names):]:
        ax.set_visible(False)

    return fig


def plot_physical_maps(proj):
    """Plot the standard physically derived diagnostics from one proj2d frame.

    Each panel is shown only when its required fields are present. Line
    integrals cancel in weighted ratios.
    """
    fields = proj["fields"]
    b_to_microgauss = tigress_units()["magnetic_field_microgauss"]

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.full_like(
            numerator, np.nan, dtype=float), where=denominator > 0)

    panels = []
    if all(name in fields for name in ("nH", "nHI", "nH2")):
        nHII = np.clip(fields["nH"] - fields["nHI"] - 2.0*fields["nH2"], 0, None)
        species = np.stack((2.0*fields["nH2"], fields["nHI"], nHII), axis=-1)
        scale = np.nanpercentile(species, 99)
        rgb = np.arcsinh(10.0*species/(scale if scale > 0 else 1.0)) / np.arcsinh(10.0)
        panels.append(("H phases: 2H2(R)/HI(G)/HII(B)", np.clip(rgb, 0, 1), None, None))
    if "nH" in fields:
        panels.append((r"$N_{\rm H}\;[\mathrm{cm}^{-2}]$",
                       fields["nH"]*PC_CGS, "magma", "log"))
    if "nH*Vlos" in fields and "nH" in fields:
        panels.append((r"$\langle v_{\rm los}\rangle_{n_H}$",
                       ratio(fields["nH*Vlos"], fields["nH"]), "RdBu_r", "symlog"))
    if "ne*Blos" in fields and "ne" in fields:
        panels.append((r"$\langle B_{\rm los}\rangle_{n_e}\;[\mu\mathrm{G}]$",
                       ratio(fields["ne*Blos"], fields["ne"])*b_to_microgauss,
                       "PuOr", "symlog"))
    if "nHI*Vlos" in fields and "nHI" in fields:
        mean = ratio(fields["nHI*Vlos"], fields["nHI"])
        panels.append((r"$\langle v_{\rm los}\rangle_{\rm HI}$", mean, "RdBu_r", "symlog"))
    if all(name in fields for name in ("nHI", "nHI*Vlos", "nHI*Vlos2")):
        variance = ratio(fields["nHI*Vlos2"], fields["nHI"]) - mean**2
        panels.append((r"$\sigma_{v,{\rm HI}}$", np.sqrt(np.clip(variance, 0, None)), "viridis", "linear"))
    for name, title in (("nHI_CNM", r"$N_{\rm HI}(T<500\,K)\;[\mathrm{cm}^{-2}]$"),
                        ("nHI_WNM", r"$N_{\rm HI}(T>6000\,K)\;[\mathrm{cm}^{-2}]$")):
        if name in fields:
            panels.append((title, fields[name]*PC_CGS, "magma", "log"))
    if not panels:
        raise ValueError("no supported physical maps found in proj2d frame")

    ncol = min(4, len(panels))
    fig, axes = plt.subplots((len(panels) + ncol - 1)//ncol, ncol,
                             figsize=(4*ncol, 3.4*((len(panels) + ncol - 1)//ncol)),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, data, cmap, norm) in zip(axes, panels):
        if data.ndim == 3:
            ax.imshow(data, origin="lower", extent=(proj["x_edges"][0], proj["x_edges"][-1],
                      proj["y_edges"][0], proj["y_edges"][-1]), aspect="equal")
        else:
            im = ax.pcolormesh(proj["x_edges"], proj["y_edges"], data,
                               cmap=cmap, norm=_field_norm(data, norm))
            fig.colorbar(im, ax=ax)
        ax.set(title=title, xlabel="image x", ylabel="image y")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read one proj2d file")
    parser.add_argument("path")
    args = parser.parse_args()
    print_metadata(read_proj2d(args.path))
