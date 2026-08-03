"""Collective diagnostic plots for TIGRESS-NCR pdf2d and slicevtk output."""

import glob
import os
import re
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from .pdf2d_reader import read_pdf2d
from .slice_fields import derive_plane_fields
from .slicevtk_reader import read_slicevtk
from .units import DEFAULT_MUH, PC_CGS


FIELD_META = {
    "N_nH": dict(short=r"$N_{\rm H}$", label=r"$N_{\rm H}\;[\rm cm^{-2}]$", cmap="pink_r", vmin=1e18, vmax=1e23, log=True),
    "N_nH2": dict(short=r"$N_{\rm H_2}$", label=r"$N_{\rm H_2}\;[\rm cm^{-2}]$", cmap="pink_r", vmin=1e18, vmax=1e23, log=True),
    "N_nHI": dict(short=r"$N_{\rm H\,I}$", label=r"$N_{\rm H\,I}\;[\rm cm^{-2}]$", cmap="pink_r", vmin=1e18, vmax=1e23, log=True),
    "N_nHII": dict(short=r"$N_{\rm H\,II}$", label=r"$N_{\rm H\,II}\;[\rm cm^{-2}]$", cmap="pink_r", vmin=1e18, vmax=1e23, log=True),
    "nH": dict(short=r"$n_{\rm H}$", label=r"$n_{\rm H}\;[\rm cm^{-3}]$", cmap="Spectral_r", vmin=1e-4, vmax=1e4, log=True),
    "nH2": dict(short=r"$n_{\rm H_2}$", label=r"$n_{\rm H_2}\;[\rm cm^{-3}]$", cmap="Spectral_r", vmin=1e-4, vmax=1e4, log=True),
    "nHI": dict(short=r"$n_{\rm H\,I}$", label=r"$n_{\rm H\,I}\;[\rm cm^{-3}]$", cmap="Spectral_r", vmin=1e-4, vmax=1e4, log=True),
    "nHII": dict(short=r"$n_{\rm H\,II}$", label=r"$n_{\rm H\,II}\;[\rm cm^{-3}]$", cmap="Spectral_r", vmin=1e-4, vmax=1e4, log=True),
    "T": dict(short=r"$T$", label=r"$T\;[\rm K]$", cmap="RdYlBu_r", vmin=1e1, vmax=1e7, log=True),
    "P": dict(short=r"$P/k_{\rm B}$", label=r"$P/k_{\rm B}\;[\rm K\,cm^{-3}]$", cmap="inferno", vmin=1e2, vmax=1e7, log=True),
    "vz": dict(short=r"$v_z$", label=r"$v_z\;[\rm km\,s^{-1}]$", cmap="RdBu_r", vmin=-100.0, vmax=100.0, log=False),
    "Bmag": dict(short=r"$|\mathbf{B}|$", label=r"$|\mathbf{B}|\;[\mu\rm G]$", cmap="cividis", vmin=1e-1, vmax=1e2, log=True),
    "Erad_PE": dict(short=r"$\mathcal{E}_{\rm PE}$", label=r"$\mathcal{E}_{\rm PE}\;[\rm erg\,cm^{-3}]$", cmap="viridis", vmin=1e-15, vmax=1e-10, log=True),
    "Erad_PH": dict(short=r"$\mathcal{E}_{\rm PH}$", label=r"$\mathcal{E}_{\rm PH}\;[\rm erg\,cm^{-3}]$", cmap="viridis", vmin=1e-15, vmax=1e-10, log=True),
}


_PDF_RE = re.compile(r"^(?P<problem>.+)\.(?P<num>\d+)\.(?P<id>.+)\.pdf2d$")
_PROJ_RE = re.compile(r"^(?P<problem>.+)\.(?P<num>\d+)\.(?P<id>.+)\.proj2d$")
_SLICE_RE = re.compile(r"^(?P<problem>.+)\.(?P<num>\d+)\.(?P<id>.+)\.slice\.vtk$")


def _parse_output_filename(path, kind):
    regex = _PDF_RE if kind == "pdf2d" else _PROJ_RE if kind == "proj2d" else _SLICE_RE
    match = regex.match(os.path.basename(path))
    if match is None:
        return None
    return match.group("problem"), int(match.group("num")), match.group("id")


def _find_output_files(basedir, kind, num=None, out_id=None):
    if kind == "pdf2d":
        roots = [os.path.join(basedir, "pdf2d")]
        suffix = "pdf2d"
    elif kind == "proj2d":
        roots = [os.path.join(basedir, "proj2d")]
        suffix = "proj2d"
    elif kind == "slicevtk":
        roots = [os.path.join(basedir, "slice")]
        suffix = "slice.vtk"
    else:
        raise ValueError(f"unknown output kind {kind!r}")

    paths = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        if out_id is None:
            pattern = os.path.join(root, "*", f"*.{suffix}")
        else:
            pattern = os.path.join(root, out_id, f"*.{suffix}")
        for path in glob.glob(pattern):
            parsed = _parse_output_filename(path, kind)
            if parsed is None:
                continue
            _, fnum, fid = parsed
            if num is not None and fnum != num:
                continue
            if out_id is not None and fid != out_id:
                continue
            paths.append(path)
    return sorted(paths)


def _detect_pdf_id(basedir, num):
    preferred = "x1-x2"
    if _find_output_files(basedir, "pdf2d", num=num, out_id=preferred):
        return preferred
    for path in _find_output_files(basedir, "pdf2d", num=num):
        pdf = read_pdf2d(path)
        if pdf.get("binx_name") == "x1" and pdf.get("biny_name") == "x2":
            return pdf["id"]
    raise FileNotFoundError(
        f"could not find an x1-x2 pdf2d file for output {num:04d} under {basedir}"
    )


def _detect_slice_id(basedir, num):
    preferred = "slice-x0"
    if _find_output_files(basedir, "slicevtk", num=num, out_id=preferred):
        return preferred
    paths = _find_output_files(basedir, "slicevtk", num=num)
    if not paths:
        raise FileNotFoundError(
            f"could not find a slicevtk file for output {num:04d} under {basedir}"
        )
    parsed = _parse_output_filename(paths[0], "slicevtk")
    return parsed[2]


def _detect_problem_id(basedir, num, pdf_id=None, slice_id=None):
    candidates = []
    if pdf_id is not None:
        candidates.extend(_find_output_files(basedir, "pdf2d", num=num, out_id=pdf_id))
    if slice_id is not None:
        candidates.extend(_find_output_files(basedir, "slicevtk", num=num, out_id=slice_id))
    if not candidates:
        candidates.extend(_find_output_files(basedir, "pdf2d", num=num))
        candidates.extend(_find_output_files(basedir, "slicevtk", num=num))
    problems = set()
    for path in candidates:
        kind = "pdf2d" if path.endswith(".pdf2d") else "slicevtk"
        parsed = _parse_output_filename(path, kind)
        if parsed is not None:
            problems.add(parsed[0])
    problems = sorted(problems)
    if len(problems) == 1:
        return problems[0]
    if not problems:
        raise FileNotFoundError(
            f"could not infer problem_id for output {num:04d} under {basedir}"
        )
    raise ValueError(
        f"could not infer a unique problem_id for output {num:04d}: {problems}"
    )


def _resolve_summary_ids(basedir, num, problem_id, pdf_id, slice_id):
    if pdf_id is None:
        pdf_id = _detect_pdf_id(basedir, num)
    if slice_id is None:
        slice_id = _detect_slice_id(basedir, num)
    if problem_id is None:
        problem_id = _detect_problem_id(basedir, num, pdf_id=pdf_id, slice_id=slice_id)
    return problem_id, pdf_id, slice_id

def _frame_path(basedir, problem_id, num, out_id, kind):
    stem = f"{problem_id}.{num:04d}.{out_id}"
    if kind == "pdf2d":
        candidates = [
            os.path.join(basedir, "pdf2d", out_id, f"{stem}.pdf2d"),
            os.path.join(basedir, "pdf", out_id, f"{stem}.pdf2d"),
        ]
    elif kind == "slicevtk":
        candidates = [os.path.join(basedir, "slice", out_id, f"{stem}.slice.vtk")]
    else:
        raise ValueError(f"unknown output kind {kind!r}")
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(" or ".join(candidates))


def _slice_fields(slc, plane, muH=DEFAULT_MUH):
    return derive_plane_fields(slc, plane, muH=muH)


def _projection_fields(pdf):
    missing = [f for f in ("nH", "nH2", "nHI") if f not in pdf["weights"]]
    if missing:
        raise KeyError(f"pdf2d projection is missing weights: {', '.join(missing)}")
    dA = pdf["x_spacing"] * pdf["y_spacing"]
    out = {f"N_{name}": pdf["weights"][name] / dA * PC_CGS for name in ("nH", "nH2", "nHI")}
    out["N_nHII"] = out["N_nH"] - out["N_nHI"] - 2.0 * out["N_nH2"]
    return out


def _norm(meta):
    if meta.get("log", False):
        return LogNorm(meta["vmin"], meta["vmax"])
    return Normalize(meta["vmin"], meta["vmax"])


def _format_cbar_value(value, log=False):
    if log:
        if value <= 0.0:
            return ""
        exponent = np.log10(value)
        if abs(exponent - round(exponent)) < 1.0e-10:
            return f"{int(round(exponent))}"
        return f"{exponent:.2g}"
    if value == 0.0:
        return "0"
    avalue = abs(value)
    if avalue < 1.0e-2 or avalue >= 1.0e3:
        return f"{value:.0e}"
    if avalue < 1.0:
        return f"{value:.2g}"
    return f"{value:.3g}"


def _nice_scale(span):
    return 500.0


def _text_outline(linewidth=1.9, foreground="black"):
    return [path_effects.withStroke(linewidth=linewidth, foreground=foreground)]


def _axis_display_name(name):
    return {"x1": "x", "x2": "y", "x3": "z"}.get(name, name)


def _add_inset_colorbar(fig, ax, im, field_name, meta, height=0.055):
    cax = ax.inset_axes([0.10, 0.88, 0.80, height])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.outline.set_linewidth(0.9)
    effects = _text_outline()
    cax.text(0.02, 0.5, _format_cbar_value(meta["vmin"], log=meta.get("log", False)),
             ha="left", va="center", fontsize=8, color="white",
             transform=cax.transAxes, path_effects=effects)
    cax.text(0.50, 0.5, field_name,
             ha="center", va="center", fontsize=10, color="white",
             transform=cax.transAxes, path_effects=effects)
    cax.text(0.98, 0.5, _format_cbar_value(meta["vmax"], log=meta.get("log", False)),
             ha="right", va="center", fontsize=8, color="white",
             transform=cax.transAxes, path_effects=effects)
    return cbar


def _add_coordinate_glyph(ax, x_edges, y_edges, axis_names,
                          glyph_scale=1.0, scale_label_gap=0.025,
                          axis_label_size=10, scale_label_size=10):
    if axis_names is None:
        axis_names = ("x", "y")
    axis_names = tuple(_axis_display_name(n) for n in axis_names)
    xmin, xmax = float(np.nanmin(x_edges)), float(np.nanmax(x_edges))
    ymin, ymax = float(np.nanmin(y_edges)), float(np.nanmax(y_edges))
    xspan = xmax - xmin
    yspan = ymax - ymin
    length = 0.11 * glyph_scale * min(abs(xspan), abs(yspan))
    scale = _nice_scale(min(abs(xspan), abs(yspan)))
    color = "white"
    effects = _text_outline()

    x0 = xmin + 0.07 * xspan
    yb = ymin + 0.05 * yspan
    y0 = yb
    ax.annotate("", xy=(x0 + length, y0), xytext=(x0, y0),
                xycoords="data",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                path_effects=effects))
    ax.annotate("", xy=(x0, y0 + length), xytext=(x0, y0),
                xycoords="data",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                path_effects=effects))
    ax.text(x0 + 1.12 * length, y0, axis_names[0], color=color,
            fontsize=axis_label_size,
            ha="left", va="center", path_effects=effects)
    ax.text(x0, y0 + 1.12 * length, axis_names[1], color=color,
            fontsize=axis_label_size,
            ha="center", va="bottom", path_effects=effects)

    xb1 = xmax - 0.07 * xspan
    xb0 = max(xmin + 0.07 * xspan, xb1 - scale)
    ax.plot([xb0, xb1], [yb, yb], color=color, lw=2.0,
            path_effects=effects, solid_capstyle="butt")
    ax.text(0.5 * (xb0 + xb1), yb + scale_label_gap * yspan, f"{scale:g} pc",
            color=color, fontsize=scale_label_size, ha="center", va="bottom",
            path_effects=effects)


def _x2_slice_aspect(slc):
    plane = slc["planes"].get("x2")
    if plane is None:
        return 1.0 / 6.0
    xspan = float(np.nanmax(plane["x_edges"]) - np.nanmin(plane["x_edges"]))
    yspan = float(np.nanmax(plane["y_edges"]) - np.nanmin(plane["y_edges"]))
    return abs(xspan / yspan) if yspan != 0.0 else 1.0


def _summary_width_ratios(slc):
    # The x2 strip has 7 equal-aspect panels spanning all 4 rows. A square
    # panel in the first three columns occupies one row, so the ideal fourth
    # column width is 7 * 4 * (x2 panel width / height).
    fourth = min(14.0, max(4.1, 28.0 * _x2_slice_aspect(slc)))
    return [1.0, 1.0, 1.0, fourth]


def _auto_summary_figsize(width_ratios, base_height=11.0,
                          min_width=18.0, max_width=34.0):
    target_ratio = sum(width_ratios) / 4.0
    target_width = base_height * target_ratio
    if target_width > max_width:
        return (max_width, max(7.0, max_width / target_ratio))
    return (max(min_width, target_width), base_height)


def _plot_panel(fig, ax, x_edges, y_edges, data, name, title=None,
                colorbar=True, axis_names=None, colorbar_height=0.055,
                glyph=True, glyph_scale=1.0, scale_label_gap=0.025,
                axis_label_size=10, scale_label_size=10):
    meta = FIELD_META[name]
    arr = np.asarray(data)
    if meta.get("log", False):
        arr = np.ma.masked_less_equal(arr, 0.0)
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        arr,
        shading="auto",
        cmap=meta["cmap"],
        norm=_norm(meta),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if colorbar:
        _add_inset_colorbar(fig, ax, im, title or meta.get("short", name), meta,
                            height=colorbar_height)
    else:
        ax.text(0.50, 0.93, title or meta.get("short", name),
                ha="center", va="top", fontsize=11, color="white",
                transform=ax.transAxes, path_effects=_text_outline())
    if glyph:
        _add_coordinate_glyph(ax, x_edges, y_edges, axis_names,
                              glyph_scale=glyph_scale,
                              scale_label_gap=scale_label_gap,
                              axis_label_size=axis_label_size,
                              scale_label_size=scale_label_size)
    return im


def plot_summary(
    basedir,
    problem_id=None,
    num=0,
    pdf_id=None,
    slice_id=None,
    muH=DEFAULT_MUH,
    figsize=None,
    colorbar=True,
):
    """Plot the standard four-column R8 summary figure.

    Parameters
    ----------
    basedir : str
        Directory containing ``pdf2d/`` and ``slice/`` output directories.
    problem_id : str or None
        Athena problem id used in output filenames. If None, infer it from
        matching output files in ``basedir``.
    num : int
        Output number.
    pdf_id : str or None
        pdf2d id for the x1-x2 projection. If None, prefer ``x1-x2`` and then
        scan pdf2d files for coordinate axes ``x1,x2``.
    slice_id : str or None
        slicevtk id containing x1/x2/x3 slices. If None, prefer ``slice-x0``
        and then use the first slicevtk id found for ``num``.
    muH : float
        Hydrogen mass per H nucleus in units of proton mass for unit conversion.
    figsize : tuple or None
        Matplotlib figure size. If None, choose a size from the dynamic column
        ratios so z-extent changes do not collapse the x2 panel column or leave
        large whitespace in the first three columns.
    colorbar : bool
        Add one colorbar to each panel.

    Returns
    -------
    fig, axes : matplotlib Figure and dict
        ``axes`` contains lists keyed by ``projection``, ``slice_x3_species``,
        ``slice_x3_thermal``, and ``slice_x2``.
    """
    problem_id, pdf_id, slice_id = _resolve_summary_ids(
        basedir, num, problem_id, pdf_id, slice_id
    )
    pdf = read_pdf2d(_frame_path(basedir, problem_id, num, pdf_id, "pdf2d"))
    slc = read_slicevtk(_frame_path(basedir, problem_id, num, slice_id, "slicevtk"))

    proj = _projection_fields(pdf)
    x3 = _slice_fields(slc, "x3", muH=muH)
    x2 = _slice_fields(slc, "x2", muH=muH)

    width_ratios = _summary_width_ratios(slc)
    if figsize is None:
        figsize = _auto_summary_figsize(width_ratios)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    outer = fig.add_gridspec(4, 4, width_ratios=width_ratios)
    axes = {
        "projection": [],
        "slice_x3_species": [],
        "slice_x3_thermal": [],
        "slice_x2": [],
    }

    proj_order = [("N_nH", "nH"), ("N_nH2", "nH2"), ("N_nHI", "nHI"), ("N_nHII", "nHII")]
    species_order = ["nH", "nH2", "nHI", "nHII"]
    thermal_order = ["T", "P", "Erad_PE", "Erad_PH"]
    x2_order = ["nH", "T", "vz", "P", "Bmag", "Erad_PE", "Erad_PH"]

    for row, (field, title) in enumerate(proj_order):
        ax = fig.add_subplot(outer[row, 0])
        decorate = row == 0
        _plot_panel(fig, ax, pdf["x_edges"], pdf["y_edges"], proj[field], field,
                    title=FIELD_META[field]["short"],
                    colorbar=colorbar and decorate,
                    axis_names=(pdf["binx_name"], pdf["biny_name"]),
                    colorbar_height=0.070, glyph=decorate)
        axes["projection"].append(ax)

    plane = slc["planes"]["x3"]
    for row, field in enumerate(species_order):
        ax = fig.add_subplot(outer[row, 1])
        decorate = row == 0
        _plot_panel(fig, ax, plane["x_edges"], plane["y_edges"], x3[field], field,
                    title=FIELD_META[field]["short"],
                    colorbar=colorbar and decorate,
                    axis_names=(plane["xaxis"], plane["yaxis"]),
                    colorbar_height=0.070, glyph=decorate)
        axes["slice_x3_species"].append(ax)

    for row, field in enumerate(thermal_order):
        ax = fig.add_subplot(outer[row, 2])
        _plot_panel(fig, ax, plane["x_edges"], plane["y_edges"], x3[field], field,
                    title=FIELD_META[field]["short"], colorbar=colorbar,
                    axis_names=(plane["xaxis"], plane["yaxis"]),
                    colorbar_height=0.070, glyph=False)
        axes["slice_x3_thermal"].append(ax)

    sub = outer[:, 3].subgridspec(1, len(x2_order))
    plane = slc["planes"]["x2"]
    for n, field in enumerate(x2_order):
        ax = fig.add_subplot(sub[0, n])
        if field not in x2:
            raise KeyError(f"x2 slice cannot derive {field!r}")
        _plot_panel(fig, ax, plane["x_edges"], plane["y_edges"], x2[field], field,
                    title=FIELD_META[field]["short"], colorbar=colorbar,
                    axis_names=(plane["xaxis"], plane["yaxis"]),
                    colorbar_height=0.055, glyph=(n == 0), glyph_scale=1.45,
                    scale_label_gap=0.012)
        axes["slice_x2"].append(ax)

    fig.suptitle(f"{problem_id} output {num:04d}  t={pdf['time']:.6g}", fontsize=12)
    return fig, axes
