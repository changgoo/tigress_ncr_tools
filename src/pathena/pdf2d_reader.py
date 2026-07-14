"""Reader for pdf2d output files written by Athena-TIGRESS-NCR.

The pdf2d format is a binary VTK legacy STRUCTURED_POINTS file containing
a 2D weighted histogram (or projection) with one SCALARS record per weight.
The second header line carries pdf2d-specific metadata:

    PDF2D <pdf_id> at time= <t>, axes=<binx>,<biny>, log=<binx_log>,<biny_log>

See src/output_pdf2d.c for the writer.

Typical use:

    from pathena.pdf2d_reader import read_pdf2d

    pdf = read_pdf2d("pdf/n-T/R8_8pc_NCR.0010.n-T.pdf2d")
    pdf["time"]          # simulation time
    pdf["binx_name"]     # "nH"
    pdf["binx_log"]      # 1  -> binx edges/centers are in log10 space
    pdf["x_edges"]       # length Nbinx+1 array of bin edges along binx
    pdf["x_centers"]     # length Nbinx       array of bin centers
    pdf["y_edges"]       # length Nbiny+1 array along biny
    pdf["y_centers"]     # length Nbiny       array
    pdf["weights"]       # dict {weight_name: ndarray of shape (Nbiny, Nbinx)}

Use ``pdf["x_edges_linear"]`` / ``pdf["x_centers_linear"]`` to get the
linear-space versions (i.e. 10**values) when ``binx_log == 1``.
"""

import glob
import os
import re
import sys

import numpy as np

from .vtk_helper import (
    axis_arrays,
    read_ascii_line,
    read_structured_points_header,
    read_vtk_magic,
    read_vtk_scalar_2d,
)

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# Second header line, e.g.:
#   PDF2D n-T at time= 1.234e+00, axes=nH,T, log=1,1
_HDR_RE = re.compile(
    r"^PDF2D\s+(?P<id>\S+)\s+at\s+time=\s*(?P<time>[-+0-9.eE]+),"
    r"\s*axes=(?P<binx>[^,]+),(?P<biny>[^,]+),"
    r"\s*log=(?P<lx>\d+),(?P<ly>\d+)"
)



def _bin_arrays(origin, spacing, n, is_log):
    """Build edges and centers (plus their linear-space counterparts).

    edges    has length n+1, centers has length n. If is_log==1, the values
    stored in the file are already log10(...), so linear-space arrays are
    10**values.
    """
    edges, centers = axis_arrays(origin, spacing, n)
    if is_log:
        edges_lin = np.power(10.0, edges)
        centers_lin = np.power(10.0, centers)
    else:
        edges_lin = edges
        centers_lin = centers
    return edges, centers, edges_lin, centers_lin


def read_pdf2d(path):
    """Read one pdf2d file.

    Parameters
    ----------
    path : str
        Path to the .pdf2d file.

    Returns
    -------
    dict with keys:

        time, id, binx_name, biny_name, binx_log, biny_log
        Nbinx, Nbiny
        x_origin, y_origin, x_spacing, y_spacing
        x_edges, x_centers            (in stored space; log10 if binx_log)
        x_edges_linear, x_centers_linear   (always linear)
        y_edges, y_centers, y_edges_linear, y_centers_linear
        weight_names : list of str (preserves file order)
        weights      : dict[name -> ndarray of shape (Nbiny, Nbinx)]
                       (row index = iy along biny, column index = ix along binx)
        header_lines : the raw ASCII header lines, for debugging
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as fp:
        header_lines = []

        # Line 1: VTK version magic
        ln = read_vtk_magic(fp, path)
        header_lines.append(ln.rstrip())

        # Line 2: pdf2d metadata
        meta_ln = read_ascii_line(fp).rstrip()
        header_lines.append(meta_ln)
        m = _HDR_RE.match(meta_ln)
        if not m:
            raise ValueError(
                f"{path}: cannot parse PDF2D metadata line: {meta_ln!r}"
            )
        out = {
            "id": m.group("id"),
            "time": float(m.group("time")),
            "binx_name": m.group("binx").strip(),
            "biny_name": m.group("biny").strip(),
            "binx_log": int(m.group("lx")),
            "biny_log": int(m.group("ly")),
        }

        # Lines 3..: BINARY, DATASET, DIMENSIONS, ORIGIN, SPACING, CELL_DATA
        extra_header, dims, origin, spacing, _ = read_structured_points_header(fp, path)
        header_lines.extend(extra_header)
        Nx = int(dims[0] - 1)
        Ny = int(dims[1] - 1)
        x_orig = float(origin[0])
        y_orig = float(origin[1])
        dx = float(spacing[0])
        dy = float(spacing[1])

        out.update({
            "Nbinx": Nx,
            "Nbiny": Ny,
            "x_origin": x_orig,
            "y_origin": y_orig,
            "x_spacing": dx,
            "y_spacing": dy,
        })

        xe, xc, xe_lin, xc_lin = _bin_arrays(x_orig, dx, Nx, out["binx_log"])
        ye, yc, ye_lin, yc_lin = _bin_arrays(y_orig, dy, Ny, out["biny_log"])
        out.update({
            "x_edges": xe, "x_centers": xc,
            "x_edges_linear": xe_lin, "x_centers_linear": xc_lin,
            "y_edges": ye, "y_centers": yc,
            "y_edges_linear": ye_lin, "y_centers_linear": yc_lin,
        })

        # Now alternate: blank line / SCALARS <name> float 1 / LOOKUP_TABLE / <Nx*Ny float32 BE>
        weight_names = []
        weights = {}
        while True:
            ln = read_ascii_line(fp)
            if not ln:
                break  # EOF — done
            stripped = ln.strip()
            if not stripped:
                continue  # blank separator
            if not stripped.startswith("SCALARS"):
                # Unknown header line; skip
                continue
            tok = stripped.split()
            # SCALARS <name> float [num_components]
            name = tok[1]
            # The next line must be LOOKUP_TABLE
            lookup_ln = read_ascii_line(fp).strip()
            if not lookup_ln.startswith("LOOKUP_TABLE"):
                raise ValueError(
                    f"{path}: expected LOOKUP_TABLE after SCALARS {name}, "
                    f"got {lookup_ln!r}"
                )
            arr = read_vtk_scalar_2d(fp, path, name, Nx, Ny)
            weight_names.append(name)
            weights[name] = arr

        out["weight_names"] = weight_names
        out["weights"] = weights
        out["header_lines"] = header_lines

    return out


def read_pdf2d_series(pattern):
    """Read every file matching a glob pattern, return list of dicts sorted
    by filename. Convenience wrapper around read_pdf2d.

        from pathena.pdf2d_reader import read_pdf2d_series
        frames = read_pdf2d_series("pdf/n-T/*.pdf2d")
        times = [f["time"] for f in frames]
    """
    paths = sorted(glob.glob(pattern))
    return [read_pdf2d(p) for p in paths]

def print_metadata(d):
    print(f"id={d['id']!r}  time={d['time']:g}")
    print(f"axes: {d['binx_name']} (log={d['binx_log']})  x  "
          f"{d['biny_name']} (log={d['biny_log']})")
    print(f"shape: Nbinx={d['Nbinx']}  Nbiny={d['Nbiny']}")
    print(f"x range: {d['x_edges'][0]:g} -> {d['x_edges'][-1]:g}")
    print(f"y range: {d['y_edges'][0]:g} -> {d['y_edges'][-1]:g}")
    print(f"weights ({len(d['weight_names'])}):")
    for name in d["weight_names"]:
        a = d["weights"][name]
        nz = int(np.count_nonzero(a))
        nan = int(np.isnan(a).sum())
        print(f"  {name:22s}  sum={a.sum():.4e}  nonzero={nz}/{a.size}  nans={nan}")

def plot_all(pdf, ncol=5, projection=False):
    if projection:
        wfields = pdf["weight_names"][1:]
    else:
        wfields = pdf["weight_names"]
    nw = len(wfields)
    if nw < ncol:
        ncol = nw
    nrow=nw//ncol + (nw%ncol > 0)
    fig, axes = plt.subplots(nrow,ncol,figsize=(ncol*4,nrow*3),constrained_layout=True)

    for ax,wf in zip(axes.flat,wfields):
        if projection:
            data = pdf["weights"][wf]/pdf["weights"]["volume"]
        else:
            data = pdf["weights"][wf]
        dmin, dmax = data.min(), data.max()
        if dmin == dmax:
            dmin = dmin - 1
            dmax = dmax + 1
        if dmin == 0.0:
            dmin = data[data>0.0].min()
        if dmin < 0.0:
            norm = Normalize(dmin, dmax)
        else:
            norm = LogNorm(dmin, dmax)
        plt.sca(ax)
        plt.pcolormesh(pdf["x_edges"],pdf["y_edges"],data,norm=norm)
        plt.colorbar(label=wf)
        plt.xlabel(pdf["binx_name"])
        plt.ylabel(pdf["biny_name"])
        if projection:
            ax.set_aspect("equal")
    return fig

def plot_column_density(pdf):
    pc_cgs = 3.0856775814913674e+18
    pdf["weights"]["nHII"] = pdf["weights"]["nH"] - pdf["weights"]["nHI"] - 2.0*pdf["weights"]["nH2"]
    dA=pdf["x_spacing"]*pdf["y_spacing"]
    fig, axes = plt.subplots(1, 4, figsize=(15,3),constrained_layout=True)
    for wf,ax in zip(["nH","nHI","nH2","nHII"],axes.flat):
        plt.sca(ax)
        Sigma = pdf["weights"][wf]/dA*pc_cgs
        plt.pcolormesh(pdf["x_edges"],pdf["y_edges"], Sigma, norm=LogNorm(1.e18,1.e22))
        plt.colorbar(label=wf.upper())
        ax.set_aspect('equal')
        plt.title(pdf["time"])

def read_all_pdfs(basedir, problem_id, verbose=True):
    pdf_ids = os.listdir(os.path.join(basedir,"pdf2d"))
    pdf_dict = dict()
    for pdf_id in pdf_ids:
        if verbose:
            print(f"Searching for id={pdf_id}....................")
        pdf_pattern = os.path.join(basedir,"pdf2d",pdf_id,f"{problem_id}.????.{pdf_id}.pdf2d")
        pdfs = read_pdf2d_series(pdf_pattern)
        if verbose:
            print(f"  found and read {len(pdfs)} pdf files")
        pdf_dict[pdf_id] = pdfs
    return pdf_dict

if __name__ == "__main__":
    # Minimal CLI: python -m pdf2d_reader <path>
    if len(sys.argv) != 2:
        print("Usage: python -m pdf2d_reader <file.pdf2d>")
        sys.exit(1)
    d = read_pdf2d(sys.argv[1])
    print_metadata(d)
