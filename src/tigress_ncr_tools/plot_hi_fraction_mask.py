#!/usr/bin/env python3
"""Compare face-on H I columns with and without low-xHI VTK cells."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pathena.proj2d_reader import read_proj2d

from .plot_suite_evolution import (
    nearest_indexed_projection,
    projection_number_index,
)
from .plot_suite_xz import read_vtk_snapshot_time


PC_CGS = 3.0856775814913673e18


def _read_vtk_hi_block(path, xhi_min):
    """Read geometry, density, and xHI from one flat Athena VTK rank."""
    geometry = {}
    fields = {}
    dtype_map = {b"float": ">f4", b"double": ">f8", b"int": ">i4"}
    with Path(path).open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"CELL_DATA not found in {path}")
            words = line.split()
            if words[:1] == [b"DIMENSIONS"]:
                geometry["shape"] = np.asarray(words[1:4], dtype=int) - 1
            elif words[:1] == [b"ORIGIN"]:
                geometry["left_edge"] = np.asarray(words[1:4], dtype=float)
            elif words[:1] == [b"SPACING"]:
                geometry["spacing"] = np.asarray(words[1:4], dtype=float)
            elif words[:1] == [b"CELL_DATA"]:
                geometry["cell_count"] = int(words[1])
                break

        while len(fields) < 2:
            line = stream.readline()
            if not line:
                raise ValueError(f"density/xHI fields not found in {path}")
            words = line.split()
            if not words:
                continue
            if words[0] == b"SCALARS":
                name, dtype = words[1], dtype_map[words[2]]
                if not stream.readline().startswith(b"LOOKUP_TABLE"):
                    raise ValueError(f"missing lookup table in {path}")
                count = geometry["cell_count"]
            elif words[0] == b"VECTORS":
                name, dtype = words[1], dtype_map[words[2]]
                count = 3 * geometry["cell_count"]
            else:
                continue
            byte_count = np.dtype(dtype).itemsize * count
            if name in (b"density", b"xHI"):
                raw = stream.read(byte_count)
                if len(raw) != byte_count:
                    raise ValueError(f"truncated {name.decode()} field in {path}")
                array = np.frombuffer(raw, dtype=dtype, count=count)
                shape = tuple(geometry["shape"][::-1])
                fields[name.decode()] = array.reshape(shape).astype(float)
            else:
                stream.seek(byte_count, 1)

    nhi = fields["density"] * fields["xHI"]
    stored_threshold = float(np.float32(xhi_min))
    retained = fields["xHI"] >= stored_threshold
    return {
        "geometry": geometry,
        "raw": nhi.sum(axis=0, dtype=float),
        "masked": np.where(retained, nhi, 0.0).sum(axis=0, dtype=float),
        "retained_cells": retained.sum(axis=0, dtype=int),
    }


def face_on_hi_columns_from_vtk(snapshot, xhi_min=0.01):
    """Assemble raw and cell-masked face-on H I columns from rank VTKs."""
    if not 0.0 <= xhi_min <= 1.0:
        raise ValueError("xhi_min must lie between zero and one")
    paths = sorted(Path(snapshot).glob("*.vtk"))
    if not paths:
        raise FileNotFoundError(f"no VTK rank files under {snapshot}")
    blocks = [_read_vtk_hi_block(path, xhi_min) for path in paths]
    spacing = blocks[0]["geometry"]["spacing"]
    for block in blocks[1:]:
        if not np.allclose(block["geometry"]["spacing"], spacing):
            raise ValueError(f"inconsistent VTK spacing under {snapshot}")

    left_edge = np.min(
        [block["geometry"]["left_edge"] for block in blocks], axis=0
    )
    right_edge = np.max(
        [
            block["geometry"]["left_edge"]
            + block["geometry"]["shape"] * spacing
            for block in blocks
        ],
        axis=0,
    )
    shape = np.rint((right_edge - left_edge) / spacing).astype(int)
    raw = np.zeros((shape[1], shape[0]), dtype=float)
    masked = np.zeros_like(raw)
    retained_cells = np.zeros_like(raw, dtype=int)
    coverage = np.zeros_like(raw, dtype=int)
    for block in blocks:
        geometry = block["geometry"]
        offset_float = (geometry["left_edge"] - left_edge) / spacing
        offset = np.rint(offset_float).astype(int)
        if not np.allclose(offset, offset_float):
            raise ValueError(f"misaligned VTK block under {snapshot}")
        nz, ny, nx = geometry["shape"][::-1]
        yslice = slice(offset[1], offset[1] + ny)
        xslice = slice(offset[0], offset[0] + nx)
        raw[yslice, xslice] += block["raw"]
        masked[yslice, xslice] += block["masked"]
        retained_cells[yslice, xslice] += block["retained_cells"]
        coverage[yslice, xslice] += nz
    if not np.all(coverage == shape[2]):
        raise ValueError(f"incomplete or overlapping VTK blocks under {snapshot}")
    raw *= spacing[2]
    masked *= spacing[2]
    return {
        "raw": raw,
        "masked": masked,
        "removed": raw - masked,
        "retained_cells": retained_cells,
        "coverage": coverage,
        "left_edge": left_edge,
        "right_edge": right_edge,
        "spacing": spacing,
        "shape": shape,
    }


def normalized_log_pdf(values, edges):
    """Return an all-pixel-normalized PDF of positive log columns."""
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if not positive.size:
        raise ValueError("no positive columns are available")
    log_ratio = np.log(positive / positive.mean())
    counts, _ = np.histogram(log_ratio, bins=edges)
    return counts / (values.size * np.diff(edges))


def hi_mask_statistics(raw, masked):
    """Summarize how a cell-level neutral-fraction cut changes H I columns."""
    raw = np.asarray(raw, dtype=float)
    masked = np.asarray(masked, dtype=float)
    if raw.shape != masked.shape or np.any(raw < masked) or np.any(masked < 0.0):
        raise ValueError("raw and masked columns are inconsistent")
    removed = raw - masked
    fraction = np.divide(removed, raw, out=np.zeros_like(raw), where=raw > 0.0)
    return {
        "global_hi_removed_fraction": float(removed.sum() / raw.sum()),
        "median_sightline_removed_fraction": float(np.median(fraction)),
        "p99_sightline_removed_fraction": float(np.percentile(fraction, 99.0)),
        "zero_sightline_fraction_after_mask": float(np.mean(masked <= 0.0)),
    }


def render_hi_mask_test(
    model,
    vtk_number,
    *,
    xhi_min=0.01,
    output_dir=None,
    s_range=(-12.0, 4.0),
    bins=320,
    dpi=180,
    overwrite=False,
):
    """Reconstruct, validate, save, and plot one masked H I snapshot."""
    model = Path(model)
    snapshot = model / "vtk" / f"{vtk_number:04d}"
    stem = f"{model.name}_vtk{vtk_number:04d}_hi_xhi{xhi_min:g}"
    output_dir = Path(output_dir or model.parent / "hi_low_fraction_mask_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    products = {
        suffix: output_dir / f"{stem}.{suffix}"
        for suffix in ("npz", "json", "png")
    }
    existing = [path for path in products.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {existing[0]}")

    columns = face_on_hi_columns_from_vtk(snapshot, xhi_min=xhi_min)
    time = read_vtk_snapshot_time(snapshot)
    index = projection_number_index(model, "theta0")
    proj_path, proj_time, _ = nearest_indexed_projection(
        index, time, tolerance=0.05
    )
    stored = np.asarray(
        read_proj2d(proj_path, fields="nHI")["fields"]["nHI"], dtype=float
    )
    raw = columns["raw"]
    masked = columns["masked"]
    if stored.shape != raw.shape:
        raise ValueError(
            f"proj2d/VTK shape mismatch: {stored.shape} != {raw.shape}"
        )
    ratio = np.divide(
        stored, raw, out=np.full_like(raw, np.nan), where=raw > 0.0
    )
    scale = float(np.nanmedian(ratio))
    relative_error = np.abs(stored - scale * raw) / np.maximum(
        np.abs(stored), np.finfo(float).tiny
    )
    statistics = hi_mask_statistics(raw, masked)
    statistics.update(
        model=model.name,
        vtk_output=int(vtk_number),
        vtk_time_Myr=float(time),
        proj2d_file=proj_path.name,
        proj2d_time_Myr=float(proj_time),
        xHI_keep_threshold=float(xhi_min),
        stored_to_vtk_scale_median=scale,
        stored_vs_scaled_vtk_median_abs_relative_error=float(
            np.nanmedian(relative_error)
        ),
        stored_vs_scaled_vtk_p99_abs_relative_error=float(
            np.nanpercentile(relative_error, 99.0)
        ),
    )

    edges = np.linspace(*s_range, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf_stored = normalized_log_pdf(stored, edges)
    pdf_raw = normalized_log_pdf(raw, edges)
    pdf_masked = normalized_log_pdf(masked, edges)
    np.savez_compressed(
        products["npz"],
        raw_sigma_hi=raw,
        masked_sigma_hi=masked,
        removed_sigma_hi=columns["removed"],
        retained_cell_count=columns["retained_cells"],
        sightline_cell_count=columns["coverage"],
        stored_proj2d_sigma_hi=stored,
        s_edges=edges,
        s_centers=centers,
        pdf_stored=pdf_stored,
        pdf_vtk_raw=pdf_raw,
        pdf_vtk_masked=pdf_masked,
        xhi_threshold=np.asarray(xhi_min),
        time=np.asarray(time),
    )
    products["json"].write_text(json.dumps(statistics, indent=2) + "\n")

    removed_fraction = np.divide(
        columns["removed"],
        raw,
        out=np.zeros_like(raw),
        where=raw > 0.0,
    )
    maps = [stored * PC_CGS, raw * scale * PC_CGS, masked * scale * PC_CGS]
    logs = np.concatenate([np.log10(data[data > 0.0]) for data in maps])
    vmin, vmax = np.percentile(logs, [0.2, 99.8])
    with mpl.rc_context({"figure.dpi": dpi}):
        figure, axes = plt.subplots(
            2, 3, figsize=(14, 8.5), constrained_layout=True
        )
        titles = (
            "stored proj2d",
            "VTK reconstruction",
            f"VTK, xHI >= {xhi_min:g}",
        )
        for axis, data, title in zip(axes[0], maps, titles):
            image = np.full_like(data, np.nan)
            image[data > 0.0] = np.log10(data[data > 0.0])
            artist = axis.imshow(
                image,
                origin="lower",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_title(title)
        colorbar = figure.colorbar(artist, ax=axes[0], shrink=0.9)
        colorbar.set_label(
            r"$\log_{10}N_{\rm HI}\;[\mathrm{cm}^{-2}]$"
        )
        axes[1, 0].semilogy(
            centers, pdf_stored, label="stored proj2d", lw=2
        )
        axes[1, 0].semilogy(
            centers, pdf_raw, "--", label="VTK raw"
        )
        axes[1, 0].semilogy(
            centers,
            pdf_masked,
            label=f"VTK xHI >= {xhi_min:g}",
            lw=2,
        )
        axes[1, 0].set(
            xlabel=r"$s=\ln(\Sigma_{\rm HI}/\langle\Sigma_{\rm HI}\rangle)$",
            ylabel=r"$p_A(s)$",
        )
        axes[1, 0].legend(fontsize=8)
        fraction_artist = axes[1, 1].imshow(
            removed_fraction,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=max(0.01, np.percentile(removed_fraction, 99.5)),
        )
        axes[1, 1].set_title("fraction of H I column removed")
        figure.colorbar(fraction_artist, ax=axes[1, 1])
        positive = raw > 0.0
        axes[1, 2].scatter(
            np.log(raw[positive] / raw[positive].mean()),
            removed_fraction[positive],
            s=2,
            alpha=0.15,
        )
        axes[1, 2].set(
            xlabel="raw VTK s",
            ylabel="H I fraction from masked cells",
            yscale="log",
            ylim=(1e-7, 1.2),
        )
        figure.suptitle(
            f"{model.name}, t={time:.3f} Myr (VTK {vtk_number:04d})"
        )
        figure.savefig(products["png"], dpi=dpi)
        plt.close(figure)
    return statistics, products


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("vtk_number", type=int)
    parser.add_argument("--xhi-min", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--s-min", type=float, default=-12.0)
    parser.add_argument("--s-max", type=float, default=4.0)
    parser.add_argument("--bins", type=int, default=320)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0.0 <= args.xhi_min <= 1.0:
        parser.error("--xhi-min must lie between zero and one")
    if args.s_max <= args.s_min or args.bins <= 0 or args.dpi <= 0:
        parser.error("require s-max > s-min and positive bins/dpi")
    statistics, products = render_hi_mask_test(
        args.model,
        args.vtk_number,
        xhi_min=args.xhi_min,
        output_dir=args.output_dir,
        s_range=(args.s_min, args.s_max),
        bins=args.bins,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )
    print(json.dumps(statistics, indent=2))
    print(f"wrote {products['png']}")


if __name__ == "__main__":
    main()
