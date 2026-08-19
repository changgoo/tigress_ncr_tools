#!/usr/bin/env python3
"""Render SFR-ranked XZ gas-surface-density grids for an NCR suite."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pathena.pdf2d_reader import read_pdf2d, read_pdf2d_metadata
from pathena.units import star_particle_units

from .plot_suite_evolution import (
    DEFAULT_MODEL_GLOB,
    DEFAULT_SFR_RANGE,
    DEFAULT_SUITE,
    make_grid_movie,
    rank_models_by_sfr,
    short_model_name,
    write_model_order,
)


DEFAULT_PDF_ID = "x1-x3"
DEFAULT_OUTPUT_NAME = "surface_density_evolution_x1-x3"
DEFAULT_FRAME_STEM = "surface_density_xz"
DEFAULT_VMIN = 0.01
DEFAULT_VMAX = 100.0
VTK_TIME_RE = re.compile(r"\btime=\s*([+\-0-9.eE]+)")
VTK_OUTPUT_INTERVAL = 10.0


def discover_xz_models(
    suite,
    model_glob=DEFAULT_MODEL_GLOB,
    pdf_id=DEFAULT_PDF_ID,
):
    """Return models with a primary history and the requested PDF2D output."""
    suite = Path(suite)
    models = []
    for model in sorted(suite.glob(model_glob)):
        histories = [
            path
            for path in sorted((model / "hst").glob("*.hst"))
            if ".phase" not in path.name and ".whole" not in path.name
        ]
        if (
            model.is_dir()
            and histories
            and (model / "pdf2d" / pdf_id).is_dir()
        ):
            models.append(model)
    if not models:
        raise FileNotFoundError(
            f"no models under {suite} match {model_glob!r} with hst and "
            f"pdf2d/{pdf_id}"
        )
    return models


def xz_number_index(model, pdf_id=DEFAULT_PDF_ID):
    """Return output-number to PDF2D path mappings for one model."""
    directory = Path(model) / "pdf2d" / pdf_id
    index = {}
    for path in directory.glob("*.pdf2d"):
        try:
            number = int(path.name.split(".")[-3])
        except (IndexError, ValueError):
            continue
        if number in index:
            raise ValueError(
                f"duplicate output number {number:04d} under {directory}"
            )
        index[number] = path
    if not index:
        raise FileNotFoundError(f"no PDF2D files under {directory}")
    return index


def nearest_indexed_xz(
    index,
    target_time,
    guess_offset=0,
    *,
    tolerance=0.05,
    search_radius=32,
):
    """Find a nearby filename whose PDF2D header time matches target_time."""
    target_number = int(round(target_time))
    base_number = target_number + int(guess_offset)
    corrections = [0]
    for distance in range(1, search_radius + 1):
        corrections.extend((distance, -distance))

    best = None
    for correction in corrections:
        number = base_number + correction
        path = index.get(number)
        if path is None:
            continue
        stored_time = float(read_pdf2d_metadata(path)["time"])
        difference = abs(stored_time - target_time)
        candidate = (difference, path, stored_time, number - target_number)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if difference <= tolerance:
            return path, stored_time, number - target_number

    if best is None:
        raise ValueError(
            f"no XZ projection candidate is available near t={target_time:g}"
        )
    raise ValueError(
        f"nearest XZ projection to t={target_time:g} is offset by {best[0]:g}"
    )


def nearest_xz_paths(
    indices,
    target_time,
    guess_offsets=None,
    *,
    tolerance=0.05,
    search_radius=32,
):
    """Select one XZ PDF2D file per model at a common physical time."""
    if guess_offsets is None:
        guess_offsets = [0] * len(indices)
    if len(guess_offsets) != len(indices):
        raise ValueError("guess offsets and XZ indices must have equal lengths")

    paths = []
    times = []
    offsets = []
    for index, guess in zip(indices, guess_offsets):
        path, stored_time, offset = nearest_indexed_xz(
            index,
            target_time,
            guess,
            tolerance=tolerance,
            search_radius=search_radius,
        )
        paths.append(path)
        times.append(stored_time)
        offsets.append(offset)
    return paths, np.asarray(times), offsets


def read_vtk_snapshot_time(snapshot):
    """Read the physical time from one rank file in a flat VTK snapshot."""
    first_file = next(Path(snapshot).glob("*.vtk"), None)
    if first_file is None:
        raise FileNotFoundError(f"no VTK rank files under {snapshot}")
    with first_file.open("rb") as stream:
        stream.readline()
        description = stream.readline().decode("ascii", errors="replace")
    match = VTK_TIME_RE.search(description)
    if match is None:
        raise ValueError(f"cannot read VTK time from {first_file}")
    return float(match.group(1))


def vtk_snapshot_index(model):
    """Return output-number mappings for the archived VTK snapshots."""
    snapshots = {}
    for path in sorted((Path(model) / "vtk").glob("[0-9][0-9][0-9][0-9]")):
        if path.is_dir():
            snapshots[int(path.name)] = path
    return snapshots


def nearest_vtk_snapshot(index, target_time, *, tolerance=0.05):
    """Find an archived VTK snapshot at the requested physical time."""
    if not index:
        raise FileNotFoundError("no archived VTK snapshots are available")
    estimated_number = target_time / VTK_OUTPUT_INTERVAL
    candidates = sorted(
        index,
        key=lambda number: abs(number - estimated_number),
    )[:16]
    best = None
    for number in candidates:
        path = index[number]
        time = read_vtk_snapshot_time(path)
        difference = abs(time - target_time)
        if best is None or difference < best[0]:
            best = (difference, path, time)
        if difference <= tolerance:
            return path, time
    difference, _, _ = best
    raise ValueError(
        f"nearest VTK snapshot to t={target_time:g} is offset by "
        f"{difference:g}"
    )


def _read_vtk_density_block(path):
    """Read geometry and the leading density scalar from one rank VTK."""
    geometry = {}
    with Path(path).open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"density scalar not found in {path}")
            words = line.split()
            if words[:1] == [b"DIMENSIONS"]:
                geometry["shape"] = np.asarray(words[1:4], dtype=int) - 1
            elif words[:1] == [b"ORIGIN"]:
                geometry["left_edge"] = np.asarray(words[1:4], dtype=float)
            elif words[:1] == [b"SPACING"]:
                geometry["spacing"] = np.asarray(words[1:4], dtype=float)
            elif words[:1] == [b"CELL_DATA"]:
                geometry["cell_count"] = int(words[1])
            elif words[:1] == [b"SCALARS"]:
                if words[1:3] != [b"density", b"float"]:
                    raise ValueError(
                        f"first VTK scalar in {path} is not float density"
                    )
                required = {"shape", "left_edge", "spacing", "cell_count"}
                missing = required.difference(geometry)
                if missing:
                    raise ValueError(
                        f"incomplete VTK geometry in {path}: {sorted(missing)}"
                    )
                if not stream.readline().startswith(b"LOOKUP_TABLE"):
                    raise ValueError(f"missing density lookup table in {path}")
                count = geometry["cell_count"]
                raw = stream.read(4 * count)
                if len(raw) != 4 * count:
                    raise ValueError(f"truncated density field in {path}")
                shape = tuple(geometry["shape"][::-1])
                density = np.frombuffer(raw, dtype=">f4").reshape(shape)
                return geometry, density


def xz_surface_density_from_vtk(snapshot):
    """Reconstruct a y-integrated XZ column from flat rank VTK files."""
    blocks = [
        _read_vtk_density_block(path)
        for path in sorted(Path(snapshot).glob("*.vtk"))
    ]
    if not blocks:
        raise FileNotFoundError(f"no VTK rank files under {snapshot}")

    spacing = blocks[0][0]["spacing"]
    if np.any(spacing <= 0.0):
        raise ValueError(f"non-positive VTK spacing under {snapshot}")
    for geometry, _ in blocks[1:]:
        if not np.allclose(geometry["spacing"], spacing):
            raise ValueError(f"inconsistent VTK spacing under {snapshot}")

    left_edge = np.min(
        [geometry["left_edge"] for geometry, _ in blocks], axis=0
    )
    right_edge = np.max(
        [
            geometry["left_edge"]
            + geometry["shape"] * geometry["spacing"]
            for geometry, _ in blocks
        ],
        axis=0,
    )
    domain_shape = np.rint((right_edge - left_edge) / spacing).astype(int)
    column = np.zeros((domain_shape[2], domain_shape[0]), dtype=float)
    coverage = np.zeros_like(column, dtype=int)
    for geometry, density in blocks:
        offset_float = (geometry["left_edge"] - left_edge) / spacing
        offset = np.rint(offset_float).astype(int)
        if not np.allclose(offset, offset_float):
            raise ValueError(f"misaligned VTK block under {snapshot}")
        nz, ny, nx = density.shape
        zslice = slice(offset[2], offset[2] + nz)
        xslice = slice(offset[0], offset[0] + nx)
        column[zslice, xslice] += density.sum(axis=1, dtype=float)
        coverage[zslice, xslice] += ny
    if not np.all(coverage == domain_shape[1]):
        raise ValueError(f"incomplete or overlapping VTK blocks under {snapshot}")

    column *= spacing[1] * star_particle_units()["mass_msun"]
    extent = (
        float(left_edge[0]),
        float(right_edge[0]),
        float(left_edge[2]),
        float(right_edge[2]),
    )
    return column, extent


def xz_surface_density_map(pdf):
    """Convert a y-integrated x1-x3 PDF2D nH weight to Msun/pc^2."""
    if pdf.get("id") != DEFAULT_PDF_ID:
        raise ValueError(
            f"expected PDF2D id {DEFAULT_PDF_ID!r}, got {pdf.get('id')!r}"
        )
    if (pdf.get("binx_name"), pdf.get("biny_name")) != ("x1", "x3"):
        raise ValueError(
            "XZ projection must use linear x1 and x3 coordinate axes"
        )
    if pdf.get("binx_log") or pdf.get("biny_log"):
        raise ValueError("XZ projection coordinate axes must be linear")
    if "nH" not in pdf.get("weights", {}):
        raise KeyError("XZ PDF2D projection is missing the nH weight")

    dx = float(pdf["x_spacing"])
    dz = float(pdf["y_spacing"])
    projected_area = dx * dz
    if not np.isfinite(projected_area) or projected_area <= 0.0:
        raise ValueError("XZ PDF2D cell area must be finite and positive")

    data = (
        np.asarray(pdf["weights"]["nH"], dtype=float)
        / projected_area
        * star_particle_units()["mass_msun"]
    )
    expected_shape = (int(pdf["Nbiny"]), int(pdf["Nbinx"]))
    if data.shape != expected_shape:
        raise ValueError(
            f"XZ nH array has shape {data.shape}, expected {expected_shape}"
        )
    extent = (
        float(pdf["x_edges"][0]),
        float(pdf["x_edges"][-1]),
        float(pdf["y_edges"][0]),
        float(pdf["y_edges"][-1]),
    )
    return data, extent


def load_xz_surface_density_maps(paths):
    """Load aligned XZ maps with one shared physical extent."""
    maps = []
    common_extent = None
    for path in paths:
        pdf = read_pdf2d(path, fields="nH")
        data, extent = xz_surface_density_map(pdf)
        if common_extent is None:
            common_extent = extent
        elif not np.allclose(extent, common_extent):
            raise ValueError(
                f"XZ projection extent differs at {path}: {extent} "
                f"versus {common_extent}"
            )
        maps.append(data)
    return maps, common_extent


def load_xz_maps_with_vtk_fallback(
    paths,
    vtk_indices,
    target_time,
    *,
    reference_extent=None,
    tolerance=0.05,
):
    """Load native XZ maps, rebuilding malformed products from VTK."""
    if len(paths) != len(vtk_indices):
        raise ValueError("XZ paths and VTK indices must have equal lengths")
    maps = []
    sources = []
    common_extent = reference_extent
    for path, vtk_index in zip(paths, vtk_indices):
        replaced = ""
        reason = ""
        try:
            pdf = read_pdf2d(path, fields="nH")
            data, extent = xz_surface_density_map(pdf)
            if not np.any(np.isfinite(data) & (data > 0.0)):
                raise ValueError("native XZ map has no positive finite data")
            if common_extent is not None and not np.allclose(extent, common_extent):
                raise ValueError(
                    f"native XZ extent {extent} differs from {common_extent}"
                )
            source = "pdf2d"
            source_path = path
            source_time = float(pdf["time"])
        except (OSError, ValueError, KeyError, EOFError) as error:
            snapshot, source_time = nearest_vtk_snapshot(
                vtk_index, target_time, tolerance=tolerance
            )
            data, extent = xz_surface_density_from_vtk(snapshot)
            source = "vtk-reconstructed"
            source_path = snapshot
            replaced = str(path)
            reason = str(error)
        if common_extent is None:
            common_extent = extent
        elif not np.allclose(extent, common_extent):
            raise ValueError(
                f"XZ extent at {source_path} differs from {common_extent}"
            )
        if maps and data.shape != maps[0].shape:
            raise ValueError(f"XZ shape differs at {source_path}")
        maps.append(data)
        sources.append({
            "source": source,
            "path": str(source_path),
            "source_time": float(source_time),
            "replaced_pdf2d": replaced,
            "reason": reason,
        })
    return maps, common_extent, sources


def create_xz_grid_figure(
    ranked,
    maps,
    extent,
    time,
    *,
    nrows=4,
    ncols=8,
    vmin=DEFAULT_VMIN,
    vmax=DEFAULT_VMAX,
    cmap="managua_r",
):
    """Create a physical-aspect 4x8 XZ surface-density grid."""
    if len(ranked) != len(maps):
        raise ValueError("ranked models and XZ maps must have equal lengths")
    if len(ranked) > nrows * ncols:
        raise ValueError(
            f"{len(ranked)} models do not fit in a {nrows}x{ncols} grid"
        )
    if vmin <= 0.0 or vmax <= vmin:
        raise ValueError("require 0 < vmin < vmax")

    xmin, xmax, zmin, zmax = map(float, extent)
    width = xmax - xmin
    height = zmax - zmin
    if width <= 0.0 or height <= 0.0:
        raise ValueError("XZ extent must be increasing")
    panel_width = 1.7
    panel_height = panel_width * height / width
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows + 0.8),
        squeeze=False,
    )
    norm = LogNorm(vmin=vmin, vmax=vmax)
    images = []
    text_effect = [
        path_effects.withStroke(linewidth=1.5, foreground="black")
    ]
    for rank, (axis, item, data) in enumerate(
        zip(axes.flat, ranked, maps),
        start=1,
    ):
        model, mean_sfr = item
        image = axis.imshow(
            data,
            origin="lower",
            extent=extent,
            aspect="equal",
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        images.append(image)
        axis.text(
            0.03,
            0.985,
            f"{rank:02d} {short_model_name(model)}\n"
            rf"$\langle\Sigma_{{\rm SFR,10}}\rangle={mean_sfr:.2e}$",
            transform=axis.transAxes,
            color="white",
            fontsize=5.8,
            ha="left",
            va="top",
            linespacing=1.1,
            path_effects=text_effect,
        )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(zmin, zmax)
        axis.set_axis_off()
    for axis in axes.flat[len(ranked):]:
        axis.set_axis_off()

    time_text = fig.suptitle(
        rf"XZ total gas surface density (integrated along $y$): $t={time:.1f}$",
        y=0.997,
        fontsize=13,
    )
    color_axis = fig.add_axes((0.29, 0.012, 0.42, 0.011))
    colorbar = fig.colorbar(images[0], cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        r"$\Sigma_{\rm gas,y}\;[M_\odot\,{\rm pc}^{-2}]$",
        labelpad=1,
    )
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(labelsize=8, pad=1)
    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.04,
        top=0.982,
        wspace=0.025,
        hspace=0.018,
    )
    return fig, images, time_text


def update_xz_grid_figure(images, time_text, maps, time):
    """Update a reusable XZ grid for another aligned time."""
    if len(images) != len(maps):
        raise ValueError("images and XZ maps must have equal lengths")
    for image, data in zip(images, maps):
        image.set_data(data)
    time_text.set_text(
        rf"XZ total gas surface density (integrated along $y$): $t={time:.1f}$"
    )


def make_xz_surface_density_movie(output_dir, movie_path, fps=10.0):
    """Encode the rendered XZ suite frames."""
    return make_grid_movie(
        output_dir,
        DEFAULT_FRAME_STEM,
        movie_path,
        fps=fps,
    )


def write_xz_sources(records, path):
    """Atomically write panel-level XZ source provenance."""
    fieldnames = (
        "target_time",
        "model",
        "source",
        "source_time",
        "path",
        "replaced_pdf2d",
        "reason",
    )
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        ordered = sorted(
            records,
            key=lambda record: float(record["target_time"]),
        )
        writer.writerows(ordered)
    temporary.replace(path)


def render_xz_surface_density(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    pdf_id=DEFAULT_PDF_ID,
    output_dir=None,
    start=0,
    stop=600,
    stride=100,
    sfr_bounds=DEFAULT_SFR_RANGE,
    history_samples=10000,
    nrows=4,
    ncols=8,
    vmin=DEFAULT_VMIN,
    vmax=DEFAULT_VMAX,
    cmap="managua_r",
    dpi=150,
    time_tolerance=0.05,
    search_radius=32,
    overwrite=False,
    movie=False,
    fps=10.0,
):
    """Render SFR-ranked XZ suite figures and optionally encode an MP4."""
    suite = Path(suite).expanduser()
    output_dir = (
        Path(output_dir)
        if output_dir
        else suite / DEFAULT_OUTPUT_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    models = discover_xz_models(suite, model_glob, pdf_id)
    ranked = rank_models_by_sfr(
        models,
        bounds=sfr_bounds,
        max_rows=history_samples,
    )
    if len(ranked) != nrows * ncols:
        raise ValueError(
            f"expected exactly {nrows * ncols} models for a "
            f"{nrows}x{ncols} grid, found {len(ranked)}"
        )
    write_model_order(ranked, output_dir / "model_order.csv", sfr_bounds)
    indices = [
        xz_number_index(model, pdf_id)
        for model, _ in ranked
    ]
    vtk_indices = [
        vtk_snapshot_index(model)
        for model, _ in ranked
    ]
    guess_offsets = [0] * len(ranked)

    fig = images = time_text = None
    reference_extent = None
    written = []
    provenance = []
    source_table = output_dir / "xz_sources.csv"
    target_times = list(range(start, stop + 1, stride))
    regenerated = {
        target_time
        for target_time in target_times
        if overwrite
        or not (
            output_dir / f"{DEFAULT_FRAME_STEM}.{target_time:04d}.png"
        ).exists()
    }
    if source_table.exists():
        with source_table.open(newline="") as stream:
            provenance = [
                record
                for record in csv.DictReader(stream)
                if int(float(record["target_time"])) not in regenerated
            ]
    try:
        for target_time in target_times:
            output = output_dir / f"{DEFAULT_FRAME_STEM}.{target_time:04d}.png"
            if output.exists() and not overwrite:
                print(f"Skipping existing {output}", flush=True)
                continue

            paths, times, guess_offsets = nearest_xz_paths(
                indices,
                target_time,
                guess_offsets,
                tolerance=time_tolerance,
                search_radius=search_radius,
            )
            if np.ptp(times) > time_tolerance:
                detail = ", ".join(f"{value:g}" for value in times)
                raise ValueError(
                    f"target t={target_time:g} is not time-aligned: {detail}"
                )
            maps, extent, frame_sources = load_xz_maps_with_vtk_fallback(
                paths,
                vtk_indices,
                target_time,
                reference_extent=reference_extent,
                tolerance=time_tolerance,
            )
            for (model, _), source in zip(ranked, frame_sources):
                provenance.append({
                    "target_time": target_time,
                    "model": model.name,
                    **source,
                })
            time = float(np.mean(times))
            if fig is None:
                reference_extent = extent
                fig, images, time_text = create_xz_grid_figure(
                    ranked,
                    maps,
                    extent,
                    time,
                    nrows=nrows,
                    ncols=ncols,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
            else:
                if not np.allclose(extent, reference_extent):
                    raise ValueError("XZ extent changed between suite frames")
                update_xz_grid_figure(images, time_text, maps, time)

            fig.savefig(output, dpi=dpi, facecolor="white")
            written.append(output)
            write_xz_sources(provenance, source_table)
            fallback_count = sum(
                source["source"] == "vtk-reconstructed"
                for source in frame_sources
            )
            print(f"  VTK fallback panels: {fallback_count}", flush=True)
            print(f"Wrote {output}", flush=True)
    finally:
        if fig is not None:
            plt.close(fig)

    if movie:
        make_xz_surface_density_movie(
            output_dir,
            output_dir / "surface_density_xz_evolution.mp4",
            fps=fps,
        )
    return written, ranked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    parser.add_argument("--pdf-id", default=DEFAULT_PDF_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=600)
    parser.add_argument(
        "--stride",
        type=int,
        default=100,
        help="spacing between target times; use 1 for full-cadence evolution",
    )
    parser.add_argument("--sfr-start", type=float, default=DEFAULT_SFR_RANGE[0])
    parser.add_argument("--sfr-stop", type=float, default=DEFAULT_SFR_RANGE[1])
    parser.add_argument("--history-samples", type=int, default=10000)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--vmin", type=float, default=DEFAULT_VMIN)
    parser.add_argument("--vmax", type=float, default=DEFAULT_VMAX)
    parser.add_argument("--cmap", default="managua_r")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--time-tolerance", type=float, default=0.05)
    parser.add_argument("--search-radius", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.start < 0 or args.stop < args.start:
        parser.error("require 0 <= --start <= --stop")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.sfr_stop <= args.sfr_start:
        parser.error("--sfr-stop must be greater than --sfr-start")
    if args.history_samples < 2:
        parser.error("--history-samples must be at least 2")
    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive")
    if args.vmin <= 0.0 or args.vmax <= args.vmin:
        parser.error("require 0 < --vmin < --vmax")
    if args.dpi <= 0 or args.fps <= 0.0:
        parser.error("--dpi and --fps must be positive")
    if args.time_tolerance <= 0.0:
        parser.error("--time-tolerance must be positive")
    if args.search_radius < 0:
        parser.error("--search-radius must be non-negative")

    render_xz_surface_density(
        args.suite,
        model_glob=args.model_glob,
        pdf_id=args.pdf_id,
        output_dir=args.output_dir,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        sfr_bounds=(args.sfr_start, args.sfr_stop),
        history_samples=args.history_samples,
        nrows=args.rows,
        ncols=args.cols,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        dpi=args.dpi,
        time_tolerance=args.time_tolerance,
        search_radius=args.search_radius,
        overwrite=args.overwrite,
        movie=args.movie,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
