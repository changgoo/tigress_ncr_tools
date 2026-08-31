#!/usr/bin/env python3
"""Render a SFR-ranked grid of surface-density maps for an NCR suite."""

import argparse
import csv
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pathena.hst_reader import read_hst
from pathena.proj2d_reader import read_proj2d, read_proj2d_metadata
from pathena.units import star_particle_units

from .plot_suite_projections import make_projection_movie


DEFAULT_SUITE = Path("/tigress/changgoo/anvil/TIGRESS-NCR-suite")
DEFAULT_MODEL_GLOB = "R8_8pc_NCR_row????"
DEFAULT_OUTPUT_NAME = "surface_density_evolution_theta0"
DEFAULT_PHASE_OUTPUT_NAME = "hydrogen_phase_evolution_theta0"
DEFAULT_SFR_RANGE = (200.0, 600.0)
DEFAULT_PHASE_SCALE = 25.0
DEFAULT_ASINH_Q = 10.0
DEFAULT_HI_GREEN_SCALE = 0.65


def discover_evolution_models(suite, model_glob=DEFAULT_MODEL_GLOB, proj_id="theta0"):
    """Return model directories that have both history and projection output."""
    suite = Path(suite)
    models = []
    for model in sorted(suite.glob(model_glob)):
        histories = [
            path for path in sorted((model / "hst").glob("*.hst"))
            if ".phase" not in path.name and ".whole" not in path.name
        ]
        if model.is_dir() and histories and (model / "proj2d" / proj_id).is_dir():
            models.append(model)
    if not models:
        raise FileNotFoundError(
            f"no models under {suite} match {model_glob!r} with hst and proj2d/{proj_id}"
        )
    return models


def time_average(time, values, bounds=DEFAULT_SFR_RANGE):
    """Return the trapezoidal time average within inclusive bounds."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.shape != values.shape:
        raise ValueError("time and values must have the same shape")
    if len(bounds) != 2 or bounds[1] <= bounds[0]:
        raise ValueError("average bounds must be increasing")
    finite = np.isfinite(time) & np.isfinite(values)
    time = time[finite]
    values = values[finite]
    if len(time) < 2:
        raise ValueError("time average requires at least two finite samples")
    order = np.argsort(time, kind="stable")
    time = time[order]
    values = values[order]
    # Keep the last value at duplicate times, matching restart handling.
    keep = np.r_[time[1:] > time[:-1], True]
    time = time[keep]
    values = values[keep]
    lower, upper = map(float, bounds)
    if time[0] > lower or time[-1] < upper:
        raise ValueError(
            f"history covers [{time[0]:g}, {time[-1]:g}], not [{lower:g}, {upper:g}]"
        )
    interior = (time > lower) & (time < upper)
    sample_time = np.r_[lower, time[interior], upper]
    sample_values = np.interp(sample_time, time, values)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(sample_values, sample_time) / (upper - lower))


def model_mean_sfr(model, bounds=DEFAULT_SFR_RANGE, max_rows=10000):
    """Read one model and return its time-averaged sfr10."""
    files = [
        path for path in sorted((Path(model) / "hst").glob("*.hst"))
        if ".phase" not in path.name and ".whole" not in path.name
    ]
    if not files:
        raise FileNotFoundError(f"no primary history file under {Path(model) / 'hst'}")
    history = read_hst(files[0], max_rows=max_rows)
    return time_average(history["time"], history["sfr10"], bounds)


def rank_models_by_sfr(models, bounds=DEFAULT_SFR_RANGE, max_rows=10000):
    """Return (model, mean_sfr) pairs from highest to lowest SFR."""
    ranked = [
        (Path(model), model_mean_sfr(model, bounds=bounds, max_rows=max_rows))
        for model in models
    ]
    return sorted(ranked, key=lambda item: (-item[1], item[0].name))


def projection_path(model, output_number, proj_id="theta0"):
    """Resolve a projection by output number without assuming the problem id."""
    directory = Path(model) / "proj2d" / proj_id
    matches = sorted(directory.glob(f"*.{output_number:04d}.{proj_id}.proj2d"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one output {output_number:04d} under {directory}, found {len(matches)}"
        )
    return matches[0]


def projection_number_index(model, proj_id="theta0"):
    """Return output-number to path mappings without reading map files."""
    directory = Path(model) / "proj2d" / proj_id
    index = {}
    for path in directory.glob("*.proj2d"):
        try:
            number = int(path.name.split(".")[-3])
        except (IndexError, ValueError):
            continue
        if number in index:
            raise ValueError(f"duplicate output number {number:04d} under {directory}")
        index[number] = path
    if not index:
        raise FileNotFoundError(f"no projection files under {directory}")
    return index


def nearest_indexed_projection(
    index,
    target_time,
    guess_offset=0,
    *,
    tolerance=0.05,
    search_radius=32,
):
    """Find a local filename candidate whose stored time matches target_time."""
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
        stored_time = float(read_proj2d_metadata(path)["time"])
        offset = abs(stored_time - target_time)
        if best is None or offset < best[0]:
            best = (offset, path, stored_time, number - target_number)
        if offset <= tolerance:
            return path, stored_time, number - target_number
    if best is None:
        raise ValueError(f"no projection candidate is available near t={target_time:g}")
    raise ValueError(
        f"nearest projection to t={target_time:g} is offset by {best[0]:g}"
    )


def nearest_projection_paths(
    indices,
    target_time,
    guess_offsets=None,
    *,
    tolerance=0.05,
    search_radius=32,
):
    """Select one projection per model at a common stored physical time."""
    if guess_offsets is None:
        guess_offsets = [0] * len(indices)
    if len(guess_offsets) != len(indices):
        raise ValueError("guess offsets and projection indices must have equal lengths")
    paths = []
    times = []
    offsets = []
    for index, guess in zip(indices, guess_offsets):
        path, stored_time, offset = nearest_indexed_projection(
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


def load_surface_density_maps(paths):
    """Load nH projections and convert them to Msun/pc^2."""
    surface_density_unit = star_particle_units()["mass_msun"]
    maps = []
    for path in paths:
        frame = read_proj2d(path, fields="nH")
        maps.append(np.asarray(frame["fields"]["nH"]) * surface_density_unit)
    return maps


def hydrogen_phase_rgb(
    nH,
    nHI,
    nH2,
    *,
    scale=DEFAULT_PHASE_SCALE,
    asinh_q=DEFAULT_ASINH_Q,
    hi_green_scale=DEFAULT_HI_GREEN_SCALE,
):
    """Return a fixed-stretch RGB map of H2 (red), HI (green), and HII (blue).

    All three channels count hydrogen nuclei and use the same surface-density
    stretch, so both hue and brightness remain comparable between frames.
    Input arrays are line-integrated number densities in proj2d code units.
    """
    if scale <= 0:
        raise ValueError("phase scale must be positive")
    if asinh_q <= 0:
        raise ValueError("asinh q must be positive")
    if hi_green_scale < 0:
        raise ValueError("HI green scale must be non-negative")
    nH = np.asarray(nH, dtype=float)
    nHI = np.asarray(nHI, dtype=float)
    nH2 = np.asarray(nH2, dtype=float)
    if nH.shape != nHI.shape or nH.shape != nH2.shape:
        raise ValueError("nH, nHI, and nH2 must have matching shapes")

    surface_density_unit = star_particle_units()["mass_msun"]
    nHII = np.clip(nH - nHI - 2.0 * nH2, 0.0, None)
    species = np.stack((2.0 * nH2, nHI, nHII), axis=-1)
    species = np.clip(species * surface_density_unit, 0.0, None)
    rgb = np.arcsinh(asinh_q * species / scale) / np.arcsinh(asinh_q)
    rgb[..., 1] *= hi_green_scale
    return np.clip(rgb, 0.0, 1.0)


def load_hydrogen_phase_maps(
    paths,
    *,
    scale=DEFAULT_PHASE_SCALE,
    asinh_q=DEFAULT_ASINH_Q,
    hi_green_scale=DEFAULT_HI_GREEN_SCALE,
):
    """Load proj2d fields and form fixed-stretch H-phase RGB maps."""
    maps = []
    for path in paths:
        fields = read_proj2d(path, fields=("nH", "nHI", "nH2"))["fields"]
        maps.append(
            hydrogen_phase_rgb(
                fields["nH"],
                fields["nHI"],
                fields["nH2"],
                scale=scale,
                asinh_q=asinh_q,
                hi_green_scale=hi_green_scale,
            )
        )
    return maps


def short_model_name(model):
    """Use the row id where possible so 32 panel labels stay legible."""
    name = Path(model).name
    marker = "_row"
    return name[name.rfind(marker) + 1:] if marker in name else name


def create_grid_figure(
    ranked,
    maps,
    time,
    *,
    nrows=4,
    ncols=8,
    vmin=0.1,
    vmax=100.0,
    cmap="managua_r",
):
    """Create the shared-scale grid and return figure artists for reuse."""
    if len(ranked) != len(maps):
        raise ValueError("ranked models and maps must have equal lengths")
    if len(ranked) > nrows * ncols:
        raise ValueError(f"{len(ranked)} models do not fit in a {nrows}x{ncols} grid")
    if vmin <= 0 or vmax <= vmin:
        raise ValueError("require 0 < vmin < vmax")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.25 * ncols, 2.25 * nrows + 0.45),
        squeeze=False,
    )
    norm = LogNorm(vmin=vmin, vmax=vmax)
    images = []
    text_effect = [path_effects.withStroke(linewidth=1.8, foreground="black")]
    for rank, (axis, item, data) in enumerate(
        zip(axes.flat, ranked, maps), start=1
    ):
        model, mean_sfr = item
        image = axis.imshow(
            data,
            origin="lower",
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        images.append(image)
        axis.text(
            0.025,
            0.975,
            f"{rank:02d} {short_model_name(model)}\n"
            rf"$\langle\Sigma_{{\rm SFR,10}}\rangle={mean_sfr:.2e}$",
            transform=axis.transAxes,
            color="white",
            fontsize=7.2,
            ha="left",
            va="top",
            linespacing=1.15,
            path_effects=text_effect,
        )
        axis.set_axis_off()
    for axis in axes.flat[len(ranked):]:
        axis.set_axis_off()

    time_text = fig.suptitle(rf"Total gas surface density: $t={time:.1f}$", y=0.995)
    color_axis = fig.add_axes((0.32, 0.022, 0.36, 0.018))
    colorbar = fig.colorbar(images[0], cax=color_axis, orientation="horizontal")
    colorbar.set_label(r"$\Sigma_{\rm gas}\;[M_\odot\,{\rm pc}^{-2}]$", labelpad=1)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.ax.tick_params(labelsize=8, pad=1)
    fig.subplots_adjust(
        left=0.006,
        right=0.994,
        bottom=0.085,
        top=0.96,
        wspace=0.018,
        hspace=0.018,
    )
    return fig, images, time_text


def update_grid_figure(images, time_text, maps, time):
    """Update a reusable grid figure for the next output."""
    if len(images) != len(maps):
        raise ValueError("images and maps must have equal lengths")
    for image, data in zip(images, maps):
        image.set_data(data)
    time_text.set_text(rf"Total gas surface density: $t={time:.1f}$")


def create_phase_grid_figure(
    ranked,
    maps,
    time,
    *,
    nrows=4,
    ncols=8,
    scale=DEFAULT_PHASE_SCALE,
    asinh_q=DEFAULT_ASINH_Q,
):
    """Create the ranked H2/HI/HII pseudocolor grid."""
    if len(ranked) != len(maps):
        raise ValueError("ranked models and maps must have equal lengths")
    if len(ranked) > nrows * ncols:
        raise ValueError(f"{len(ranked)} models do not fit in a {nrows}x{ncols} grid")

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.25 * ncols, 2.25 * nrows + 0.45),
        squeeze=False,
    )
    images = []
    text_effect = [path_effects.withStroke(linewidth=1.8, foreground="black")]
    for rank, (axis, item, data) in enumerate(
        zip(axes.flat, ranked, maps), start=1
    ):
        model, mean_sfr = item
        image = axis.imshow(data, origin="lower", interpolation="nearest")
        images.append(image)
        axis.text(
            0.025,
            0.975,
            f"{rank:02d} {short_model_name(model)}\n"
            rf"$\langle\Sigma_{{\rm SFR,10}}\rangle={mean_sfr:.2e}$",
            transform=axis.transAxes,
            color="white",
            fontsize=7.2,
            ha="left",
            va="top",
            linespacing=1.15,
            path_effects=text_effect,
        )
        axis.set_axis_off()
    for axis in axes.flat[len(ranked):]:
        axis.set_axis_off()

    time_text = fig.suptitle(rf"Hydrogen phases: $t={time:.1f}$", y=0.995)
    legend_y = 0.027
    fig.text(0.365, legend_y, r"$2\mathrm{H}_2$", color="#ff3030", ha="right")
    fig.text(0.405, legend_y, "/", color="0.3", ha="center")
    fig.text(0.445, legend_y, r"$\mathrm{H\,I}$", color="#00a800", ha="center")
    fig.text(0.485, legend_y, "/", color="0.3", ha="center")
    fig.text(0.525, legend_y, r"$\mathrm{H\,II}$", color="#2864ff", ha="center")
    fig.text(
        0.555,
        legend_y,
        rf"(R/G/B; asinh $Q={asinh_q:g}$, scale={scale:g} "
        rf"$M_\odot\,\mathrm{{pc}}^{{-2}}$)",
        color="0.25",
        ha="left",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.006,
        right=0.994,
        bottom=0.065,
        top=0.96,
        wspace=0.018,
        hspace=0.018,
    )
    return fig, images, time_text


def update_phase_grid_figure(images, time_text, maps, time):
    """Update a reusable H-phase pseudocolor grid."""
    if len(images) != len(maps):
        raise ValueError("images and maps must have equal lengths")
    for image, data in zip(images, maps):
        image.set_data(data)
    time_text.set_text(rf"Hydrogen phases: $t={time:.1f}$")


def write_model_order(ranked, path, bounds=DEFAULT_SFR_RANGE):
    """Write the exact panel ordering and ranking statistic."""
    path = Path(path)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("rank", "model", "mean_sfr10", "time_start", "time_stop"))
        for rank, (model, mean_sfr) in enumerate(ranked, start=1):
            writer.writerow((rank, model.name, f"{mean_sfr:.12g}", bounds[0], bounds[1]))


def make_grid_movie(output_dir, frame_stem, movie_path, fps=30.0):
    """Encode collective grid frames using the projection movie helper."""
    output_dir = Path(output_dir)
    alias_id = "grid"
    aliases = []
    try:
        pattern = f"{frame_stem}.[0-9][0-9][0-9][0-9].png"
        for source in sorted(output_dir.glob(pattern)):
            alias = output_dir / f"projection.{alias_id}.{source.stem.rsplit('.', 1)[-1]}.png"
            if not alias.exists():
                alias.hardlink_to(source)
                aliases.append(alias)
        if not aliases and not list(output_dir.glob(f"projection.{alias_id}.*.png")):
            raise FileNotFoundError(f"no {frame_stem} frames under {output_dir}")
        return make_projection_movie(
            output_dir,
            alias_id,
            movie_path=movie_path,
            fps_in=fps,
            fps_out=fps,
        )
    finally:
        for alias in aliases:
            alias.unlink(missing_ok=True)


def make_surface_density_movie(output_dir, movie_path, fps=30.0):
    """Encode collective surface-density frames."""
    return make_grid_movie(output_dir, "surface_density", movie_path, fps)


def make_hydrogen_phase_movie(output_dir, movie_path, fps=30.0):
    """Encode collective H-phase pseudocolor frames."""
    return make_grid_movie(output_dir, "hydrogen_phases", movie_path, fps)


def render_surface_density_evolution(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    proj_id="theta0",
    output_dir=None,
    start=0,
    stop=600,
    stride=1,
    sfr_bounds=DEFAULT_SFR_RANGE,
    history_samples=10000,
    nrows=4,
    ncols=8,
    vmin=0.1,
    vmax=100.0,
    cmap="managua_r",
    dpi=150,
    overwrite=False,
    movie=False,
    fps=30.0,
):
    """Render ranked collective frames and optionally encode an MP4."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_evolution_models(suite, model_glob, proj_id)
    ranked = rank_models_by_sfr(models, bounds=sfr_bounds, max_rows=history_samples)
    if len(ranked) != nrows * ncols:
        raise ValueError(
            f"expected exactly {nrows * ncols} models for a {nrows}x{ncols} grid, "
            f"found {len(ranked)}"
        )
    write_model_order(ranked, output_dir / "model_order.csv", sfr_bounds)
    projection_indices = [
        projection_number_index(model, proj_id)
        for model, _ in ranked
    ]
    guess_offsets = [0] * len(ranked)

    fig = images = time_text = None
    written = []
    try:
        for output_number in range(start, stop + 1, stride):
            output = output_dir / f"surface_density.{output_number:04d}.png"
            if output.exists() and not overwrite:
                print(f"Skipping existing {output}", flush=True)
                continue
            paths, times, guess_offsets = nearest_projection_paths(
                projection_indices, output_number, guess_offsets
            )
            maps = load_surface_density_maps(paths)
            if np.ptp(times) > 0.05:
                detail = ", ".join(f"{value:g}" for value in times)
                raise ValueError(f"output {output_number:04d} is not time-aligned: {detail}")
            time = float(np.mean(times))
            if fig is None:
                fig, images, time_text = create_grid_figure(
                    ranked,
                    maps,
                    time,
                    nrows=nrows,
                    ncols=ncols,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
            else:
                update_grid_figure(images, time_text, maps, time)
            fig.savefig(output, dpi=dpi, facecolor="white")
            written.append(output)
            print(f"Wrote {output}", flush=True)
    finally:
        if fig is not None:
            plt.close(fig)

    if movie:
        make_surface_density_movie(
            output_dir, output_dir / "surface_density_evolution.mp4", fps=fps
        )
    return written, ranked


def render_hydrogen_phase_evolution(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    proj_id="theta0",
    output_dir=None,
    start=0,
    stop=600,
    stride=1,
    sfr_bounds=DEFAULT_SFR_RANGE,
    history_samples=10000,
    nrows=4,
    ncols=8,
    scale=DEFAULT_PHASE_SCALE,
    asinh_q=DEFAULT_ASINH_Q,
    hi_green_scale=DEFAULT_HI_GREEN_SCALE,
    dpi=150,
    overwrite=False,
    movie=False,
    fps=30.0,
):
    """Render ranked H-phase pseudocolor frames and optionally encode an MP4."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_PHASE_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_evolution_models(suite, model_glob, proj_id)
    ranked = rank_models_by_sfr(models, bounds=sfr_bounds, max_rows=history_samples)
    if len(ranked) != nrows * ncols:
        raise ValueError(
            f"expected exactly {nrows * ncols} models for a {nrows}x{ncols} grid, "
            f"found {len(ranked)}"
        )
    write_model_order(ranked, output_dir / "model_order.csv", sfr_bounds)
    projection_indices = [
        projection_number_index(model, proj_id)
        for model, _ in ranked
    ]
    guess_offsets = [0] * len(ranked)

    fig = images = time_text = None
    written = []
    try:
        for output_number in range(start, stop + 1, stride):
            output = output_dir / f"hydrogen_phases.{output_number:04d}.png"
            if output.exists() and not overwrite:
                print(f"Skipping existing {output}", flush=True)
                continue
            paths, times, guess_offsets = nearest_projection_paths(
                projection_indices, output_number, guess_offsets
            )
            maps = load_hydrogen_phase_maps(
                paths,
                scale=scale,
                asinh_q=asinh_q,
                hi_green_scale=hi_green_scale,
            )
            if np.ptp(times) > 0.05:
                detail = ", ".join(f"{value:g}" for value in times)
                raise ValueError(f"output {output_number:04d} is not time-aligned: {detail}")
            time = float(np.mean(times))
            if fig is None:
                fig, images, time_text = create_phase_grid_figure(
                    ranked,
                    maps,
                    time,
                    nrows=nrows,
                    ncols=ncols,
                    scale=scale,
                    asinh_q=asinh_q,
                )
            else:
                update_phase_grid_figure(images, time_text, maps, time)
            fig.savefig(output, dpi=dpi, facecolor="white")
            written.append(output)
            print(f"Wrote {output}", flush=True)
    finally:
        if fig is not None:
            plt.close(fig)

    if movie:
        make_hydrogen_phase_movie(
            output_dir, output_dir / "hydrogen_phase_evolution.mp4", fps=fps
        )
    return written, ranked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    parser.add_argument("--projection", default="theta0", help="projection id")
    parser.add_argument(
        "--map",
        choices=("surface-density", "hydrogen-phases"),
        default="surface-density",
        help="collective map to render",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=600, help="inclusive last output")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sfr-start", type=float, default=DEFAULT_SFR_RANGE[0])
    parser.add_argument("--sfr-stop", type=float, default=DEFAULT_SFR_RANGE[1])
    parser.add_argument("--history-samples", type=int, default=10000)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--vmin", type=float, default=0.1)
    parser.add_argument("--vmax", type=float, default=100.0)
    parser.add_argument("--cmap", default="managua_r")
    parser.add_argument("--phase-scale", type=float, default=DEFAULT_PHASE_SCALE)
    parser.add_argument("--asinh-q", type=float, default=DEFAULT_ASINH_Q)
    parser.add_argument("--hi-green-scale", type=float, default=DEFAULT_HI_GREEN_SCALE)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
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
    if args.vmin <= 0 or args.vmax <= args.vmin:
        parser.error("require 0 < --vmin < --vmax")
    if args.phase_scale <= 0 or args.asinh_q <= 0:
        parser.error("--phase-scale and --asinh-q must be positive")
    if args.hi_green_scale < 0:
        parser.error("--hi-green-scale must be non-negative")
    if args.dpi <= 0 or args.fps <= 0:
        parser.error("--dpi and --fps must be positive")
    common = dict(
        model_glob=args.model_glob,
        proj_id=args.projection,
        output_dir=args.output_dir,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        sfr_bounds=(args.sfr_start, args.sfr_stop),
        history_samples=args.history_samples,
        nrows=args.rows,
        ncols=args.cols,
        dpi=args.dpi,
        overwrite=args.overwrite,
        movie=args.movie,
        fps=args.fps,
    )
    if args.map == "hydrogen-phases":
        render_hydrogen_phase_evolution(
            args.suite,
            scale=args.phase_scale,
            asinh_q=args.asinh_q,
            hi_green_scale=args.hi_green_scale,
            **common,
        )
    else:
        render_surface_density_evolution(
            args.suite,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            **common,
        )


if __name__ == "__main__":
    main()
