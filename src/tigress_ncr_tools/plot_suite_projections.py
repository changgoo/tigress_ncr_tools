#!/usr/bin/env python3
"""Make packed gas-projection maps from the late runs in a TIGRESS suite."""

import argparse
import math
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from pathena.proj2d_reader import read_proj2d, read_proj2d_metadata
from pathena.units import star_particle_units


DEFAULT_SUITE = Path("/tigress/changgoo/nasa_athena/TIGRESS-NCR")
LXY_RE = re.compile(r"Lxy(?P<size>\d+(?:\.\d+)?)")


def projection_id(angle):
    """Return an output-directory id such as ``theta30``."""
    text = str(angle)
    if text.startswith("theta"):
        text = text[5:]
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"invalid projection angle {angle!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"projection angle must be finite, got {angle!r}")
    return f"theta{value:g}"


def model_size(model):
    """Extract Lxy in pc from a model directory name."""
    match = LXY_RE.search(Path(model).name)
    if match is None:
        raise ValueError(f"cannot infer Lxy from model name {Path(model).name!r}")
    return float(match.group("size"))


def discover_late_models(suite, proj_id):
    """Find late model directories containing the requested projection."""
    suite = Path(suite)
    models = [
        path for path in suite.glob("*_late")
        if path.is_dir() and (path / "proj2d" / proj_id).is_dir()
    ]
    if not models:
        available = sorted({
            path.name
            for model in suite.glob("*_late")
            for path in (model / "proj2d").glob("theta*")
            if path.is_dir()
        })
        detail = f"; available ids: {', '.join(available)}" if available else ""
        raise FileNotFoundError(
            f"no late runs under {suite} contain proj2d/{proj_id}{detail}"
        )
    return sorted(models, key=model_size, reverse=True)


def projection_series(model, proj_id, time_tolerance=0.01):
    """Return one model's projections ordered and deduplicated by time."""
    paths = sorted((Path(model) / "proj2d" / proj_id).glob("*.proj2d"))
    if not paths:
        raise FileNotFoundError(
            f"no .proj2d files under {Path(model) / 'proj2d' / proj_id}"
        )
    records = sorted(
        ((read_proj2d_metadata(path)["time"], path) for path in paths),
        key=lambda item: item[0],
    )

    deduplicated = []
    group = []
    for record in records:
        if group and record[0] - group[0][0] > time_tolerance:
            deduplicated.append(max(
                group,
                key=lambda item: (item[1].stat().st_mtime_ns, item[1].name),
            ))
            group = []
        group.append(record)
    if group:
        deduplicated.append(max(
            group,
            key=lambda item: (item[1].stat().st_mtime_ns, item[1].name),
        ))
    return sorted(deduplicated, key=lambda item: item[0])


def align_projection_series(series, time_tolerance=0.01):
    """Match model projections by stored time, independent of output number."""
    if not series or any(not records for records in series):
        raise ValueError("every model must have at least one projection")
    ordered = [sorted(records, key=lambda item: item[0]) for records in series]
    times = [np.asarray([record[0] for record in records]) for records in ordered]
    last_used = [-1] * len(ordered)
    aligned = []

    for reference_time, reference_path in ordered[0]:
        paths = [reference_path]
        matched_indices = [None]
        for model_index in range(1, len(ordered)):
            model_times = times[model_index]
            position = int(np.searchsorted(model_times, reference_time))
            choices = [
                index for index in (position - 1, position)
                if last_used[model_index] < index < len(model_times)
            ]
            if not choices:
                break
            nearest = min(
                choices,
                key=lambda index: abs(model_times[index] - reference_time),
            )
            if abs(model_times[nearest] - reference_time) > time_tolerance:
                break
            paths.append(ordered[model_index][nearest][1])
            matched_indices.append(nearest)
        if len(paths) != len(ordered):
            continue
        aligned.append(paths)
        for model_index, matched in enumerate(matched_indices[1:], start=1):
            last_used[model_index] = matched

    if not aligned:
        raise ValueError(
            f"no projection times match across all models within {time_tolerance:g}"
        )
    return aligned


def packed_layout(widths, gap=16.0):
    """Pack the largest square left and smaller squares up its right edge.

    ``widths`` must be largest-first. The return value is
    ``(extents, xlim, ylim)``, where each extent is suitable for ``imshow``.
    """
    widths = np.asarray(widths, dtype=float)
    if widths.ndim != 1 or not len(widths) or np.any(widths <= 0):
        raise ValueError("widths must be a non-empty sequence of positive values")
    if np.any(widths[1:] > widths[:-1]):
        raise ValueError("widths must be sorted largest-first")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    largest = float(widths[0])
    half = 0.5 * largest
    extents = [(-half, half, -half, half)]
    right_left = half + gap
    y = -half
    for width in widths[1:]:
        width = float(width)
        extents.append((right_left, right_left + width, y, y + width))
        y += width + gap

    if y - gap > half + 1.0e-10:
        raise ValueError("smaller maps do not fit along the largest map")
    xmax = half if len(widths) == 1 else right_left + float(widths[1])
    return extents, (-half, xmax), (-half, half)


def _frame_width(frame):
    xwidth = float(frame["x_edges"][-1] - frame["x_edges"][0])
    ywidth = float(frame["y_edges"][-1] - frame["y_edges"][0])
    if not np.isclose(xwidth, ywidth):
        raise ValueError(
            f"projection {frame.get('path', '')} is not square: "
            f"{xwidth:g} x {ywidth:g}"
        )
    return xwidth


def plot_combined_frame(
    models,
    frames,
    *,
    gap=16.0,
    vmin=0.1,
    vmax=100.0,
    cmap="managua_r",
):
    """Plot one time-aligned set of late-run ``nH`` projections."""
    if len(models) != len(frames) or not frames:
        raise ValueError("models and frames must have the same non-zero length")
    widths = [_frame_width(frame) for frame in frames]
    extents, xlim, ylim = packed_layout(widths, gap=gap)
    surface_density_unit = star_particle_units()["mass_msun"]
    norm = LogNorm(vmin=vmin, vmax=vmax)

    canvas_ratio = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    fig, ax = plt.subplots(figsize=(6.0 * canvas_ratio, 6.0))
    image = None
    text_effect = [path_effects.withStroke(linewidth=2.0, foreground="black")]
    for model, frame, extent in zip(models, frames, extents):
        data = frame["fields"]["nH"] * surface_density_unit
        image = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.text(
            extent[0] + 0.025 * (extent[1] - extent[0]),
            extent[3] - 0.035 * (extent[3] - extent[2]),
            rf"$L_{{xy}}={model_size(model):g}\,\mathrm{{pc}}$",
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            path_effects=text_effect,
        )

    times = np.asarray([frame["time"] for frame in frames])
    theta = float(frames[0]["theta"])
    phi = float(frames[0]["phi"])
    time_myr = float(np.mean(times)) * star_particle_units()["time_myr"]
    ax.text(
        0.975,
        0.965,
        rf"$t={time_myr:.1f}\,\mathrm{{Myr}}$"
        "\n"
        rf"$\theta={theta:g}^\circ,\ \phi={phi:g}^\circ$",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    color_axis = ax.inset_axes((0.745, 0.84, 0.21, 0.025))
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        r"$\Sigma_{\rm gas}\;[M_\odot\,\mathrm{pc}^{-2}]$",
        fontsize=9,
    )
    colorbar.ax.tick_params(labelsize=8)
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def ffmpeg_encoders():
    """Return encoder names advertised by ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    encoders = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def resolve_video_codec(codec):
    """Resolve ``auto`` to an ffmpeg encoder available on this host."""
    if codec != "auto":
        return codec
    encoders = ffmpeg_encoders()
    if "libx264" in encoders:
        return "libx264"
    if "mpeg4" in encoders:
        return "mpeg4"
    raise RuntimeError("ffmpeg has neither libx264 nor mpeg4 video encoders")


def codec_quality_args(codec, crf, qscale, bitrate):
    """Return summary_movie.py-compatible ffmpeg quality arguments."""
    if bitrate:
        return ["-b:v", bitrate]
    if codec == "libx264":
        return ["-crf", str(crf), "-preset", "slow"]
    if codec == "mpeg4":
        return ["-qscale:v", str(qscale)]
    return []


def make_projection_movie(
    output_dir,
    proj_id,
    movie_path=None,
    fps_in=15,
    fps_out=15,
    overwrite=True,
    codec="auto",
    crf=18,
    qscale=2,
    bitrate=None,
):
    """Encode projection frames using summary_movie.py conventions."""
    if fps_in <= 0 or fps_out <= 0:
        raise ValueError("movie frame rates must be positive")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH")

    output_dir = Path(output_dir)
    frame_glob = str(output_dir / f"projection.{proj_id}.*.png")
    frames = sorted(output_dir.glob(f"projection.{proj_id}.*.png"))
    if not frames:
        raise FileNotFoundError(f"no frames match {frame_glob!r}")

    codec = resolve_video_codec(codec)
    if movie_path is None:
        movie_path = output_dir / f"projection.{proj_id}.mp4"
    movie_path = Path(movie_path)
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-r", str(fps_in),
        "-f", "image2",
        "-pattern_type", "glob",
        "-i", frame_glob,
        "-r", str(fps_out),
        "-pix_fmt", "yuv420p",
        "-vcodec", codec,
        *codec_quality_args(codec, crf, qscale, bitrate),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-f", "mp4",
        str(movie_path),
    ]
    print("[projection_movie]: ffmpeg command:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    print(f"[projection_movie]: wrote {movie_path}")
    return movie_path


def make_projection_maps(
    suite,
    angle,
    *,
    output_dir=None,
    start=0,
    stop=None,
    stride=1,
    gap=16.0,
    vmin=0.1,
    vmax=100.0,
    cmap="managua_r",
    dpi=200,
    time_tolerance=0.01,
    overwrite=False,
    movie=False,
    movie_path=None,
    fps_in=15,
    fps_out=15,
    movie_overwrite=True,
    codec="auto",
    crf=18,
    qscale=2,
    bitrate=None,
):
    """Render time-aligned packed maps and return the paths written."""
    suite = Path(suite).expanduser()
    proj_id = projection_id(angle)
    models = discover_late_models(suite, proj_id)
    series = [
        projection_series(model, proj_id, time_tolerance=time_tolerance)
        for model in models
    ]
    aligned = align_projection_series(series, time_tolerance=time_tolerance)
    nframes = len(aligned)
    stop = nframes if stop is None else min(stop, nframes)
    indices = range(start, stop, stride)
    if output_dir is None:
        output_dir = suite / f"projection_{proj_id}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Using {len(models)} late runs and {nframes} aligned frames "
        f"for {proj_id}"
    )
    for model, records in zip(models, series):
        print(f"  {model.name}: {len(records)} unique physical times")

    written = []
    for index in indices:
        paths = aligned[index]
        frames = [read_proj2d(path, fields="nH") for path in paths]
        times = np.asarray([frame["time"] for frame in frames])
        spread = float(np.ptp(times))
        if spread > time_tolerance:
            values = ", ".join(f"{time:g}" for time in times)
            raise ValueError(
                f"frame index {index} is not time-aligned: {values} "
                f"(spread {spread:g} > {time_tolerance:g})"
            )
        time_tag = int(round(float(np.mean(times))))
        output = output_dir / f"projection.{proj_id}.{time_tag:04d}.png"
        if output.exists() and not overwrite:
            print(f"Skipping existing {output}")
            continue
        fig = plot_combined_frame(
            models,
            frames,
            gap=gap,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        fig.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
        plt.close(fig)
        written.append(output)
        print(f"Wrote {output}")
    if movie:
        if movie_path is None:
            movie_path = suite / "movies" / f"projection_{proj_id}.mp4"
        make_projection_movie(
            output_dir,
            proj_id,
            movie_path=movie_path,
            fps_in=fps_in,
            fps_out=fps_out,
            overwrite=movie_overwrite,
            codec=codec,
            crf=crf,
            qscale=qscale,
            bitrate=bitrate,
        )
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "angle",
        help="projection theta in degrees (for example 30 or theta30)",
    )
    parser.add_argument(
        "suite",
        nargs="?",
        type=Path,
        default=DEFAULT_SUITE,
        help=f"simulation suite (default: {DEFAULT_SUITE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: SUITE/projection_thetaANGLE)",
    )
    parser.add_argument("--start", type=int, default=0, help="first late-frame index")
    parser.add_argument("--stop", type=int, help="exclusive late-frame index")
    parser.add_argument("--stride", type=int, default=1, help="frame-index stride")
    parser.add_argument("--gap", type=float, default=16.0, help="map gap in pc")
    parser.add_argument("--vmin", type=float, default=0.1)
    parser.add_argument("--vmax", type=float, default=100.0)
    parser.add_argument("--cmap", default="managua_r")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--time-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--movie",
        action="store_true",
        help="encode all projection frames in the output directory as MP4",
    )
    parser.add_argument(
        "--movie-path",
        type=Path,
        help="movie output path (default: SUITE/movies/projection_thetaANGLE.mp4)",
    )
    parser.add_argument("--fps-in", type=float, default=15, help="input frame rate")
    parser.add_argument("--fps-out", type=float, default=15, help="output frame rate")
    parser.add_argument("--codec", default="auto", help="video codec (default: auto)")
    parser.add_argument("--crf", type=int, default=18, help="libx264 CRF quality")
    parser.add_argument("--qscale", type=int, default=2, help="mpeg4 qscale quality")
    parser.add_argument("--bitrate", help="explicit video bitrate, e.g. 12M")
    parser.add_argument(
        "--no-movie-overwrite",
        action="store_true",
        help="fail rather than overwrite an existing movie",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing frames instead of skipping them",
    )
    args = parser.parse_args(argv)
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.stop is not None and args.stop < args.start:
        parser.error("--stop must be greater than or equal to --start")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.vmin <= 0 or args.vmax <= args.vmin:
        parser.error("require 0 < --vmin < --vmax")
    if args.fps_in <= 0 or args.fps_out <= 0:
        parser.error("--fps-in and --fps-out must be positive")
    if args.movie_path is not None and not args.movie:
        parser.error("--movie-path requires --movie")
    make_projection_maps(
        args.suite,
        args.angle,
        output_dir=args.output_dir,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        gap=args.gap,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        dpi=args.dpi,
        time_tolerance=args.time_tolerance,
        overwrite=args.overwrite,
        movie=args.movie,
        movie_path=args.movie_path,
        fps_in=args.fps_in,
        fps_out=args.fps_out,
        movie_overwrite=not args.no_movie_overwrite,
        codec=args.codec,
        crf=args.crf,
        qscale=args.qscale,
        bitrate=args.bitrate,
    )


if __name__ == "__main__":
    main()
