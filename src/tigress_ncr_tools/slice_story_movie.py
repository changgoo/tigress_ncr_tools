#!/usr/bin/env python3
"""Render the slice and full-volume stages of the TIGRESS-NCR story movie."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from pathena.slice_fields import derive_plane_fields
from pathena.slicevtk_reader import index_slicevtk_series, read_slicevtk
from pathena.starpar_reader import read_starpar
from pathena.vtk3d_reader import (
    discover_vtk_pieces,
    estimate_volume_bytes,
    inspect_vtk_volume,
    read_vtk_volume,
)

from .plot_suite_projections import codec_quality_args, resolve_video_codec
from .story_renderers import (
    CanvasSettings,
    blend_rgba,
    render_radiation_view,
    render_slice_view,
    render_streamline_view,
    render_volume_view,
    write_png_frame,
)
from .storyboard import (
    StoryDurations,
    build_slice_storyboard,
    write_frame_manifest,
)
from .volume_renderer import render_temperature_volume, volume_input_fields


REQUIRED_FIELD_GROUPS = (
    ("density",),
    ("pressure",),
    ("velocity",),
    ("xHI", "specific_scalar_2"),
    ("xH2", "specific_scalar_3"),
)
FINAL_SIDE_FIELD_GROUPS = (
    ("cell_centered_B",),
    ("rad_energy_density_PE",),
    ("rad_energy_density_PH",),
)

CONFIG_ARGUMENTS = {
    ("source", "problem_id"): ("problem_id", "--problem-id", str, False),
    ("source", "slice_id"): ("slice_id", "--slice-id", str, False),
    ("source", "time_tolerance"): (
        "time_tolerance", "--time-tolerance", float, False
    ),
    ("story", "fps"): ("fps", "--fps", float, False),
    ("story", "start_index"): ("start_index", "--start-index", int, False),
    ("story", "freeze_index"): ("freeze_index", "--freeze-index", int, False),
    ("story", "stop_index"): ("stop_index", "--stop-index", int, False),
    ("story", "duration_scale"): (
        "duration_scale", "--duration-scale", float, False
    ),
    ("story", "preview"): ("preview", "--preview", bool, False),
    ("canvas", "width"): ("width", "--width", int, False),
    ("canvas", "height"): ("height", "--height", int, False),
    ("volume", "enabled"): ("no_volume", "--no-volume", bool, True),
    ("volume", "stride"): ("volume_stride", "--volume-stride", int, False),
    ("volume", "max_gib"): (
        "volume_max_gib", "--volume-max-gib", float, False
    ),
    ("volume", "opacity_scale"): (
        "volume_opacity_scale", "--volume-opacity-scale", float, False
    ),
    ("render", "particles"): (
        "no_particles", "--no-particles", bool, True
    ),
    ("render", "start_frame"): (
        "start_frame", "--start-frame", int, False
    ),
    ("render", "stop_frame"): ("stop_frame", "--stop-frame", int, False),
    ("render", "overwrite"): ("overwrite", "--overwrite", bool, False),
    ("output", "directory"): ("output_dir", "--output-dir", Path, False),
    ("movie", "enabled"): ("movie", "--movie", bool, False),
    ("movie", "path"): ("movie_path", "--movie-path", Path, False),
    ("movie", "codec"): ("codec", "--codec", str, False),
    ("movie", "crf"): ("crf", "--crf", int, False),
    ("movie", "qscale"): ("qscale", "--qscale", int, False),
    ("movie", "bitrate"): ("bitrate", "--bitrate", str, False),
}


def slice_series_pattern(run_dir, problem_id, slice_id):
    """Return the standard slice-series glob for one run and output id."""
    return (
        Path(run_dir).expanduser()
        / "slice"
        / slice_id
        / f"{problem_id}.*.{slice_id}.slice.vtk"
    )


def discover_slice_story_records(
    run_dir, problem_id, slice_id="midplane", time_tolerance=0.01
):
    """Discover and physically order the slice series used by the story."""
    pattern = slice_series_pattern(run_dir, problem_id, slice_id)
    records = index_slicevtk_series(pattern, time_tolerance=time_tolerance)
    if not records:
        raise FileNotFoundError(f"no slicevtk files match {str(pattern)!r}")
    return records


def _missing_field_groups(names, groups):
    names = set(names)
    return ["/".join(group) for group in groups if not names.intersection(group)]


def slice_preflight_report(records, stop_index=None):
    """Return a JSON-serializable capability report for indexed slices."""
    if not records:
        raise ValueError("preflight requires at least one slice record")
    stop_index = len(records) - 1 if stop_index is None else stop_index
    if not 0 <= stop_index < len(records):
        raise ValueError("stop_index lies outside the slice series")
    issues = []
    plane_intersections = {}
    plane_unions = {}
    for plane in ("x3", "x2"):
        field_sets = []
        groups = REQUIRED_FIELD_GROUPS
        for record in records[:stop_index + 1]:
            names = record.get("field_names", {}).get(plane)
            if names is None:
                issues.append(
                    f"output {record.get('num')} at t={record['time']:g} "
                    f"has no {plane} plane"
                )
                continue
            field_sets.append(set(names))
            missing = _missing_field_groups(names, groups)
            if missing:
                issues.append(
                    f"output {record.get('num')} plane {plane} is missing "
                    + ", ".join(missing)
                )
        if field_sets:
            plane_intersections[plane] = sorted(set.intersection(*field_sets))
            plane_unions[plane] = sorted(set.union(*field_sets))
        else:
            plane_intersections[plane] = []
            plane_unions[plane] = []
    final_names = records[stop_index].get("field_names", {}).get("x2", [])
    final_missing = _missing_field_groups(final_names, FINAL_SIDE_FIELD_GROUPS)
    if final_missing:
        issues.append(
            f"final output {records[stop_index].get('num')} plane x2 is missing "
            + ", ".join(final_missing)
        )
    return {
        "frame_count": len(records),
        "first_time": float(records[0]["time"]),
        "last_time": float(records[-1]["time"]),
        "first_num": records[0].get("num"),
        "last_num": records[-1].get("num"),
        "validated_stop_index": stop_index,
        "validated_stop_num": records[stop_index].get("num"),
        "plane_field_intersection": plane_intersections,
        "plane_field_union": plane_unions,
        "issues": issues,
        "ready_for_slice_story": not issues,
    }


def require_slice_story_capabilities(records, stop_index=None):
    """Raise one compact error if required slice fields or planes are absent."""
    report = slice_preflight_report(records, stop_index=stop_index)
    if report["issues"]:
        details = "\n  ".join(report["issues"][:12])
        suffix = (
            f"\n  ... and {len(report['issues']) - 12} more"
            if len(report["issues"]) > 12
            else ""
        )
        raise ValueError(f"slice-story preflight failed:\n  {details}{suffix}")
    return report


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    raise TypeError(f"unsupported resolved-config value {value!r}")


def write_resolved_config(path, sections):
    """Atomically write simple resolved settings as valid TOML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is not None:
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    return path


def volume_preflight_report(run_dir, problem_id, num, time_tolerance=0.01):
    """Inspect one full-volume output without allocating its field arrays."""
    paths = discover_vtk_pieces(run_dir, problem_id, num)
    layout = inspect_vtk_volume(paths, time_tolerance=time_tolerance)
    selected = volume_input_fields(layout.field_names)
    required_bytes = estimate_volume_bytes(layout, selected)
    return {
        "output_num": str(num),
        "time": layout.time,
        "piece_count": len(paths),
        "shape_xyz": list(layout.shape_xyz),
        "left_edge": list(layout.left_edge),
        "right_edge": list(layout.right_edge),
        "selected_fields": list(selected),
        "assembly_bytes": required_bytes,
        "scipy_available": importlib.util.find_spec("scipy") is not None,
        "ready_for_volume_story": importlib.util.find_spec("scipy") is not None,
    }


def scaled_durations(scale):
    """Return default storyboard durations multiplied by ``scale``."""
    if scale <= 0.0:
        raise ValueError("duration scale must be positive")
    defaults = StoryDurations()
    return StoryDurations(**{
        item.name: getattr(defaults, item.name) * scale
        for item in fields(defaults)
    })


def _load_toml(path):
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as error:
            raise RuntimeError(
                "TOML configuration on Python < 3.11 requires tomli"
            ) from error
    with Path(path).expanduser().open("rb") as stream:
        return tomllib.load(stream)


def _config_value(value, expected, label):
    if expected is bool:
        if not isinstance(value, bool):
            raise ValueError(f"config {label} must be a boolean")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"config {label} must be an integer")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config {label} must be a number")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ValueError(f"config {label} must be a string")
        return value
    if expected is Path:
        if not isinstance(value, str):
            raise ValueError(f"config {label} must be a path string")
        return Path(value).expanduser()
    raise TypeError(f"unknown config value type for {label}")


def apply_movie_config(args, config, argv):
    """Merge validated TOML settings unless their CLI option was explicit."""
    if not isinstance(config, dict):
        raise ValueError("movie config must contain TOML tables")
    explicit = {
        token.split("=", 1)[0]
        for token in argv
        if isinstance(token, str) and token.startswith("--")
    }
    duration_names = {item.name for item in fields(StoryDurations)}
    allowed = set(CONFIG_ARGUMENTS)
    allowed.update(("durations", name) for name in duration_names)
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"config [{section}] must be a TOML table")
        for key in values:
            if (section, key) not in allowed:
                raise ValueError(f"unknown config key {section}.{key}")

    for (section, key), (destination, option, expected, invert) in (
        CONFIG_ARGUMENTS.items()
    ):
        if section not in config or key not in config[section] or option in explicit:
            continue
        value = _config_value(
            config[section][key], expected, f"{section}.{key}"
        )
        setattr(args, destination, not value if invert else value)

    duration_values = config.get("durations", {})
    overrides = {
        name: _config_value(value, float, f"durations.{name}")
        for name, value in duration_values.items()
    }
    if any(value < 0.0 for value in overrides.values()):
        raise ValueError("configured durations must be non-negative")
    return overrides


def configured_durations(scale, overrides=None):
    """Apply per-scene base durations and then the global duration scale."""
    base = StoryDurations()
    values = {item.name: getattr(base, item.name) for item in fields(base)}
    values.update(overrides or {})
    return StoryDurations(**{
        name: value * scale for name, value in values.items()
    })


def _particle_frame(run_dir, problem_id, num):
    if num is None:
        return None
    candidates = (
        Path(run_dir) / "starpar" / f"{problem_id}.{num}.starpar.vtk",
        Path(run_dir) / "id0" / f"{problem_id}.{num}.starpar.vtk",
        Path(run_dir) / f"{problem_id}.{num}.starpar.vtk",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    return None if path is None else read_starpar(path)


def _render_single_view(
    slc,
    view,
    plane,
    field,
    *,
    derived,
    particles,
    particle_alpha,
    settings,
):
    if view == "slice":
        return render_slice_view(
            slc,
            plane,
            field,
            derived=derived,
            particles=particles,
            particle_alpha=particle_alpha,
            settings=settings,
        )
    if view == "velocity_streamlines":
        return render_streamline_view(
            slc, plane, "velocity", derived=derived, settings=settings
        )
    if view == "magnetic_streamlines":
        return render_streamline_view(
            slc, plane, "magnetic", derived=derived, settings=settings
        )
    if view == "radiation_composite":
        return render_radiation_view(
            slc, plane=plane, derived=derived, settings=settings
        )
    raise ValueError(f"unknown storyboard view {view!r}")


def render_story_frames(
    requests,
    run_dir,
    problem_id,
    output_dir,
    *,
    settings=None,
    time_tolerance=0.01,
    start_frame=0,
    stop_frame=None,
    overwrite=False,
    particles=True,
    volume_stride=1,
    volume_max_bytes=None,
    volume_opacity_scale=0.08,
):
    """Render a sequence of storyboard requests with source/view caching."""
    settings = CanvasSettings() if settings is None else settings
    settings.validate()
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    stop_frame = len(requests) if stop_frame is None else min(
        stop_frame, len(requests)
    )
    if stop_frame < start_frame:
        raise ValueError("stop_frame must be greater than or equal to start_frame")
    if not isinstance(volume_stride, int) or volume_stride < 1:
        raise ValueError("volume_stride must be a positive integer")
    if volume_max_bytes is not None and volume_max_bytes <= 0:
        raise ValueError("volume_max_bytes must be positive")
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    slice_cache = {}
    derived_cache = {}
    particle_cache = {}
    volume_cache = {}
    view_cache = {}
    written = []
    skipped = []

    def load_slice(path):
        if path not in slice_cache:
            slice_cache[path] = read_slicevtk(path)
        return slice_cache[path]

    def load_derived(slc, path, plane):
        key = (path, plane)
        if key not in derived_cache:
            derived_cache[key] = derive_plane_fields(
                slc, plane, muH=settings.muH
            )
        return derived_cache[key]

    def load_particles(request):
        if not particles or request.particle_alpha <= 0.0:
            return None
        key = request.source_num
        if key not in particle_cache:
            particle_cache[key] = _particle_frame(run_dir, problem_id, key)
        particle_frame = particle_cache[key]
        if particle_frame is not None:
            difference = abs(
                float(particle_frame["time"]) - request.simulation_time
            )
            if difference > time_tolerance:
                raise ValueError(
                    f"particle time mismatch for output {key}: "
                    f"{particle_frame['time']:g} versus {request.simulation_time:g}"
                )
        return particle_frame

    def load_volume(request):
        key = request.source_num
        if key is None:
            raise ValueError("volume rendering requires a numbered source output")
        if key not in volume_cache:
            paths = discover_vtk_pieces(run_dir, problem_id, key)
            layout = inspect_vtk_volume(paths, time_tolerance=time_tolerance)
            difference = abs(layout.time - request.simulation_time)
            if difference > time_tolerance:
                raise ValueError(
                    f"volume time mismatch for output {key}: "
                    f"{layout.time:g} versus {request.simulation_time:g}"
                )
            selected = volume_input_fields(layout.field_names)
            volume_cache[key] = read_vtk_volume(
                paths,
                selected,
                time_tolerance=time_tolerance,
                max_bytes=volume_max_bytes,
            )
        return volume_cache[key]

    def render_view(request, which):
        view = getattr(request, f"view_{which}")
        plane = getattr(request, f"plane_{which}")
        field = getattr(request, f"field_{which}")
        if view is None:
            return None
        cache_key = (
            request.source_path,
            view,
            plane,
            field,
            round(request.camera_fraction, 10) if view == "volume" else None,
            round(request.particle_alpha, 10),
            bool(particles),
        )
        if cache_key not in view_cache:
            if view == "volume":
                volume = load_volume(request)
                raw_image = render_temperature_volume(
                    volume,
                    request.camera_fraction,
                    stride=volume_stride,
                    opacity_scale=volume_opacity_scale,
                    muH=settings.muH,
                )
                view_cache[cache_key] = render_volume_view(
                    raw_image, volume["time"], settings=settings
                )
            else:
                slc = load_slice(request.source_path)
                derived = None if plane is None else load_derived(
                    slc, request.source_path, plane
                )
                view_cache[cache_key] = _render_single_view(
                    slc,
                    view,
                    plane,
                    field,
                    derived=derived,
                    particles=load_particles(request),
                    particle_alpha=request.particle_alpha,
                    settings=settings,
                )
        return view_cache[cache_key]

    for request in requests[start_frame:stop_frame]:
        output = frames_dir / f"frame_{request.frame_number:06d}.png"
        if output.exists() and not overwrite:
            skipped.append(output)
            continue
        first = render_view(request, "a")
        second = render_view(request, "b")
        pixels = (
            first
            if second is None
            else blend_rgba(first, second, request.blend)
        )
        write_png_frame(output, pixels, overwrite=True)
        written.append(output)
    return {"written": written, "skipped": skipped, "frames_dir": frames_dir}


def encode_story_movie(
    frames_dir,
    movie_path,
    *,
    fps=30.0,
    codec="auto",
    crf=18,
    qscale=2,
    bitrate=None,
    overwrite=True,
):
    """Encode contiguous story PNGs using the repository's codec conventions."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH")
    frames_dir = Path(frames_dir)
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"no frame_*.png files under {frames_dir}")
    numbers = [int(path.stem.rsplit("_", 1)[1]) for path in frames]
    expected = list(range(numbers[0], numbers[-1] + 1))
    if numbers != expected or numbers[0] != 0:
        raise ValueError("movie encoding requires contiguous frames starting at zero")

    codec = resolve_video_codec(codec)
    movie_path = Path(movie_path)
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(frames_dir / "frame_*.png"),
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-vcodec", codec,
        *codec_quality_args(codec, crf, qscale, bitrate),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(movie_path),
    ]
    subprocess.run(command, check=True)
    return movie_path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--problem-id")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--slice-id", default="midplane")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--time-tolerance", type=float, default=0.01)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--freeze-index", type=int)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int)
    parser.add_argument("--duration-scale", type=float)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no-volume", action="store_true")
    parser.add_argument("--volume-stride", type=int)
    parser.add_argument("--volume-max-gib", type=float, default=8.0)
    parser.add_argument("--volume-opacity-scale", type=float, default=0.08)
    parser.add_argument("--no-particles", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--movie-path", type=Path)
    parser.add_argument("--codec", default="auto")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--qscale", type=int, default=2)
    parser.add_argument("--bitrate")
    return parser


def main(argv=None):
    parser = _parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    duration_overrides = {}
    if args.config is not None:
        try:
            duration_overrides = apply_movie_config(
                args, _load_toml(args.config), raw_argv
            )
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    if not args.problem_id:
        parser.error("--problem-id is required (directly or in [source])")
    if args.time_tolerance < 0.0:
        parser.error("--time-tolerance must be non-negative")
    records = discover_slice_story_records(
        args.run_dir,
        args.problem_id,
        slice_id=args.slice_id,
        time_tolerance=args.time_tolerance,
    )
    try:
        report = slice_preflight_report(records, stop_index=args.stop_index)
    except ValueError as error:
        parser.error(str(error))
    if args.preflight:
        if not args.no_volume:
            last_index = len(records) - 1
            stop_index = last_index if args.stop_index is None else args.stop_index
            freeze_index = (
                (args.start_index + stop_index) // 2
                if args.freeze_index is None
                else args.freeze_index
            )
            if not 0 <= args.start_index <= freeze_index <= stop_index <= last_index:
                parser.error(
                    "require 0 <= start-index <= freeze-index <= stop-index"
                )
            try:
                report["volume"] = volume_preflight_report(
                    args.run_dir,
                    args.problem_id,
                    records[freeze_index].get("num"),
                    time_tolerance=args.time_tolerance,
                )
            except (FileNotFoundError, KeyError, ValueError, NotImplementedError) as error:
                report["volume"] = {
                    "ready_for_volume_story": False,
                    "error": str(error),
                }
        print(json.dumps(report, indent=2))
        return report
    require_slice_story_capabilities(records, stop_index=args.stop_index)
    if args.volume_max_gib <= 0.0:
        parser.error("--volume-max-gib must be positive")
    if args.volume_opacity_scale <= 0.0:
        parser.error("--volume-opacity-scale must be positive")

    fps = args.fps if args.fps is not None else (4.0 if args.preview else 30.0)
    width = args.width if args.width is not None else (960 if args.preview else 1920)
    height = args.height if args.height is not None else (540 if args.preview else 1080)
    duration_scale = (
        args.duration_scale
        if args.duration_scale is not None
        else (0.15 if args.preview else 1.0)
    )
    volume_stride = (
        args.volume_stride
        if args.volume_stride is not None
        else (4 if args.preview else 2)
    )
    if volume_stride < 1:
        parser.error("--volume-stride must be a positive integer")
    try:
        durations = configured_durations(duration_scale, duration_overrides)
        requests = build_slice_storyboard(
            records,
            fps=fps,
            start_index=args.start_index,
            freeze_index=args.freeze_index,
            stop_index=args.stop_index,
            durations=durations,
            include_volume=not args.no_volume,
        )
    except ValueError as error:
        parser.error(str(error))

    output_dir = args.output_dir or args.run_dir / "movie_slice_story"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame_manifest(output_dir / "frame_manifest.csv", requests)
    resolved_stop_index = (
        len(records) - 1 if args.stop_index is None else args.stop_index
    )
    resolved_freeze_index = (
        (args.start_index + resolved_stop_index) // 2
        if args.freeze_index is None
        else args.freeze_index
    )
    write_resolved_config(output_dir / "resolved_config.toml", {
        "source": {
            "run_dir": str(args.run_dir.expanduser().resolve()),
            "problem_id": args.problem_id,
            "slice_id": args.slice_id,
            "time_tolerance": args.time_tolerance,
            "config_path": (
                str(args.config.expanduser().resolve())
                if args.config is not None else None
            ),
        },
        "story": {
            "fps": fps,
            "frame_count": len(requests),
            "start_index": args.start_index,
            "freeze_index": resolved_freeze_index,
            "stop_index": resolved_stop_index,
            "duration_scale": duration_scale,
        },
        "durations_seconds": {
            item.name: getattr(durations, item.name)
            for item in fields(durations)
        },
        "canvas": {"width": width, "height": height},
        "volume": {
            "enabled": not args.no_volume,
            "stride": volume_stride,
            "max_gib": args.volume_max_gib,
            "opacity_scale": args.volume_opacity_scale,
        },
        "render": {
            "particles": not args.no_particles,
            "start_frame": args.start_frame,
            "stop_frame": args.stop_frame,
            "overwrite": args.overwrite,
        },
        "movie": {
            "enabled": args.movie,
            "codec": args.codec,
            "crf": args.crf,
            "qscale": args.qscale,
            "bitrate": args.bitrate,
        },
    })
    result = render_story_frames(
        requests,
        args.run_dir,
        args.problem_id,
        output_dir,
        settings=CanvasSettings(width=width, height=height),
        time_tolerance=args.time_tolerance,
        start_frame=args.start_frame,
        stop_frame=args.stop_frame,
        overwrite=args.overwrite,
        particles=not args.no_particles,
        volume_stride=volume_stride,
        volume_max_bytes=int(args.volume_max_gib * 1024**3),
        volume_opacity_scale=args.volume_opacity_scale,
    )
    print(
        f"Rendered {len(result['written'])} frames; "
        f"skipped {len(result['skipped'])} existing frames"
    )
    if args.movie:
        movie_path = args.movie_path or output_dir / "slice_story.mp4"
        encode_story_movie(
            result["frames_dir"],
            movie_path,
            fps=fps,
            codec=args.codec,
            crf=args.crf,
            qscale=args.qscale,
            bitrate=args.bitrate,
        )
        print(f"Wrote {movie_path}")
    return result


if __name__ == "__main__":
    main()
