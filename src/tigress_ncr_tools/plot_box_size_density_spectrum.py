#!/usr/bin/env python3
"""Compare shear-aware gas-column density spectra across physical box sizes."""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathena.hst_reader import read_hst
from pathena.proj2d_reader import read_proj2d

from .plot_suite_density_spectrum import (
    DEFAULT_K_BINS,
    DEFAULT_POWER2D_MODE_LIMIT,
    DEFAULT_SPECTRAL_BAND_PC,
    _atomic_csv,
    _atomic_savez,
    _axial_angle_statistics,
    _finite_statistics,
    _positive_limits,
    _uniform_center_edges,
    integral_scale,
    load_spectrum_archive,
    overdensity_power_2d,
    physical_time_mean_power_2d,
    spectral_slope_alpha,
    spectrum_time_diagnostics,
    time_mean_power,
)
from .plot_suite_evolution import nearest_indexed_projection, time_average
from .surface_density_stats import (
    annular_power_statistics_2d,
    band_power_quadrupole_2d,
    read_shear_parameters,
)

DEFAULT_SUITE = Path("/projects/c/changgoo/nasa_athena/TIGRESS-NCR")
DEFAULT_MODEL_GLOB = "R8_8pc_NCR_Lxy*_early"
DEFAULT_PROJECTION = "theta0"
DEFAULT_TIME_RANGE = (201, 600)
DEFAULT_AVERAGE_RANGE = (201.0, 600.0)
DEFAULT_OUTPUT_NAME = "density_power_spectrum_box_size_theta0"
DEFAULT_ARCHIVE_NAME = "box_size_density_power_spectra.npz"
DEFAULT_MEAN_FIGURE = "box_size_density_power_spectrum_time_mean.png"
DEFAULT_POWER2D_FIGURE = "box_size_density_power_2d_time_mean.png"
DEFAULT_DIAGNOSTIC_NAME = "box_size_density_power_spectrum_diagnostics"
POWER2D_DIRECTORY = "density_power_2d"
POWER2D_MEAN_DIRECTORY = "density_power_2d_time_mean"
BOX_SIZE_PATTERN = re.compile(r"_Lxy(?P<size>\d+)$")


@dataclass(frozen=True)
class SegmentedBoxModel:
    """One physical model whose output is split across run directories."""

    name: str
    segments: tuple
    box_size_pc: float


def _logical_model_name(segment_name):
    for suffix in ("_early", "_late"):
        if segment_name.endswith(suffix):
            return segment_name[: -len(suffix)]
    raise ValueError(f"segment name has no early/late suffix: {segment_name}")


def _box_size_from_name(name):
    match = BOX_SIZE_PATTERN.search(name)
    if match is None:
        raise ValueError(f"cannot determine Lxy from model name {name!r}")
    return float(match.group("size"))


def discover_box_size_models(
    suite, model_glob=DEFAULT_MODEL_GLOB, proj_id=DEFAULT_PROJECTION
):
    """Discover complete early/late pairs and return them in increasing Lxy."""
    suite = Path(suite)
    models = []
    for early in sorted(suite.glob(model_glob)):
        if not early.is_dir() or not early.name.endswith("_early"):
            continue
        name = _logical_model_name(early.name)
        late = early.with_name(f"{name}_late")
        segments = (early, late)
        missing = [
            segment
            for segment in segments
            if not segment.is_dir()
            or not (segment / "proj2d" / proj_id).is_dir()
            or not list((segment / "hst").glob("*.hst"))
        ]
        if missing:
            raise FileNotFoundError(
                f"incomplete segmented model {name}: "
                + ", ".join(str(path) for path in missing)
            )
        models.append(
            SegmentedBoxModel(
                name=name,
                segments=segments,
                box_size_pc=_box_size_from_name(name),
            )
        )
    if not models:
        raise FileNotFoundError(
            f"no complete early/late models under {suite} match {model_glob!r}"
        )
    return sorted(models, key=lambda model: (model.box_size_pc, model.name))


def segmented_projection_index(model, proj_id=DEFAULT_PROJECTION):
    """Join segment projections by output number, preferring later segments."""
    index = {}
    for segment in model.segments:
        directory = segment / "proj2d" / proj_id
        for path in sorted(directory.glob("*.proj2d")):
            try:
                number = int(path.name.split(".")[-3])
            except (IndexError, ValueError):
                continue
            index[number] = path
    if not index:
        raise FileNotFoundError(f"no {proj_id} projections for {model.name}")
    return index


def segmented_shear_parameters(model):
    """Read and validate one common q and Omega across all run segments."""
    found = [read_shear_parameters(segment) for segment in model.segments]
    reference = np.asarray(found[0][1:], dtype=float)
    if any(not np.allclose(item[1:], reference) for item in found[1:]):
        details = ", ".join(
            f"{segment.name}: q={item[1]:g}, Omega={item[2]:g}"
            for segment, item in zip(model.segments, found)
        )
        raise ValueError(f"inconsistent shear parameters for {model.name}: {details}")
    return tuple(item[0] for item in found), float(reference[0]), float(reference[1])


def _primary_history_path(segment):
    paths = [
        path
        for path in sorted((segment / "hst").glob("*.hst"))
        if ".phase" not in path.name and ".whole" not in path.name
    ]
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one primary history under {segment / 'hst'}, found {len(paths)}"
        )
    return paths[0]


def segmented_mean_sfr(model, bounds=DEFAULT_AVERAGE_RANGE, max_rows=100000):
    """Join early/late histories and time-average sfr10 over bounds."""
    times = []
    values = []
    for segment in model.segments:
        history = read_hst(_primary_history_path(segment), max_rows=max_rows)
        times.append(np.asarray(history["time"], dtype=float))
        values.append(np.asarray(history["sfr10"], dtype=float))
    time = np.concatenate(times)
    sfr10 = np.concatenate(values)
    order = np.argsort(time, kind="stable")
    time = time[order]
    sfr10 = sfr10[order]
    keep = np.r_[time[1:] > time[:-1], True]
    return time_average(time[keep], sfr10[keep], bounds)


def model_power2d_archive(output_dir, model):
    """Return the cache path for one logical box-size model."""
    return Path(output_dir) / POWER2D_DIRECTORY / f"{model.name}.npz"


def _power2d_cache_matches(path, model, targets, window, tukey_alpha, pad_factor):
    try:
        with np.load(path) as saved:
            return (
                str(saved["model"].item()) == model.name
                and np.array_equal(saved["target_time"], targets)
                and np.array_equal(
                    saved["segment_directory"].astype(str),
                    np.asarray([str(item) for item in model.segments]),
                )
                and str(saved["window"].item()) == str(window)
                and np.isclose(float(saved["tukey_alpha"]), tukey_alpha)
                and np.isclose(float(saved["pad_factor"]), pad_factor)
            )
    except (OSError, KeyError, ValueError):
        return False


def generate_segmented_model_power2d(
    model,
    *,
    proj_id,
    targets,
    window,
    tukey_alpha,
    pad_factor,
    output,
):
    """Generate a joined early/late 2D periodogram time series."""
    parameter_paths, qshear, omega = segmented_shear_parameters(model)
    projection_index = segmented_projection_index(model, proj_id)
    rows = []
    guess_offset = 0
    common_kx0 = common_ky = None
    pixel_size_xy = box_size_xy = None
    for frame_index, target in enumerate(targets, start=1):
        path, stored_time, guess_offset = nearest_indexed_projection(
            projection_index, float(target), guess_offset, tolerance=0.05
        )
        frame = read_proj2d(path, fields="nH")
        result = overdensity_power_2d(
            frame,
            qshear,
            omega,
            window=window,
            tukey_alpha=tukey_alpha,
            pad_factor=pad_factor,
        )
        if common_kx0 is None:
            common_kx0 = result["kx0"]
            common_ky = result["ky"]
            pixel_size_xy = result["pixel_size_xy"]
            box_size_xy = result["box_size_xy"]
            if not np.allclose(box_size_xy, model.box_size_pc):
                raise ValueError(
                    f"{model.name} has projection extent {box_size_xy.tolist()} pc; "
                    f"select times after its expansion to {model.box_size_pc:g} pc"
                )
        elif not (
            np.array_equal(common_kx0, result["kx0"])
            and np.array_equal(common_ky, result["ky"])
            and np.allclose(pixel_size_xy, result["pixel_size_xy"])
            and np.allclose(box_size_xy, result["box_size_xy"])
        ):
            raise ValueError(f"2D spectrum grid changes within {model.name}")
        rows.append(
            (
                stored_time,
                str(path),
                result["power_2d"].astype(np.float32),
                result["mean_sigma"],
                result["remap_time"],
                result["shear"],
                result["has_negative_sigma"],
            )
        )
        if frame_index == 1 or frame_index % 100 == 0 or frame_index == len(targets):
            print(f"{model.name}: {frame_index}/{len(targets)} 2D spectra", flush=True)
    data = {
        "model": np.asarray(model.name),
        "segment_directory": np.asarray([str(path) for path in model.segments]),
        "projection_id": np.asarray(proj_id),
        "field": np.asarray("nH"),
        "delta_definition": np.asarray("Sigma/<Sigma>-1"),
        "coordinate_remap": np.asarray("g(x,y)=Sigma(x,y-shear*x)"),
        "wavenumber_mapping": np.asarray("kx=kx0+shear*ky"),
        "power_normalization": np.asarray(
            "|dx dy FFT(delta)|^2 / (dx dy sum(window^2))"
        ),
        "power_unit": np.asarray("pc^2"),
        "k_unit": np.asarray("pc^-1"),
        "target_time": np.asarray(targets),
        "time": np.asarray([row[0] for row in rows]),
        "source_projection": np.asarray([row[1] for row in rows]),
        "power_2d": np.asarray([row[2] for row in rows]),
        "kx0": common_kx0,
        "ky": common_ky,
        "mean_sigma_code": np.asarray([row[3] for row in rows]),
        "remap_time": np.asarray([row[4] for row in rows]),
        "shear": np.asarray([row[5] for row in rows]),
        "has_negative_sigma": np.asarray([row[6] for row in rows]),
        "qshear": np.asarray(qshear),
        "omega_kms_per_pc": np.asarray(omega),
        "pixel_size_xy_pc": pixel_size_xy,
        "box_size_xy_pc": box_size_xy,
        "parameter_source": np.asarray([str(path) for path in parameter_paths]),
        "window": np.asarray(window),
        "tukey_alpha": np.asarray(tukey_alpha),
        "pad_factor": np.asarray(pad_factor),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(output, **data)
    print(f"Wrote {output}", flush=True)
    return str(output)


def _ensure_power2d(task):
    model, proj_id, targets, window, tukey_alpha, pad_factor, output, overwrite = task
    if (
        output.exists()
        and not overwrite
        and _power2d_cache_matches(
            output, model, targets, window, tukey_alpha, pad_factor
        )
    ):
        print(f"Keeping compatible {output}", flush=True)
        return str(output)
    return generate_segmented_model_power2d(
        model,
        proj_id=proj_id,
        targets=targets,
        window=window,
        tukey_alpha=tukey_alpha,
        pad_factor=pad_factor,
        output=output,
    )


def common_k_edges_from_archives(archives, bins=DEFAULT_K_BINS):
    """Span the largest-box fundamental through the common Nyquist limit."""
    box_sizes = [np.asarray(data["box_size_xy_pc"], dtype=float) for data in archives]
    pixel_sizes = [
        np.asarray(data["pixel_size_xy_pc"], dtype=float) for data in archives
    ]
    kmin = min(2.0 * np.pi / np.min(size) for size in box_sizes)
    kmax = min(np.pi / np.max(size) for size in pixel_sizes)
    kmin = np.nextafter(kmin, 0.0)
    if not 0.0 < kmin < kmax:
        raise ValueError("box and pixel sizes do not define a common k range")
    return np.geomspace(kmin, kmax, int(bins) + 1)


def analyze_box_size_power(
    models,
    *,
    output_dir,
    proj_id=DEFAULT_PROJECTION,
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    average_bounds=DEFAULT_AVERAGE_RANGE,
    k_bins=DEFAULT_K_BINS,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
    overwrite_2d=False,
    workers=1,
):
    """Build per-model 2D caches and one common-grid annular archive."""
    targets = np.arange(int(start), int(stop) + 1, int(stride), dtype=int)
    tasks = [
        (
            model,
            proj_id,
            targets,
            window,
            tukey_alpha,
            pad_factor,
            model_power2d_archive(output_dir, model),
            overwrite_2d,
        )
        for model in models
    ]
    if int(workers) == 1:
        paths = [_ensure_power2d(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            paths = list(executor.map(_ensure_power2d, tasks))

    metadata = []
    for path in paths:
        with np.load(path) as saved:
            metadata.append(
                {
                    "box_size_xy_pc": saved["box_size_xy_pc"],
                    "pixel_size_xy_pc": saved["pixel_size_xy_pc"],
                }
            )
    k_edges = common_k_edges_from_archives(metadata, bins=k_bins)
    k_centers = np.sqrt(k_edges[:-1] * k_edges[1:])
    rows = []
    for model, path in zip(models, paths):
        data = load_spectrum_archive(path)
        powers = []
        counts = []
        quadrupoles = []
        band_quadrupoles = []
        band_counts = []
        for index, shear in enumerate(data["shear"]):
            power, count, q2 = annular_power_statistics_2d(
                data["power_2d"][index],
                data["kx0"],
                data["ky"],
                shear,
                k_edges,
            )
            band_q2, band_count = band_power_quadrupole_2d(
                data["power_2d"][index],
                data["kx0"],
                data["ky"],
                shear,
                DEFAULT_SPECTRAL_BAND_PC,
            )
            powers.append(power)
            counts.append(count)
            quadrupoles.append(q2)
            band_quadrupoles.append(band_q2)
            band_counts.append(band_count)
        rows.append(
            {
                "time": data["time"],
                "power": powers,
                "count": counts,
                "q2": quadrupoles,
                "band_q2": band_quadrupoles,
                "band_count": band_counts,
                "mean_sigma": data["mean_sigma_code"],
                "remap_time": data["remap_time"],
                "shear": data["shear"],
                "negative": data["has_negative_sigma"],
                "qshear": float(data["qshear"]),
                "omega": float(data["omega_kms_per_pc"]),
                "pixel_size": float(np.max(data["pixel_size_xy_pc"])),
                "box_size": float(np.min(data["box_size_xy_pc"])),
                "parameter_source": data["parameter_source"],
            }
        )
        print(f"{model.name}: reduced {len(targets)} spectra", flush=True)
        del data
    time = np.asarray([row["time"] for row in rows])
    if np.any(np.ptp(time, axis=0) > 0.05):
        bad = np.flatnonzero(np.ptp(time, axis=0) > 0.05)
        raise ValueError(
            f"models are not time-aligned at target indices {bad.tolist()}"
        )
    power = np.asarray([row["power"] for row in rows])
    q2 = np.asarray([row["q2"] for row in rows])
    band_q2 = np.asarray([row["band_q2"] for row in rows])
    mean_sfr = np.asarray(
        [segmented_mean_sfr(model, bounds=average_bounds) for model in models]
    )
    return {
        "model": np.asarray([model.name for model in models]),
        "segment_directory": np.asarray(
            [[str(path) for path in model.segments] for model in models]
        ),
        "box_size_pc": np.asarray([row["box_size"] for row in rows]),
        "mean_sfr10": mean_sfr,
        "sfr_time_bounds": np.asarray(average_bounds),
        "projection_id": np.asarray(proj_id),
        "field": np.asarray("nH"),
        "target_time": targets,
        "time": time,
        "k_edges": k_edges,
        "k_centers": k_centers,
        "power_delta": power,
        "dimensionless_power": (k_centers[None, None, :] ** 2 * power / (2.0 * np.pi)),
        "mode_count": np.asarray([row["count"] for row in rows]),
        "q2_real": q2.real,
        "q2_imaginary": q2.imag,
        "anisotropy_amplitude": np.abs(q2),
        "anisotropy_angle_rad": 0.5 * np.angle(q2),
        "anisotropy_band_wavelength_pc": np.asarray(DEFAULT_SPECTRAL_BAND_PC),
        "anisotropy_band_q2_real_time": band_q2.real,
        "anisotropy_band_q2_imaginary_time": band_q2.imag,
        "anisotropy_band_amplitude_time": np.abs(band_q2),
        "anisotropy_band_angle_rad_time": 0.5 * np.angle(band_q2),
        "anisotropy_band_mode_count_time": np.asarray(
            [row["band_count"] for row in rows]
        ),
        "mean_sigma_code": np.asarray([row["mean_sigma"] for row in rows]),
        "remap_time": np.asarray([row["remap_time"] for row in rows]),
        "shear": np.asarray([row["shear"] for row in rows]),
        "has_negative_sigma": np.asarray([row["negative"] for row in rows]),
        "qshear": np.asarray([row["qshear"] for row in rows]),
        "omega_kms_per_pc": np.asarray([row["omega"] for row in rows]),
        "pixel_size_pc": np.asarray([row["pixel_size"] for row in rows]),
        "parameter_source": np.asarray([row["parameter_source"] for row in rows]),
        "power2d_archive": np.asarray(paths),
        "power2d_storage_dtype": np.asarray("float32"),
        "window": np.asarray(window),
        "tukey_alpha": np.asarray(tukey_alpha),
        "pad_factor": np.asarray(pad_factor),
        "delta_definition": np.asarray("Sigma/<Sigma>-1"),
        "coordinate_remap": np.asarray("g(x,y)=Sigma(x,y-shear*x)"),
        "wavenumber_mapping": np.asarray("kx=kx0+shear*ky"),
        "power_normalization": np.asarray(
            "|dx dy FFT(delta)|^2 / (dx dy sum(window^2))"
        ),
        "power_unit": np.asarray("pc^2"),
        "k_unit": np.asarray("pc^-1"),
        "common_k_range_definition": np.asarray(
            "largest-box fundamental through smallest shared directional Nyquist"
        ),
    }


def box_size_diagnostic_summary(data, bounds=DEFAULT_AVERAGE_RANGE):
    """Summarize instantaneous spectral diagnostics for each box size."""
    diagnostics = spectrum_time_diagnostics(data)
    mean_power = time_mean_power(data, bounds)
    rows = []
    names = np.asarray(data["model"]).astype(str)
    for index, name in enumerate(names):
        use = (data["time"][index] >= bounds[0]) & (data["time"][index] <= bounds[1])
        scale_stats = _finite_statistics(
            diagnostics["integral_scale_time_pc"][index, use]
        )
        slope_stats = _finite_statistics(
            diagnostics["spectral_slope_alpha_time"][index, use]
        )
        amplitude_stats = _finite_statistics(
            data["anisotropy_band_amplitude_time"][index, use]
        )
        angle_stats = _axial_angle_statistics(
            data["anisotropy_band_angle_rad_time"][index, use]
        )
        reference_alpha, reference_count = spectral_slope_alpha(
            data["k_centers"], mean_power[index]
        )
        rows.append(
            {
                "model": name,
                "box_size_pc": float(data["box_size_pc"][index]),
                "pixel_size_pc": float(data["pixel_size_pc"][index]),
                "mean_sfr10": float(data["mean_sfr10"][index]),
                "integral_scale_time_mean_pc": scale_stats[0],
                "integral_scale_time_std_pc": scale_stats[1],
                "integral_scale_time_median_pc": scale_stats[2],
                "integral_scale_time_percentile16_pc": scale_stats[3],
                "integral_scale_time_percentile84_pc": scale_stats[4],
                "integral_scale_time_count": scale_stats[5],
                "spectral_slope_alpha_time_mean": slope_stats[0],
                "spectral_slope_alpha_time_std": slope_stats[1],
                "spectral_slope_alpha_time_median": slope_stats[2],
                "spectral_slope_alpha_time_percentile16": slope_stats[3],
                "spectral_slope_alpha_time_percentile84": slope_stats[4],
                "spectral_slope_alpha_time_count": slope_stats[5],
                "anisotropy_band_amplitude_time_mean": amplitude_stats[0],
                "anisotropy_band_amplitude_time_std": amplitude_stats[1],
                "anisotropy_band_amplitude_time_median": amplitude_stats[2],
                "anisotropy_band_amplitude_time_percentile16": amplitude_stats[3],
                "anisotropy_band_amplitude_time_percentile84": amplitude_stats[4],
                "anisotropy_band_amplitude_time_count": amplitude_stats[5],
                "anisotropy_band_angle_circular_mean_deg": np.rad2deg(angle_stats[0]),
                "anisotropy_band_angle_resultant_length": angle_stats[1],
                "anisotropy_band_angle_time_median_deg": np.rad2deg(angle_stats[2]),
                "anisotropy_band_angle_time_percentile16_deg": np.rad2deg(
                    angle_stats[3]
                ),
                "anisotropy_band_angle_time_percentile84_deg": np.rad2deg(
                    angle_stats[4]
                ),
                "anisotropy_band_angle_time_count": angle_stats[5],
                "integral_scale_pc": integral_scale(data["k_edges"], mean_power[index]),
                "spectral_slope_alpha": reference_alpha,
                "slope_fit_bin_count": reference_count,
                "average_start": float(bounds[0]),
                "average_stop": float(bounds[1]),
            }
        )
    return pd.DataFrame(rows), diagnostics


def attach_box_size_diagnostics(data, summary, diagnostics, bounds):
    """Attach per-time and summary diagnostics to the numerical archive."""
    result = dict(data)
    result.update(diagnostics)
    for column in summary.columns:
        if column not in ("model", "average_start", "average_stop"):
            result[column] = summary[column].to_numpy()
    result["diagnostic_time_bounds"] = np.asarray(bounds, dtype=float)
    result["integral_scale_definition"] = np.asarray(
        "integral(E(k)*2pi/k dk)/integral(E(k) dk), E(k)=k*P_delta(k)/(2pi)"
    )
    result["spectral_slope_definition"] = np.asarray(
        "P_delta(k) proportional to k^-alpha for 64 pc < 2pi/k < 256 pc"
    )
    return result


def _box_colors(box_sizes, cmap_name="viridis"):
    values = np.asarray(box_sizes, dtype=float)
    cmap = mpl.colormaps[cmap_name]
    norm = mpl.colors.LogNorm(vmin=np.min(values), vmax=np.max(values))
    return cmap, norm


def _reciprocal_wavelength(value):
    value = np.asarray(value, dtype=float)
    return np.divide(
        2.0 * np.pi,
        value,
        out=np.full_like(value, np.inf),
        where=value != 0.0,
    )


def plot_box_size_time_mean(data, output, *, bounds=DEFAULT_AVERAGE_RANGE, dpi=180):
    """Plot time-mean dimensional spectra and median quadrupole amplitude."""
    k = np.asarray(data["k_centers"], dtype=float)
    power = time_mean_power(data, bounds)
    amplitude = np.full_like(power, np.nan)
    for index in range(power.shape[0]):
        use = (data["time"][index] >= bounds[0]) & (data["time"][index] <= bounds[1])
        selected = np.asarray(data["anisotropy_amplitude"][index, use], dtype=float)
        populated = np.any(np.isfinite(selected), axis=0)
        amplitude[index, populated] = np.nanmedian(selected[:, populated], axis=0)
    cmap, norm = _box_colors(data["box_size_pc"])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), sharex=True)
    for index in range(power.shape[0]):
        size = float(data["box_size_pc"][index])
        color = cmap(norm(size))
        label = rf"$L={size:g}\,$pc"
        valid = np.isfinite(power[index]) & (power[index] > 0.0)
        axes[0].plot(k[valid], power[index, valid], color=color, label=label)
        valid = np.isfinite(amplitude[index])
        axes[1].plot(k[valid], amplitude[index, valid], color=color, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$P_\delta(k)\;[\mathrm{pc}^2]$")
    axes[1].set_ylabel(r"$A_2(k)=|Q_2(k)|$")
    axes[1].set_ylim(0.0, 1.0)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(data["k_edges"][0], data["k_edges"][-1])
        axis.set_xlabel(r"physical wavenumber $k\;[\mathrm{pc}^{-1}]$")
        axis.grid(alpha=0.18, which="both")
        axis.tick_params(direction="in", top=True, right=True)
        top = axis.secondary_xaxis(
            "top", functions=(_reciprocal_wavelength, _reciprocal_wavelength)
        )
        top.set_xlabel(r"wavelength $\lambda=2\pi/k\;[\mathrm{pc}]$")
    axes[0].legend(frameon=False)
    fig.suptitle(
        f"Box-size gas-column spectra: {bounds[0]:g}--{bounds[1]:g} Myr",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def plot_box_size_diagnostics(summary, output, *, dpi=180):
    """Plot temporal diagnostic medians and percentiles versus box size."""
    specifications = (
        (
            "integral_scale_time_median_pc",
            "integral_scale_time_percentile16_pc",
            "integral_scale_time_percentile84_pc",
            r"$L_{\rm in}\;[\mathrm{pc}]$",
        ),
        (
            "spectral_slope_alpha_time_median",
            "spectral_slope_alpha_time_percentile16",
            "spectral_slope_alpha_time_percentile84",
            r"$\alpha\quad(P_\delta\propto k^{-\alpha})$",
        ),
        (
            "anisotropy_band_amplitude_time_median",
            "anisotropy_band_amplitude_time_percentile16",
            "anisotropy_band_amplitude_time_percentile84",
            r"$A_{2,64-256}$",
        ),
        (
            "anisotropy_band_angle_circular_mean_deg",
            "anisotropy_band_angle_time_percentile16_deg",
            "anisotropy_band_angle_time_percentile84_deg",
            r"$\phi_{2,64-256}\;[\mathrm{deg}]$",
        ),
    )
    x = summary["box_size_pc"].to_numpy(dtype=float)
    colors = mpl.colormaps["viridis"](np.linspace(0.15, 0.85, len(summary)))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex=True)
    for axis, specification in zip(axes.flat, specifications):
        center = summary[specification[0]].to_numpy(dtype=float)
        low = summary[specification[1]].to_numpy(dtype=float)
        high = summary[specification[2]].to_numpy(dtype=float)
        valid = np.isfinite(center) & np.isfinite(low) & np.isfinite(high)
        for index in np.flatnonzero(valid):
            axis.errorbar(
                x[index],
                center[index],
                yerr=[
                    [center[index] - low[index]],
                    [high[index] - center[index]],
                ],
                fmt="o",
                color=colors[index],
                markeredgecolor="black",
                markeredgewidth=0.4,
                capsize=3,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(x)
        axis.set_xticklabels([f"{value:g}" for value in x])
        axis.set_ylabel(specification[3])
        axis.grid(alpha=0.18, which="both")
        axis.tick_params(direction="in", top=True, right=True)
    for axis in axes[-1]:
        axis.set_xlabel(r"box size $L\;[\mathrm{pc}]$")
    fig.suptitle("Box-size convergence: temporal medians and 16th--84th percentiles")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def calculate_box_size_time_mean_power2d(data, output_dir, bounds, overwrite=False):
    """Calculate or load each differently shaped physical-grid 2D mean."""
    results = []
    mean_dir = Path(output_dir) / POWER2D_MEAN_DIRECTORY
    mean_dir.mkdir(parents=True, exist_ok=True)
    for model, archive_path in zip(
        data["model"].astype(str), data["power2d_archive"].astype(str)
    ):
        output = mean_dir / f"{model}.npz"
        if output.exists() and not overwrite:
            saved = load_spectrum_archive(output)
            if (
                str(saved["model"].item()) == model
                and np.allclose(saved["time_bounds"], bounds)
                and str(saved["power2d_archive"].item()) == archive_path
            ):
                print(f"Loading existing {output}", flush=True)
                results.append(saved)
                continue
        archive = load_spectrum_archive(archive_path)
        use = (archive["target_time"] >= bounds[0]) & (
            archive["target_time"] <= bounds[1]
        )
        mean, count, x_mode, y_mode = physical_time_mean_power_2d(
            archive["power_2d"],
            archive["kx0"],
            archive["ky"],
            archive["shear"],
            archive["box_size_xy_pc"],
            use,
        )
        saved = {
            "model": np.asarray(model),
            "time_bounds": np.asarray(bounds),
            "time_sample_count": np.asarray(np.count_nonzero(use)),
            "mean_power_2d": mean.astype(np.float32),
            "mode_sample_count": count.astype(np.int32),
            "kx_mode": x_mode,
            "ky_mode": y_mode,
            "box_size_xy_pc": archive["box_size_xy_pc"],
            "power2d_archive": np.asarray(archive_path),
            "power_unit": np.asarray("pc^2"),
            "physical_k_mapping": np.asarray("kx=kx0+shear*ky"),
        }
        _atomic_savez(output, **saved)
        print(f"Wrote {output}", flush=True)
        results.append(saved)
        del archive
    return results


def plot_box_size_time_mean_power2d(
    results, output, *, mode_limit=DEFAULT_POWER2D_MODE_LIMIT, dpi=180
):
    """Plot differently sized 2D Fourier grids on shared central-mode axes."""
    displayed = []
    for result in results:
        x_mode = np.asarray(result["kx_mode"], dtype=float)
        y_mode = np.asarray(result["ky_mode"], dtype=float)
        values = np.asarray(result["mean_power_2d"], dtype=float).copy()
        values[np.argmin(np.abs(y_mode)), np.argmin(np.abs(x_mode))] = np.nan
        displayed.append(
            values[np.ix_(np.abs(y_mode) <= mode_limit, np.abs(x_mode) <= mode_limit)]
        )
    limits = _positive_limits(
        np.concatenate([item.ravel() for item in displayed]),
        percentiles=(2.0, 99.8),
    )
    norm = mpl.colors.LogNorm(vmin=limits[0], vmax=limits[1])
    cmap = mpl.colormaps["plasma"].copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(
        1,
        len(results),
        figsize=(4.6 * len(results), 4.4),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for axis, result, values in zip(axes.flat, results, displayed):
        x_mode = np.asarray(result["kx_mode"], dtype=float)
        y_mode = np.asarray(result["ky_mode"], dtype=float)
        x = x_mode[np.abs(x_mode) <= mode_limit]
        y = y_mode[np.abs(y_mode) <= mode_limit]
        x_edges = _uniform_center_edges(x)
        y_edges = _uniform_center_edges(y)
        image = axis.imshow(
            values,
            origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        size = float(np.min(result["box_size_xy_pc"]))
        axis.set_title(rf"$L={size:g}\,$pc")
        axis.set_xlabel(r"physical $k_xL_x/(2\pi)$")
        axis.grid(alpha=0.12)
    axes[0, 0].set_ylabel(r"physical $k_yL_y/(2\pi)$")
    colorbar = fig.colorbar(
        image, ax=axes, orientation="horizontal", fraction=0.07, pad=0.17
    )
    colorbar.set_label(r"$\langle P_{\delta,2{\rm D}}\rangle_t\;[\mathrm{pc}^2]$")
    bounds = results[0]["time_bounds"]
    fig.suptitle(f"Physical-coordinate 2D power: {bounds[0]:g}--{bounds[1]:g} Myr")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.22, top=0.88, wspace=0.10)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def _archive_matches(
    data,
    models,
    targets,
    k_bins,
    window,
    tukey_alpha,
    pad_factor,
):
    expected = [model.name for model in models]
    return (
        list(np.asarray(data.get("model", ())).astype(str)) == expected
        and np.array_equal(data.get("target_time"), targets)
        and np.asarray(data.get("k_centers", ())).size == int(k_bins)
        and str(np.asarray(data.get("window", "")).item()) == str(window)
        and np.isclose(float(np.asarray(data.get("tukey_alpha", np.nan))), tukey_alpha)
        and np.isclose(float(np.asarray(data.get("pad_factor", np.nan))), pad_factor)
        and all(Path(path).exists() for path in data.get("power2d_archive", ()))
    )


def render_box_size_density_spectrum(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    proj_id=DEFAULT_PROJECTION,
    output_dir=None,
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    average_bounds=DEFAULT_AVERAGE_RANGE,
    k_bins=DEFAULT_K_BINS,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
    power2d_mode_limit=DEFAULT_POWER2D_MODE_LIMIT,
    dpi=180,
    workers=1,
    overwrite=False,
    overwrite_2d=False,
):
    """Run the complete box-size density-spectrum analysis."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_box_size_models(suite, model_glob, proj_id)
    targets = np.arange(start, stop + 1, stride, dtype=int)
    archive_path = output_dir / DEFAULT_ARCHIVE_NAME
    data = None
    if archive_path.exists() and not (overwrite or overwrite_2d):
        candidate = load_spectrum_archive(archive_path)
        if _archive_matches(
            candidate,
            models,
            targets,
            k_bins,
            window,
            tukey_alpha,
            pad_factor,
        ):
            data = candidate
            print(f"Loading existing {archive_path}", flush=True)
    if data is None:
        data = analyze_box_size_power(
            models,
            output_dir=output_dir,
            proj_id=proj_id,
            start=start,
            stop=stop,
            stride=stride,
            average_bounds=average_bounds,
            k_bins=k_bins,
            window=window,
            tukey_alpha=tukey_alpha,
            pad_factor=pad_factor,
            overwrite_2d=overwrite_2d,
            workers=workers,
        )
    summary, diagnostics = box_size_diagnostic_summary(data, average_bounds)
    data = attach_box_size_diagnostics(data, summary, diagnostics, average_bounds)
    _atomic_savez(archive_path, **data)
    print(f"Wrote {archive_path}", flush=True)
    csv_path = output_dir / f"{DEFAULT_DIAGNOSTIC_NAME}.csv"
    _atomic_csv(summary, csv_path)
    print(f"Wrote {csv_path}", flush=True)
    plot_box_size_time_mean(
        data, output_dir / DEFAULT_MEAN_FIGURE, bounds=average_bounds, dpi=dpi
    )
    plot_box_size_diagnostics(
        summary, output_dir / f"{DEFAULT_DIAGNOSTIC_NAME}.png", dpi=dpi
    )
    mean_2d = calculate_box_size_time_mean_power2d(
        data,
        output_dir,
        average_bounds,
        overwrite=overwrite or overwrite_2d,
    )
    plot_box_size_time_mean_power2d(
        mean_2d,
        output_dir / DEFAULT_POWER2D_FIGURE,
        mode_limit=power2d_mode_limit,
        dpi=dpi,
    )
    return data, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    parser.add_argument("--projection", default=DEFAULT_PROJECTION)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=int, default=DEFAULT_TIME_RANGE[0])
    parser.add_argument("--stop", type=int, default=DEFAULT_TIME_RANGE[1])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--average-start", type=float, default=DEFAULT_AVERAGE_RANGE[0])
    parser.add_argument("--average-stop", type=float, default=DEFAULT_AVERAGE_RANGE[1])
    parser.add_argument("--k-bins", type=int, default=DEFAULT_K_BINS)
    parser.add_argument("--window", choices=("none", "hann", "tukey"), default="none")
    parser.add_argument("--tukey-alpha", type=float, default=0.25)
    parser.add_argument("--pad-factor", type=float, default=1.0)
    parser.add_argument(
        "--power2d-mode-limit",
        type=float,
        default=DEFAULT_POWER2D_MODE_LIMIT,
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-2d", action="store_true")
    args = parser.parse_args(argv)
    if args.start < 0 or args.stop < args.start:
        parser.error("require 0 <= --start <= --stop")
    if args.stride <= 0 or args.k_bins <= 0 or args.workers <= 0:
        parser.error("--stride, --k-bins, and --workers must be positive")
    if args.average_stop <= args.average_start:
        parser.error("--average-stop must be greater than --average-start")
    if not 0.0 <= args.tukey_alpha <= 1.0:
        parser.error("--tukey-alpha must be between zero and one")
    if args.pad_factor < 1.0 or args.power2d_mode_limit <= 0.0:
        parser.error("--pad-factor must be >= 1 and --power2d-mode-limit positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    render_box_size_density_spectrum(
        args.suite,
        model_glob=args.model_glob,
        proj_id=args.projection,
        output_dir=args.output_dir,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        average_bounds=(args.average_start, args.average_stop),
        k_bins=args.k_bins,
        window=args.window,
        tukey_alpha=args.tukey_alpha,
        pad_factor=args.pad_factor,
        power2d_mode_limit=args.power2d_mode_limit,
        dpi=args.dpi,
        workers=args.workers,
        overwrite=args.overwrite,
        overwrite_2d=args.overwrite_2d,
    )


if __name__ == "__main__":
    main()
