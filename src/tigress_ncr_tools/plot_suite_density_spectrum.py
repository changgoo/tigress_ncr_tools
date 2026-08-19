#!/usr/bin/env python3
"""Build shear-aware gas-column overdensity spectra for an NCR suite."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pathena.proj2d_reader import read_proj2d

from .plot_suite_evolution import (
    DEFAULT_MODEL_GLOB,
    DEFAULT_SFR_RANGE,
    discover_evolution_models,
    make_grid_movie,
    nearest_indexed_projection,
    projection_number_index,
    rank_models_by_sfr,
    short_model_name,
)
from .plot_suite_hst_evolution import sfr_colormap, write_model_colors
from .surface_density_stats import (
    angle_averaged_power,
    default_k_edges,
    read_shear_parameters,
    residual_shear,
    shear_remap_periodic,
)


DEFAULT_SUITE = Path("/tigress/changgoo/anvil/TIGRESS-NCR-suite")
DEFAULT_OUTPUT_NAME = "density_power_spectrum_theta0"
DEFAULT_ARCHIVE_NAME = "density_power_spectra.npz"
DEFAULT_CMAP = "plasma"
DEFAULT_K_BINS = 40
DEFAULT_TIME_RANGE = (0, 600)


def overdensity_power(
    frame,
    qshear,
    omega,
    *,
    k_edges=None,
    k_bins=DEFAULT_K_BINS,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
):
    """Return the shear-aware spectrum of ``Sigma/<Sigma> - 1`` for one map.

    The map is first transformed to periodic shearing coordinates. Fourier
    modes are then assigned their instantaneous physical wavenumbers through
    ``kx = kx0 + q Omega t_remap ky`` before annular averaging.
    """
    if not np.isclose(frame["theta"], 0.0):
        raise ValueError("density power spectra currently require theta0")
    sigma = np.asarray(frame["fields"]["nH"], dtype=float)
    if sigma.ndim != 2 or not np.all(np.isfinite(sigma)):
        raise ValueError("surface density must be a finite 2D array")

    x = np.asarray(frame["x_centers"], dtype=float)
    y = np.asarray(frame["y_centers"], dtype=float)
    lx = float(frame["x_edges"][-1] - frame["x_edges"][0])
    ly = float(frame["y_edges"][-1] - frame["y_edges"][0])
    remap_time, shear = residual_shear(frame["time"], qshear, omega, lx, ly)
    remapped = shear_remap_periodic(sigma, x, frame["y_spacing"], shear)
    mean_sigma = float(np.mean(remapped))
    if not np.isfinite(mean_sigma) or mean_sigma <= 0.0:
        raise ValueError("surface-density mean must be finite and positive")
    delta = remapped / mean_sigma - 1.0
    if k_edges is None:
        k_edges = default_k_edges(x, y, bins=k_bins)
    power, count = angle_averaged_power(
        delta,
        frame["x_spacing"],
        frame["y_spacing"],
        shear,
        k_edges,
        window=window,
        tukey_alpha=tukey_alpha,
        pad_factor=pad_factor,
    )
    return {
        "power": power,
        "mode_count": count,
        "k_edges": np.asarray(k_edges),
        "mean_sigma": mean_sigma,
        "remap_time": remap_time,
        "shear": shear,
        "has_negative_sigma": bool(np.any(sigma < 0.0)),
    }


def _atomic_savez(path, **data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(path)


def analyze_suite_power(
    ranked,
    *,
    proj_id="theta0",
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    k_bins=DEFAULT_K_BINS,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
    output=None,
):
    """Calculate and optionally cache ``P_delta(t,k)`` for every model."""
    if proj_id != "theta0":
        raise ValueError("shear-aware density spectra currently require theta0")
    targets = np.arange(int(start), int(stop) + 1, int(stride), dtype=int)
    if targets.size == 0:
        raise ValueError("the requested time range contains no outputs")

    model_names = []
    mean_sfr = []
    all_time = []
    all_power = []
    all_count = []
    all_mean_sigma = []
    all_remap_time = []
    all_shear = []
    all_negative = []
    all_qshear = []
    all_omega = []
    parameter_sources = []
    common_k_edges = None

    for model_index, (model, model_sfr) in enumerate(ranked, start=1):
        parameter_path, qshear, omega = read_shear_parameters(model)
        index = projection_number_index(model, proj_id)
        guess_offset = 0
        times = []
        powers = []
        counts = []
        means = []
        remap_times = []
        shears = []
        negative = []
        for frame_index, target in enumerate(targets, start=1):
            path, stored_time, guess_offset = nearest_indexed_projection(
                index,
                float(target),
                guess_offset,
                tolerance=0.05,
            )
            frame = read_proj2d(path, fields="nH")
            if not np.isclose(frame["time"], stored_time):
                raise ValueError(f"metadata time changed while reading {path}")
            result = overdensity_power(
                frame,
                qshear,
                omega,
                k_edges=common_k_edges,
                k_bins=k_bins,
                window=window,
                tukey_alpha=tukey_alpha,
                pad_factor=pad_factor,
            )
            if common_k_edges is None:
                common_k_edges = result["k_edges"]
            times.append(stored_time)
            powers.append(result["power"])
            counts.append(result["mode_count"])
            means.append(result["mean_sigma"])
            remap_times.append(result["remap_time"])
            shears.append(result["shear"])
            negative.append(result["has_negative_sigma"])
            if frame_index == 1 or frame_index % 100 == 0 or frame_index == len(targets):
                print(
                    f"{model.name}: {frame_index}/{len(targets)} "
                    f"({model_index}/{len(ranked)} models)",
                    flush=True,
                )

        model_names.append(model.name)
        mean_sfr.append(model_sfr)
        all_time.append(times)
        all_power.append(powers)
        all_count.append(counts)
        all_mean_sigma.append(means)
        all_remap_time.append(remap_times)
        all_shear.append(shears)
        all_negative.append(negative)
        all_qshear.append(qshear)
        all_omega.append(omega)
        parameter_sources.append(str(parameter_path))

    time = np.asarray(all_time)
    if np.any(np.ptp(time, axis=0) > 0.05):
        bad = np.flatnonzero(np.ptp(time, axis=0) > 0.05)
        raise ValueError(f"models are not time-aligned at target indices {bad.tolist()}")
    power = np.asarray(all_power)
    count = np.asarray(all_count)
    data = {
        "model": np.asarray(model_names),
        "mean_sfr10": np.asarray(mean_sfr),
        "sfr_time_bounds": np.asarray(DEFAULT_SFR_RANGE),
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
        "target_time": targets,
        "time": time,
        "k_edges": common_k_edges,
        "k_centers": np.sqrt(common_k_edges[:-1] * common_k_edges[1:]),
        "power_delta": power,
        "dimensionless_power": (
            np.sqrt(common_k_edges[:-1] * common_k_edges[1:])[None, None, :] ** 2
            * power
            / (2.0 * np.pi)
        ),
        "mode_count": count,
        "mean_sigma_code": np.asarray(all_mean_sigma),
        "remap_time": np.asarray(all_remap_time),
        "shear": np.asarray(all_shear),
        "has_negative_sigma": np.asarray(all_negative),
        "qshear": np.asarray(all_qshear),
        "omega_kms_per_pc": np.asarray(all_omega),
        "parameter_source": np.asarray(parameter_sources),
        "window": np.asarray(window),
        "tukey_alpha": np.asarray(tukey_alpha),
        "pad_factor": np.asarray(pad_factor),
    }
    if output is not None:
        _atomic_savez(output, **data)
        print(f"Wrote {output}", flush=True)
    return data


def load_spectrum_archive(path):
    """Load a spectrum archive into ordinary in-memory arrays."""
    with np.load(path) as saved:
        return {name: saved[name] for name in saved.files}


def _positive_limits(values, percentiles=(0.5, 99.5)):
    values = np.asarray(values, dtype=float)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        raise ValueError("spectrum has no finite positive values")
    low, high = np.percentile(positive, percentiles)
    return float(low / 1.5), float(high * 1.5)


def time_mean_power(data, bounds=DEFAULT_SFR_RANGE):
    """Return the arithmetic time mean of each model spectrum in bounds."""
    power = np.asarray(data["power_delta"], dtype=float)
    time = np.asarray(data["time"], dtype=float)
    result = np.full((power.shape[0], power.shape[2]), np.nan)
    for index in range(power.shape[0]):
        use = (time[index] >= bounds[0]) & (time[index] <= bounds[1])
        if not np.any(use):
            raise ValueError(
                f"{data['model'][index]} has no spectra in [{bounds[0]}, {bounds[1]}]"
            )
        selected = power[index, use]
        finite_count = np.sum(np.isfinite(selected), axis=0)
        np.divide(
            np.nansum(selected, axis=0),
            finite_count,
            out=result[index],
            where=finite_count > 0,
        )
    return result


def create_spectrum_figure(
    k,
    power,
    ranked,
    *,
    title,
    cmap,
    norm,
    power_limits,
    dimensionless_limits,
):
    """Create the two-panel dimensional and dimensionless spectrum figure."""
    fig = plt.figure(figsize=(12.6, 5.7))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.055), hspace=0.30)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    color_axis = fig.add_subplot(grid[1, :])
    lines = [None] * len(ranked)
    for index in reversed(range(len(ranked))):
        model, mean_sfr = ranked[index]
        color = cmap(norm(mean_sfr))
        valid = np.isfinite(power[index]) & (power[index] > 0.0)
        first = axes[0].plot(
            k[valid], power[index, valid], color=color, linewidth=1.0, alpha=0.82
        )[0]
        second = axes[1].plot(
            k[valid],
            k[valid] ** 2 * power[index, valid] / (2.0 * np.pi),
            color=color,
            linewidth=1.0,
            alpha=0.82,
        )[0]
        lines[index] = (first, second)
    axes[0].set_ylabel(r"$P_\delta(k)\;[\mathrm{pc}^2]$")
    axes[1].set_ylabel(r"$k^2P_\delta(k)/(2\pi)$")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(float(k[0]), float(k[-1]))
        axis.set_xlabel(r"physical wavenumber $k\;[\mathrm{pc}^{-1}]$")
        axis.grid(alpha=0.18, which="both")
        axis.tick_params(direction="in", top=True, right=True)
    # Apply explicit limits after selecting log scales so a uniform first frame
    # cannot leave either axis at Matplotlib's empty-data defaults.
    axes[0].set_ylim(power_limits)
    axes[1].set_ylim(dimensionless_limits)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
        r"$[M_\odot\,\mathrm{kpc}^{-2}\,\mathrm{yr}^{-1}]$"
    )
    title_text = fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.90, wspace=0.20)
    return fig, axes, lines, title_text


def plot_time_mean_spectrum(data, ranked, output, *, cmap_name=DEFAULT_CMAP, dpi=180):
    """Plot each model's t=200--600 mean overdensity spectrum."""
    k = np.asarray(data["k_centers"])
    power = time_mean_power(data, DEFAULT_SFR_RANGE)
    mean_sfr = np.asarray(data["mean_sfr10"])
    cmap, norm = sfr_colormap(mean_sfr, cmap_name, "log")
    dimensionless = k[None, :] ** 2 * power / (2.0 * np.pi)
    fig, _, _, _ = create_spectrum_figure(
        k,
        power,
        ranked,
        title=(
            r"Shear-aware gas-column overdensity spectrum: "
            r"time mean over $200\leq t\leq600$"
        ),
        cmap=cmap,
        norm=norm,
        power_limits=_positive_limits(power),
        dimensionless_limits=_positive_limits(dimensionless),
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)
    return cmap, norm


def render_spectrum_movie(
    data,
    ranked,
    output_dir,
    *,
    cmap_name=DEFAULT_CMAP,
    dpi=140,
    fps=30.0,
    overwrite=False,
):
    """Render and encode the full ``P_delta(k,t)`` evolution."""
    output_dir = Path(output_dir)
    k = np.asarray(data["k_centers"])
    power = np.asarray(data["power_delta"])
    dimensionless = np.asarray(data["dimensionless_power"])
    cmap, norm = sfr_colormap(data["mean_sfr10"], cmap_name, "log")
    power_limits = _positive_limits(power)
    dimensionless_limits = _positive_limits(dimensionless)
    fig = axes = lines = title_text = None
    try:
        for time_index, target in enumerate(np.asarray(data["target_time"], dtype=int)):
            output = output_dir / f"density_power_spectrum.{target:04d}.png"
            if output.exists() and not overwrite:
                print(f"Skipping existing {output}", flush=True)
                continue
            current = power[:, time_index, :]
            if fig is None:
                fig, axes, lines, title_text = create_spectrum_figure(
                    k,
                    current,
                    ranked,
                    title=rf"Shear-aware gas-column overdensity spectrum: $t={target:g}$",
                    cmap=cmap,
                    norm=norm,
                    power_limits=power_limits,
                    dimensionless_limits=dimensionless_limits,
                )
            else:
                for model_index, (first, second) in enumerate(lines):
                    values = current[model_index]
                    valid = np.isfinite(values) & (values > 0.0)
                    first.set_data(k[valid], values[valid])
                    second.set_data(
                        k[valid],
                        k[valid] ** 2 * values[valid] / (2.0 * np.pi),
                    )
                title_text.set_text(
                    rf"Shear-aware gas-column overdensity spectrum: $t={target:g}$"
                )
            fig.savefig(output, dpi=dpi, facecolor="white")
            print(f"Wrote {output}", flush=True)
    finally:
        if fig is not None:
            plt.close(fig)
    movie_path = output_dir / "density_power_spectrum_evolution.mp4"
    make_grid_movie(output_dir, "density_power_spectrum", movie_path, fps)
    return movie_path


def render_suite_density_spectrum(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    proj_id="theta0",
    output_dir=None,
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    sfr_bounds=DEFAULT_SFR_RANGE,
    k_bins=DEFAULT_K_BINS,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
    cmap_name=DEFAULT_CMAP,
    dpi=180,
    overwrite=False,
    movie=False,
    fps=30.0,
):
    """Analyze, cache, and plot suite gas-column overdensity spectra."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_evolution_models(suite, model_glob, proj_id)
    ranked = rank_models_by_sfr(models, bounds=sfr_bounds, max_rows=10000)
    archive = output_dir / DEFAULT_ARCHIVE_NAME
    if archive.exists() and not overwrite:
        print(f"Loading existing {archive}", flush=True)
        data = load_spectrum_archive(archive)
        if list(data["model"]) != [model.name for model, _ in ranked]:
            raise ValueError("cached model ordering does not match current SFR ranking")
    else:
        data = analyze_suite_power(
            ranked,
            proj_id=proj_id,
            start=start,
            stop=stop,
            stride=stride,
            k_bins=k_bins,
            window=window,
            tukey_alpha=tukey_alpha,
            pad_factor=pad_factor,
            output=archive,
        )
    summary = output_dir / "density_power_spectrum_time_mean.png"
    cmap, norm = plot_time_mean_spectrum(
        data, ranked, summary, cmap_name=cmap_name, dpi=dpi
    )
    write_model_colors(
        ranked,
        output_dir / "model_sfr_colors.csv",
        cmap,
        norm,
        bounds=sfr_bounds,
    )
    if movie:
        render_spectrum_movie(
            data,
            ranked,
            output_dir,
            cmap_name=cmap_name,
            dpi=max(72, int(round(dpi * 0.78))),
            fps=fps,
            overwrite=overwrite,
        )
    return data, ranked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    parser.add_argument("--projection", default="theta0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=int, default=DEFAULT_TIME_RANGE[0])
    parser.add_argument("--stop", type=int, default=DEFAULT_TIME_RANGE[1])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sfr-start", type=float, default=DEFAULT_SFR_RANGE[0])
    parser.add_argument("--sfr-stop", type=float, default=DEFAULT_SFR_RANGE[1])
    parser.add_argument("--k-bins", type=int, default=DEFAULT_K_BINS)
    parser.add_argument("--window", choices=("none", "hann", "tukey"), default="none")
    parser.add_argument("--tukey-alpha", type=float, default=0.25)
    parser.add_argument("--pad-factor", type=float, default=1.0)
    parser.add_argument("--cmap", default=DEFAULT_CMAP)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.start < 0 or args.stop < args.start:
        parser.error("require 0 <= --start <= --stop")
    if args.stride <= 0 or args.k_bins <= 0:
        parser.error("--stride and --k-bins must be positive")
    if args.sfr_stop <= args.sfr_start:
        parser.error("--sfr-stop must be greater than --sfr-start")
    if not 0.0 <= args.tukey_alpha <= 1.0:
        parser.error("--tukey-alpha must be between zero and one")
    if args.pad_factor < 1.0:
        parser.error("--pad-factor must be at least one")
    if args.dpi <= 0 or args.fps <= 0:
        parser.error("--dpi and --fps must be positive")
    render_suite_density_spectrum(
        args.suite,
        model_glob=args.model_glob,
        proj_id=args.projection,
        output_dir=args.output_dir,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        sfr_bounds=(args.sfr_start, args.sfr_stop),
        k_bins=args.k_bins,
        window=args.window,
        tukey_alpha=args.tukey_alpha,
        pad_factor=args.pad_factor,
        cmap_name=args.cmap,
        dpi=args.dpi,
        overwrite=args.overwrite,
        movie=args.movie,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
