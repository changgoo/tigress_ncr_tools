#!/usr/bin/env python3
"""Build one-point gas-column overdensity PDFs for the clean NCR suite."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathena.hst_reader import read_hst
from pathena.proj2d_reader import read_proj2d

from .plot_suite_density_spectrum import (
    EXCLUDED_MODELS,
    _atomic_csv,
    _atomic_savez,
    _finite_statistics,
    _plot_percentile_points,
)
from .plot_suite_evolution import (
    DEFAULT_MODEL_GLOB,
    DEFAULT_SFR_RANGE,
    discover_evolution_models,
    nearest_indexed_projection,
    projection_number_index,
    rank_models_by_sfr,
    short_model_name,
)
from .plot_suite_hst_evolution import (
    SPEED_QUANTITIES,
    history_speeds,
    model_history_parameters,
    sfr_colormap,
    whole_history_file,
)
from .surface_density_stats import normalized_pdf


DEFAULT_SUITE = Path("/tigress/changgoo/anvil/TIGRESS-NCR-suite")
DEFAULT_OUTPUT_NAME = "density_pdf_theta0"
DEFAULT_ARCHIVE_NAME = "density_pdfs.npz"
DEFAULT_SUMMARY_NAME = "density_pdf_widths"
DEFAULT_TIME_RANGE = (0, 600)
DEFAULT_DELTA_RANGE = (-1.0, 30.0)
DEFAULT_S_RANGE = (-6.0, 4.0)
DEFAULT_PDF_BINS = 100
DEFAULT_CMAP = "plasma"
DEFAULT_HISTORY_SAMPLES = 4000
DEFAULT_PDF_DISPLAY_FLOOR = 1.0e-4


def frame_density_pdf(frame, delta_edges, s_edges):
    """Return area-weighted delta/s PDFs and direct pixel standard deviations."""
    if not np.isclose(frame["theta"], 0.0):
        raise ValueError("density PDFs currently require theta0")
    sigma = np.asarray(frame["fields"]["nH"], dtype=float)
    if sigma.ndim != 2 or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("surface density must be a finite, positive 2D array")
    mean_sigma = float(np.mean(sigma))
    ratio = sigma.ravel() / mean_sigma
    delta = ratio - 1.0
    log_ratio = np.log(ratio)
    return {
        "pdf_delta_area": normalized_pdf(delta, delta_edges),
        "pdf_s_area": normalized_pdf(log_ratio, s_edges),
        "std_delta": float(np.std(delta)),
        "std_s": float(np.std(log_ratio)),
        "mean_sigma_code": mean_sigma,
    }


def _analyze_model_pdf(task):
    model, model_sfr, targets, proj_id, delta_edges, s_edges = task
    index = projection_number_index(model, proj_id)
    guess_offset = 0
    times = []
    delta_pdfs = []
    s_pdfs = []
    std_delta = []
    std_s = []
    mean_sigma = []
    for frame_index, target in enumerate(targets, start=1):
        path, stored_time, guess_offset = nearest_indexed_projection(
            index, float(target), guess_offset, tolerance=0.05
        )
        frame = read_proj2d(path, fields="nH")
        result = frame_density_pdf(frame, delta_edges, s_edges)
        times.append(stored_time)
        delta_pdfs.append(result["pdf_delta_area"])
        s_pdfs.append(result["pdf_s_area"])
        std_delta.append(result["std_delta"])
        std_s.append(result["std_s"])
        mean_sigma.append(result["mean_sigma_code"])
        if frame_index == 1 or frame_index % 100 == 0 or frame_index == len(targets):
            print(f"{model.name}: {frame_index}/{len(targets)} PDF maps", flush=True)
    return {
        "model": model.name,
        "mean_sfr10": model_sfr,
        "time": np.asarray(times),
        "pdf_delta_area": np.asarray(delta_pdfs),
        "pdf_s_area": np.asarray(s_pdfs),
        "std_delta_time": np.asarray(std_delta),
        "std_s_time": np.asarray(std_s),
        "mean_sigma_code": np.asarray(mean_sigma),
    }


def analyze_suite_pdfs(
    ranked,
    *,
    proj_id="theta0",
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    delta_range=DEFAULT_DELTA_RANGE,
    s_range=DEFAULT_S_RANGE,
    pdf_bins=DEFAULT_PDF_BINS,
    workers=1,
    output=None,
):
    """Calculate and optionally cache PDF time series for all ranked models."""
    if proj_id != "theta0":
        raise ValueError("density PDFs currently require theta0")
    targets = np.arange(int(start), int(stop) + 1, int(stride), dtype=int)
    delta_edges = np.linspace(delta_range[0], delta_range[1], int(pdf_bins) + 1)
    s_edges = np.linspace(s_range[0], s_range[1], int(pdf_bins) + 1)
    tasks = [
        (model, model_sfr, targets, proj_id, delta_edges, s_edges)
        for model, model_sfr in ranked
    ]
    if int(workers) == 1:
        results = [_analyze_model_pdf(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(_analyze_model_pdf, tasks))
    time = np.asarray([result["time"] for result in results])
    if np.any(np.ptp(time, axis=0) > 0.05):
        raise ValueError("model PDF sequences are not time aligned")
    data = {
        "model": np.asarray([result["model"] for result in results]),
        "mean_sfr10": np.asarray([result["mean_sfr10"] for result in results]),
        "sfr_time_bounds": np.asarray(DEFAULT_SFR_RANGE),
        "projection_id": np.asarray(proj_id),
        "field": np.asarray("nH"),
        "delta_definition": np.asarray("Sigma/<Sigma>-1"),
        "s_definition": np.asarray("ln(Sigma/<Sigma>)"),
        "pdf_weighting": np.asarray("area"),
        "pdf_normalization": np.asarray("all pixels including out-of-range tails"),
        "target_time": targets,
        "time": time,
        "delta_edges": delta_edges,
        "delta_centers": 0.5 * (delta_edges[:-1] + delta_edges[1:]),
        "s_edges": s_edges,
        "s_centers": 0.5 * (s_edges[:-1] + s_edges[1:]),
        "pdf_delta_area": np.asarray(
            [result["pdf_delta_area"] for result in results]
        ),
        "pdf_s_area": np.asarray([result["pdf_s_area"] for result in results]),
        "std_delta_time": np.asarray(
            [result["std_delta_time"] for result in results]
        ),
        "std_s_time": np.asarray([result["std_s_time"] for result in results]),
        "mean_sigma_code": np.asarray(
            [result["mean_sigma_code"] for result in results]
        ),
        "excluded_models": np.asarray(sorted(EXCLUDED_MODELS)),
    }
    if output is not None:
        _atomic_savez(output, **data)
        print(f"Wrote {output}", flush=True)
    return data


def load_pdf_archive(path):
    """Load a PDF archive into ordinary in-memory arrays."""
    with np.load(path) as saved:
        return {name: saved[name] for name in saved.files}


def gaussian_fit_from_pdf(centers, edges, density):
    """Return the moment-matched normalized Gaussian for a binned PDF."""
    centers = np.asarray(centers, dtype=float)
    widths = np.diff(np.asarray(edges, dtype=float))
    density = np.asarray(density, dtype=float)
    valid = np.isfinite(density) & (density >= 0.0)
    weights = np.where(valid, density * widths, 0.0)
    normalization = float(np.sum(weights))
    if normalization <= 0.0:
        return np.nan, np.nan, np.full(centers.shape, np.nan)
    weights /= normalization
    mean = float(np.sum(weights * centers))
    variance = float(np.sum(weights * (centers - mean) ** 2))
    if not np.isfinite(variance) or variance <= 0.0:
        return mean, np.nan, np.full(centers.shape, np.nan)
    standard_deviation = np.sqrt(variance)
    fit = np.exp(-0.5 * ((centers - mean) / standard_deviation) ** 2)
    fit /= np.sqrt(2.0 * np.pi) * standard_deviation
    return mean, float(standard_deviation), fit


def summarize_density_pdfs(data, ranked, *, bounds=DEFAULT_SFR_RANGE):
    """Return per-model width statistics, median PDFs, and Gaussian fits."""
    names = np.asarray(data["model"]).astype(str)
    expected = np.asarray([model.name for model, _ in ranked])
    if not np.array_equal(names, expected):
        raise ValueError("PDF archive and ranked model ordering differ")
    time = np.asarray(data["time"], dtype=float)
    parameters = {model.name: model_history_parameters(model) for model, _ in ranked}
    pdf_s = np.asarray(data["pdf_s_area"], dtype=float)
    pdf_delta = np.asarray(data["pdf_delta_area"], dtype=float)
    rows = []
    median_s = []
    percentile16_s = []
    percentile84_s = []
    median_delta = []
    percentile16_delta = []
    percentile84_delta = []
    gaussian_fits = []
    for index, name in enumerate(names):
        use = (time[index] >= bounds[0]) & (time[index] <= bounds[1])
        if not np.any(use):
            raise ValueError(f"{name} has no PDFs inside requested time bounds")
        delta_statistics = _finite_statistics(data["std_delta_time"][index, use])
        s_statistics = _finite_statistics(data["std_s_time"][index, use])
        delta_quantiles = np.percentile(pdf_delta[index, use], [16, 50, 84], axis=0)
        s_quantiles = np.percentile(pdf_s[index, use], [16, 50, 84], axis=0)
        fit_mean, fit_sigma, gaussian = gaussian_fit_from_pdf(
            data["s_centers"], data["s_edges"], s_quantiles[1]
        )
        model_parameters = parameters[name]
        omega = float(model_parameters["omega"])
        qshear = float(model_parameters["qshear"])
        rows.append(
            {
                "model": name,
                "mean_sfr10": float(data["mean_sfr10"][index]),
                "omega": omega,
                "kappa": np.sqrt(2.0 * (2.0 - qshear)) * omega,
                "stellar_midplane_density": float(
                    model_parameters["stellar_midplane_density"]
                ),
                "qshear": qshear,
                "std_delta_time_mean": delta_statistics[0],
                "std_delta_time_std": delta_statistics[1],
                "std_delta_time_median": delta_statistics[2],
                "std_delta_time_percentile16": delta_statistics[3],
                "std_delta_time_percentile84": delta_statistics[4],
                "std_delta_time_count": delta_statistics[5],
                "std_s_time_mean": s_statistics[0],
                "std_s_time_std": s_statistics[1],
                "std_s_time_median": s_statistics[2],
                "std_s_time_percentile16": s_statistics[3],
                "std_s_time_percentile84": s_statistics[4],
                "std_s_time_count": s_statistics[5],
                "median_s_gaussian_mean": fit_mean,
                "median_s_gaussian_sigma": fit_sigma,
                "average_start": float(bounds[0]),
                "average_stop": float(bounds[1]),
            }
        )
        percentile16_delta.append(delta_quantiles[0])
        median_delta.append(delta_quantiles[1])
        percentile84_delta.append(delta_quantiles[2])
        percentile16_s.append(s_quantiles[0])
        median_s.append(s_quantiles[1])
        percentile84_s.append(s_quantiles[2])
        gaussian_fits.append(gaussian)
    products = {
        "pdf_delta_time_percentile16": np.asarray(percentile16_delta),
        "pdf_delta_time_median": np.asarray(median_delta),
        "pdf_delta_time_percentile84": np.asarray(percentile84_delta),
        "pdf_s_time_percentile16": np.asarray(percentile16_s),
        "pdf_s_time_median": np.asarray(median_s),
        "pdf_s_time_percentile84": np.asarray(percentile84_s),
        "pdf_s_gaussian_fit": np.asarray(gaussian_fits),
    }
    return pd.DataFrame(rows), products


def attach_pdf_summary(data, summary, products):
    """Return the PDF archive augmented with per-model summary products."""
    augmented = dict(data)
    for column in summary.columns:
        if column not in ("model", "average_start", "average_stop"):
            augmented[column] = summary[column].to_numpy()
    augmented.update(products)
    augmented["diagnostic_time_bounds"] = (
        summary[["average_start", "average_stop"]].iloc[0].to_numpy(dtype=float)
    )
    augmented["instantaneous_width_definition"] = np.asarray(
        "population standard deviation measured directly from all map pixels"
    )
    augmented["gaussian_fit_definition"] = np.asarray(
        "normalized Gaussian moment-matched to the temporal median area-weighted s PDF"
    )
    return augmented


def attach_velocity_summary(
    summary,
    ranked,
    *,
    bounds=DEFAULT_SFR_RANGE,
    history_samples=DEFAULT_HISTORY_SAMPLES,
):
    """Add characteristic-speed statistics over the PDF averaging window."""
    result = summary.copy()
    expected = [model.name for model, _ in ranked]
    if result["model"].tolist() != expected:
        raise ValueError("PDF summary and ranked model ordering differ")
    statistics = {key: [] for key, _, _, _ in SPEED_QUANTITIES}
    for model, _ in ranked:
        history = read_hst(whole_history_file(model), max_rows=history_samples)
        time = np.asarray(history["time"], dtype=float)
        use = np.isfinite(time) & (time >= bounds[0]) & (time <= bounds[1])
        if not np.any(use):
            raise ValueError(f"{model.name} has no history inside requested time bounds")
        speeds = history_speeds(history)
        for key, _, _, _ in SPEED_QUANTITIES:
            statistics[key].append(_finite_statistics(speeds[key][use]))
    suffixes = ("mean", "std", "median", "percentile16", "percentile84", "count")
    for key, _, _, _ in SPEED_QUANTITIES:
        values = np.asarray(statistics[key], dtype=float)
        for index, suffix in enumerate(suffixes):
            result[f"{key}_time_{suffix}"] = values[:, index]
    return result


def pdf_display_limits(centers, density, floor=DEFAULT_PDF_DISPLAY_FLOOR):
    """Return compact x/y limits retaining bins above a PDF-density floor."""
    centers = np.asarray(centers, dtype=float)
    density = np.asarray(density, dtype=float)
    if centers.ndim != 1 or density.shape[-1] != centers.size:
        raise ValueError("PDF centers and density bins do not match")
    envelope = np.nanmax(density.reshape(-1, centers.size), axis=0)
    use = np.isfinite(envelope) & (envelope >= floor)
    if not np.any(use):
        use = np.isfinite(envelope) & (envelope > 0.0)
    if not np.any(use):
        raise ValueError("PDF has no finite positive values")
    spacing = float(np.median(np.diff(centers))) if centers.size > 1 else 1.0
    indices = np.flatnonzero(use)
    x_limits = (
        float(centers[indices[0]] - 0.5 * spacing),
        float(centers[indices[-1]] + 0.5 * spacing),
    )
    peak = float(np.nanmax(envelope[use]))
    return x_limits, (floor / 2.0, peak * 1.5)


def plot_pdf_width_correlations(summary, output, *, dpi=180):
    """Plot delta and s standard deviations against kappa, rho-star, and SFR."""
    color_values = summary["mean_sfr10"].to_numpy(dtype=float)
    cmap, norm = sfr_colormap(color_values, DEFAULT_CMAP, "log")
    x_specs = (
        ("kappa", r"$\kappa=\sqrt{2(2-q)}\,\Omega\ [{\rm Myr}^{-1}]$"),
        (
            "stellar_midplane_density",
            r"$\rho_*=\Sigma_*/(2H_*)\ [M_\odot\,{\rm pc}^{-3}]$",
        ),
        (
            "mean_sfr10",
            r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
            r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$",
        ),
    )
    y_specs = (
        (
            "std_delta_time_median",
            "std_delta_time_percentile16",
            "std_delta_time_percentile84",
            r"$\sigma_\delta$",
        ),
        (
            "std_s_time_median",
            "std_s_time_percentile16",
            "std_s_time_percentile84",
            r"$\sigma_s$",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), sharex="col")
    for column, (x_field, xlabel) in enumerate(x_specs):
        x = summary[x_field].to_numpy(dtype=float)
        for row, (field, low_field, high_field, ylabel) in enumerate(y_specs):
            axis = axes[row, column]
            median = summary[field].to_numpy(dtype=float)
            low = summary[low_field].to_numpy(dtype=float)
            high = summary[high_field].to_numpy(dtype=float)
            valid = (
                np.isfinite(x)
                & (x > 0.0)
                & np.isfinite(median)
                & np.isfinite(low)
                & np.isfinite(high)
            )
            _plot_percentile_points(
                axis,
                x[valid],
                median[valid],
                low[valid],
                high[valid],
                color_values[valid],
                cmap,
                norm,
            )
            axis.set_xscale("log")
            if row == 1:
                axis.set_xlabel(xlabel)
            if column == 0:
                axis.set_ylabel(ylabel)
            axis.grid(alpha=0.18, which="both")
            axis.tick_params(direction="in", top=True, right=True)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = fig.add_axes((0.35, 0.075, 0.30, 0.018))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )
    fig.suptitle(
        "Gas-column PDF widths: 200--600 Myr median and 16th--84th percentiles",
        fontsize=13,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.22, top=0.92, hspace=0.28, wspace=0.22
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def plot_pdf_width_velocity_correlations(summary, output, *, dpi=180):
    """Plot both PDF widths against all seven characteristic speeds."""
    color_values = summary["mean_sfr10"].to_numpy(dtype=float)
    cmap, norm = sfr_colormap(color_values, DEFAULT_CMAP, "log")
    y_specs = (
        (
            "std_delta_time_median",
            "std_delta_time_percentile16",
            "std_delta_time_percentile84",
            r"$\sigma_\delta$",
        ),
        (
            "std_s_time_median",
            "std_s_time_percentile16",
            "std_s_time_percentile84",
            r"$\sigma_s$",
        ),
    )
    fig, axes = plt.subplots(2, 7, figsize=(21.5, 7.6), sharex="col")
    for column, (speed, title, _, _) in enumerate(SPEED_QUANTITIES):
        x = summary[f"{speed}_time_median"].to_numpy(dtype=float)
        x_low = summary[f"{speed}_time_percentile16"].to_numpy(dtype=float)
        x_high = summary[f"{speed}_time_percentile84"].to_numpy(dtype=float)
        for row, (field, low_field, high_field, ylabel) in enumerate(y_specs):
            axis = axes[row, column]
            median = summary[field].to_numpy(dtype=float)
            low = summary[low_field].to_numpy(dtype=float)
            high = summary[high_field].to_numpy(dtype=float)
            valid = np.all(
                np.isfinite([x, x_low, x_high, median, low, high]), axis=0
            )
            axis.errorbar(
                x[valid],
                median[valid],
                xerr=np.vstack((x[valid] - x_low[valid], x_high[valid] - x[valid])),
                yerr=np.vstack(
                    (median[valid] - low[valid], high[valid] - median[valid])
                ),
                fmt="none",
                ecolor="0.68",
                elinewidth=0.65,
                capsize=0,
                zorder=1,
            )
            axis.scatter(
                x[valid],
                median[valid],
                c=color_values[valid],
                cmap=cmap,
                norm=norm,
                s=26,
                edgecolor="black",
                linewidth=0.28,
                zorder=2,
            )
            axis.grid(alpha=0.18)
            axis.tick_params(direction="in", top=True, right=True, labelsize=8)
            if row == 0:
                axis.set_title(title, fontsize=10)
            if row == 1:
                axis.set_xlabel(r"speed $[\mathrm{km\,s^{-1}}]$", fontsize=9)
            if column == 0:
                axis.set_ylabel(ylabel)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = fig.add_axes((0.37, 0.075, 0.26, 0.020))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )
    fig.suptitle(
        "Gas-column PDF width versus mass-weighted characteristic speed: "
        "200--600 Myr medians and 16th--84th percentiles",
        fontsize=13,
    )
    fig.subplots_adjust(
        left=0.052, right=0.995, bottom=0.21, top=0.89, hspace=0.22, wspace=0.28
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def plot_median_pdfs(data, ranked, output, *, dpi=180):
    """Plot all-model temporal median delta and s PDFs with Gaussian s fits."""
    cmap, norm = sfr_colormap(data["mean_sfr10"], DEFAULT_CMAP, "log")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.5))
    for index in reversed(range(len(ranked))):
        color = cmap(norm(data["mean_sfr10"][index]))
        axes[0].plot(
            data["delta_centers"],
            data["pdf_delta_time_median"][index],
            color=color,
            linewidth=1.0,
            alpha=0.82,
        )
        axes[1].plot(
            data["s_centers"],
            data["pdf_s_time_median"][index],
            color=color,
            linewidth=1.0,
            alpha=0.82,
        )
        axes[1].plot(
            data["s_centers"],
            data["pdf_s_gaussian_fit"][index],
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
        )
    axes[0].set_xlabel(r"$\delta=\Sigma/\langle\Sigma\rangle-1$")
    axes[1].set_xlabel(r"$s=\ln(\Sigma/\langle\Sigma\rangle)$")
    axes[0].set_ylabel(r"area-weighted $p(\delta)$")
    axes[1].set_ylabel(r"area-weighted $p(s)$")
    for axis, centers, density in zip(
        axes,
        (data["delta_centers"], data["s_centers"]),
        (data["pdf_delta_time_median"], data["pdf_s_time_median"]),
    ):
        x_limits, y_limits = pdf_display_limits(centers, density)
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_yscale("log")
        axis.grid(alpha=0.18, which="both")
        axis.tick_params(direction="in", top=True, right=True)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = fig.add_axes((0.35, 0.12, 0.30, 0.024))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )
    fig.suptitle(
        "Gas-column PDFs: 200--600 Myr temporal medians "
        "(dashed: Gaussian fits to $p(s)$)",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.27, top=0.90, wspace=0.22)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def plot_s_pdf_fit_grid(data, ranked, output, *, dpi=180):
    """Plot each model's median s PDF and Gaussian fit in a 4-by-8 grid."""
    cmap, norm = sfr_colormap(data["mean_sfr10"], DEFAULT_CMAP, "log")
    fig, axes = plt.subplots(4, 8, figsize=(18.0, 9.8), sharex=True, sharey=True)
    positive = np.concatenate(
        (data["pdf_s_time_median"].ravel(), data["pdf_s_gaussian_fit"].ravel())
    )
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    y_limits = (max(np.min(positive) / 1.5, 1.0e-5), np.max(positive) * 1.5)
    for index, axis in enumerate(axes.flat):
        if index >= len(ranked):
            axis.axis("off")
            continue
        model, mean_sfr = ranked[index]
        color = cmap(norm(mean_sfr))
        axis.fill_between(
            data["s_centers"],
            data["pdf_s_time_percentile16"][index],
            data["pdf_s_time_percentile84"][index],
            color=color,
            alpha=0.16,
            linewidth=0.0,
        )
        axis.plot(
            data["s_centers"],
            data["pdf_s_time_median"][index],
            color=color,
            linewidth=1.25,
        )
        axis.plot(
            data["s_centers"],
            data["pdf_s_gaussian_fit"][index],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        axis.set_yscale("log")
        axis.set_ylim(y_limits)
        axis.set_title(f"{index + 1}. {short_model_name(model)}", fontsize=7.5)
        axis.grid(alpha=0.12, which="both")
        axis.tick_params(direction="in", labelsize=7)
    fig.supxlabel(r"$s=\ln(\Sigma/\langle\Sigma\rangle)$")
    fig.supylabel(r"area-weighted $p(s)$")
    fig.suptitle(
        "Median gas-column $s$ PDFs, 200--600 Myr "
        "(shading: 16th--84th percentiles; dashed: Gaussian fit)",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.07, top=0.93, hspace=0.34)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def render_suite_density_pdf(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    proj_id="theta0",
    output_dir=None,
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    sfr_bounds=DEFAULT_SFR_RANGE,
    delta_range=DEFAULT_DELTA_RANGE,
    s_range=DEFAULT_S_RANGE,
    pdf_bins=DEFAULT_PDF_BINS,
    workers=1,
    history_samples=DEFAULT_HISTORY_SAMPLES,
    dpi=180,
    overwrite=False,
):
    """Analyze, cache, and plot suite gas-column density PDFs."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [
        model
        for model in discover_evolution_models(suite, model_glob, proj_id)
        if model.name not in EXCLUDED_MODELS
    ]
    ranked = rank_models_by_sfr(models, bounds=sfr_bounds, max_rows=10000)
    archive = output_dir / DEFAULT_ARCHIVE_NAME
    if archive.exists() and not overwrite:
        print(f"Loading existing {archive}", flush=True)
        data = load_pdf_archive(archive)
        if list(data["model"]) != [model.name for model, _ in ranked]:
            raise ValueError("cached PDF model ordering does not match SFR ranking")
    else:
        data = analyze_suite_pdfs(
            ranked,
            proj_id=proj_id,
            start=start,
            stop=stop,
            stride=stride,
            delta_range=delta_range,
            s_range=s_range,
            pdf_bins=pdf_bins,
            workers=workers,
            output=archive,
        )
    summary, products = summarize_density_pdfs(data, ranked, bounds=sfr_bounds)
    summary = attach_velocity_summary(
        summary,
        ranked,
        bounds=sfr_bounds,
        history_samples=history_samples,
    )
    data = attach_pdf_summary(data, summary, products)
    _atomic_savez(archive, **data)
    print(f"Wrote {archive}", flush=True)
    _atomic_csv(summary, output_dir / f"{DEFAULT_SUMMARY_NAME}.csv")
    plot_pdf_width_correlations(
        summary, output_dir / f"{DEFAULT_SUMMARY_NAME}_correlations.png", dpi=dpi
    )
    plot_pdf_width_velocity_correlations(
        summary,
        output_dir / f"{DEFAULT_SUMMARY_NAME}_velocity_correlations.png",
        dpi=dpi,
    )
    plot_median_pdfs(data, ranked, output_dir / "density_pdf_time_median.png", dpi=dpi)
    plot_s_pdf_fit_grid(
        data, ranked, output_dir / "density_s_pdf_gaussian_fits.png", dpi=dpi
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
    parser.add_argument("--delta-min", type=float, default=DEFAULT_DELTA_RANGE[0])
    parser.add_argument("--delta-max", type=float, default=DEFAULT_DELTA_RANGE[1])
    parser.add_argument("--s-min", type=float, default=DEFAULT_S_RANGE[0])
    parser.add_argument("--s-max", type=float, default=DEFAULT_S_RANGE[1])
    parser.add_argument("--pdf-bins", type=int, default=DEFAULT_PDF_BINS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--history-samples", type=int, default=DEFAULT_HISTORY_SAMPLES)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.start < 0 or args.stop < args.start:
        parser.error("require 0 <= --start <= --stop")
    if (
        args.stride <= 0
        or args.pdf_bins <= 0
        or args.workers <= 0
        or args.history_samples < 2
    ):
        parser.error(
            "--stride, --pdf-bins, and --workers must be positive; "
            "--history-samples must be at least 2"
        )
    if args.sfr_stop <= args.sfr_start:
        parser.error("--sfr-stop must be greater than --sfr-start")
    if args.delta_max <= args.delta_min or args.s_max <= args.s_min:
        parser.error("PDF maxima must be greater than minima")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    render_suite_density_pdf(
        args.suite,
        model_glob=args.model_glob,
        proj_id=args.projection,
        output_dir=args.output_dir,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        sfr_bounds=(args.sfr_start, args.sfr_stop),
        delta_range=(args.delta_min, args.delta_max),
        s_range=(args.s_min, args.s_max),
        pdf_bins=args.pdf_bins,
        workers=args.workers,
        history_samples=args.history_samples,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
