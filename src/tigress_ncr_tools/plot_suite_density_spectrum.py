#!/usr/bin/env python3
"""Build shear-aware gas-column overdensity spectra for an NCR suite."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathena.proj2d_reader import read_proj2d

from .plot_suite_evolution import (
    DEFAULT_MODEL_GLOB,
    DEFAULT_SFR_RANGE,
    discover_evolution_models,
    make_grid_movie,
    nearest_indexed_projection,
    projection_number_index,
    rank_models_by_sfr,
)
from .plot_suite_hst_evolution import (
    HISTORY_PARAMETER_COLOR_SPECS,
    model_history_parameters,
    sfr_colormap,
    write_model_colors,
)
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
EXCLUDED_MODELS = frozenset({"R8_8pc_NCR_row0000"})
DEFAULT_DIAGNOSTIC_NAME = "density_power_spectrum_integral_scale_slope"
DEFAULT_CORRELATION_NAME = "density_power_spectrum_correlations"
DIAGNOSTIC_COLOR_SPECS = (
    (
        "mean_sfr10",
        "log",
        "plasma",
        r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$",
    ),
    *HISTORY_PARAMETER_COLOR_SPECS,
)


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
        "box_size": min(lx, ly),
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
    all_pixel_size = []
    all_box_size = []
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
        pixel_size = None
        box_size = None
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
            current_pixel_size = max(
                float(frame["x_spacing"]), float(frame["y_spacing"])
            )
            if pixel_size is None:
                pixel_size = current_pixel_size
            elif not np.isclose(pixel_size, current_pixel_size):
                raise ValueError(f"projection grid spacing changes within {model}")
            if box_size is None:
                box_size = result["box_size"]
            elif not np.isclose(box_size, result["box_size"]):
                raise ValueError(f"projection box size changes within {model}")
            times.append(stored_time)
            powers.append(result["power"])
            counts.append(result["mode_count"])
            means.append(result["mean_sigma"])
            remap_times.append(result["remap_time"])
            shears.append(result["shear"])
            negative.append(result["has_negative_sigma"])
            if (
                frame_index == 1
                or frame_index % 100 == 0
                or frame_index == len(targets)
            ):
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
        all_pixel_size.append(pixel_size)
        all_box_size.append(box_size)
        parameter_sources.append(str(parameter_path))

    time = np.asarray(all_time)
    if np.any(np.ptp(time, axis=0) > 0.05):
        bad = np.flatnonzero(np.ptp(time, axis=0) > 0.05)
        raise ValueError(
            f"models are not time-aligned at target indices {bad.tolist()}"
        )
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
        "pixel_size_pc": np.asarray(all_pixel_size),
        "box_size_pc": np.asarray(all_box_size),
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


def integral_scale(k_edges, power):
    """Return the energy-weighted wavelength integral scale in parsecs."""
    edges = np.asarray(k_edges, dtype=float)
    power = np.asarray(power, dtype=float)
    if edges.ndim != 1 or power.shape != (edges.size - 1,):
        raise ValueError("power must match the one-dimensional k-bin edges")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("k-bin edges must be strictly increasing")
    k = np.sqrt(edges[:-1] * edges[1:])
    dk = np.diff(edges)
    valid = np.isfinite(power) & (power > 0.0) & np.isfinite(k) & (k > 0.0)
    if not np.any(valid):
        return np.nan
    energy = k[valid] * power[valid] / (2.0 * np.pi)
    denominator = np.sum(energy * dk[valid])
    numerator = np.sum(energy * (2.0 * np.pi / k[valid]) * dk[valid])
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan
    return float(numerator / denominator)


def spectral_slope_alpha(k, power, pixel_size, scale):
    """Fit ``P_delta proportional to k^-alpha`` for 10 dx < lambda < scale."""
    k = np.asarray(k, dtype=float)
    power = np.asarray(power, dtype=float)
    wavelength = 2.0 * np.pi / k
    valid = (
        np.isfinite(k)
        & (k > 0.0)
        & np.isfinite(power)
        & (power > 0.0)
        & (wavelength > 10.0 * float(pixel_size))
        & (wavelength < float(scale))
    )
    count = int(np.count_nonzero(valid))
    if count < 3:
        return np.nan, count
    slope, _ = np.polyfit(np.log(k[valid]), np.log(power[valid]), 1)
    return float(-slope), count


def exclude_corrupted_archive_models(data, excluded=EXCLUDED_MODELS):
    """Return an archive copy with excluded models removed on the model axis."""
    names = np.asarray(data["model"]).astype(str)
    keep = np.asarray([name not in excluded for name in names], dtype=bool)
    removed = tuple(names[~keep])
    if not removed:
        return dict(data), removed
    filtered = {}
    for key, value in data.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == names.size:
            filtered[key] = array[keep]
        else:
            filtered[key] = array
    return filtered, removed


def spectrum_time_diagnostics(data):
    """Measure integral scale and fitted slope for every instantaneous spectrum."""
    power = np.asarray(data["power_delta"], dtype=float)
    if power.ndim != 3:
        raise ValueError("power_delta must have model, time, and k dimensions")
    k_edges = np.asarray(data["k_edges"], dtype=float)
    k = np.asarray(data["k_centers"], dtype=float)
    if power.shape[2] != k.size or k_edges.size != k.size + 1:
        raise ValueError("spectrum k coordinates do not match power_delta")
    if "pixel_size_pc" in data:
        pixel_size = np.asarray(data["pixel_size_pc"], dtype=float)
    else:
        pixel_size = np.full(power.shape[0], np.pi / k_edges[-1])
    if pixel_size.ndim == 0:
        pixel_size = np.full(power.shape[0], float(pixel_size))
    if pixel_size.shape != (power.shape[0],):
        raise ValueError("pixel_size_pc must be scalar or have one value per model")

    scales = np.full(power.shape[:2], np.nan)
    slopes = np.full(power.shape[:2], np.nan)
    fit_counts = np.zeros(power.shape[:2], dtype=int)
    for model_index in range(power.shape[0]):
        for time_index in range(power.shape[1]):
            scale = integral_scale(k_edges, power[model_index, time_index])
            slope, fit_count = spectral_slope_alpha(
                k,
                power[model_index, time_index],
                pixel_size[model_index],
                scale,
            )
            scales[model_index, time_index] = scale
            slopes[model_index, time_index] = slope
            fit_counts[model_index, time_index] = fit_count
    return {
        "integral_scale_time_pc": scales,
        "spectral_slope_alpha_time": slopes,
        "slope_fit_bin_count_time": fit_counts,
    }


def _finite_statistics(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (np.nan,) * 5 + (0,)
    mean, standard_deviation = np.mean(finite), np.std(finite)
    percentile16, median, percentile84 = np.percentile(finite, [16.0, 50.0, 84.0])
    return (
        float(mean),
        float(standard_deviation),
        float(median),
        float(percentile16),
        float(percentile84),
        int(finite.size),
    )


def spectrum_diagnostic_summary(
    data,
    ranked,
    *,
    bounds=DEFAULT_SFR_RANGE,
    model_parameters=None,
    time_diagnostics=None,
):
    """Summarize mean-spectrum and instantaneous diagnostics per clean model."""
    names = np.asarray(data["model"]).astype(str)
    expected = np.asarray([model.name for model, _ in ranked])
    if not np.array_equal(names, expected):
        raise ValueError("archive and ranked model ordering differ")
    if model_parameters is None:
        model_parameters = {
            model.name: model_history_parameters(model) for model, _ in ranked
        }
    mean_power = time_mean_power(data, bounds)
    if time_diagnostics is None:
        time_diagnostics = spectrum_time_diagnostics(data)
    k_edges = np.asarray(data["k_edges"], dtype=float)
    k = np.asarray(data["k_centers"], dtype=float)
    time = np.asarray(data["time"], dtype=float)
    if time.ndim == 1:
        time = np.broadcast_to(time, mean_power.shape[:1] + time.shape)
    if "pixel_size_pc" in data:
        pixel_size = np.asarray(data["pixel_size_pc"], dtype=float)
    else:
        pixel_size = np.full(names.size, np.pi / k_edges[-1])
    rows = []
    for index, name in enumerate(names):
        mean_spectrum_scale = integral_scale(k_edges, mean_power[index])
        mean_spectrum_alpha, mean_spectrum_fit_count = spectral_slope_alpha(
            k, mean_power[index], pixel_size[index], mean_spectrum_scale
        )
        use = (time[index] >= bounds[0]) & (time[index] <= bounds[1])
        (
            scale_mean,
            scale_std,
            scale_median,
            scale_percentile16,
            scale_percentile84,
            scale_count,
        ) = _finite_statistics(
            time_diagnostics["integral_scale_time_pc"][index, use]
        )
        (
            alpha_mean,
            alpha_std,
            alpha_median,
            alpha_percentile16,
            alpha_percentile84,
            alpha_count,
        ) = _finite_statistics(
            time_diagnostics["spectral_slope_alpha_time"][index, use]
        )
        parameters = model_parameters[name]
        omega = float(parameters["omega"])
        qshear = float(parameters["qshear"])
        kappa = np.sqrt(2.0 * (2.0 - qshear)) * omega
        rows.append(
            {
                "model": name,
                "mean_sfr10": float(data["mean_sfr10"][index]),
                "omega": omega,
                "kappa": kappa,
                "stellar_midplane_density": float(
                    parameters["stellar_midplane_density"]
                ),
                "qshear": qshear,
                "pixel_size_pc": float(pixel_size[index]),
                "integral_scale_time_mean_pc": scale_mean,
                "integral_scale_time_std_pc": scale_std,
                "integral_scale_time_median_pc": scale_median,
                "integral_scale_time_percentile16_pc": scale_percentile16,
                "integral_scale_time_percentile84_pc": scale_percentile84,
                "integral_scale_time_count": scale_count,
                "spectral_slope_alpha_time_mean": alpha_mean,
                "spectral_slope_alpha_time_std": alpha_std,
                "spectral_slope_alpha_time_median": alpha_median,
                "spectral_slope_alpha_time_percentile16": alpha_percentile16,
                "spectral_slope_alpha_time_percentile84": alpha_percentile84,
                "spectral_slope_alpha_time_count": alpha_count,
                "integral_scale_pc": mean_spectrum_scale,
                "spectral_slope_alpha": mean_spectrum_alpha,
                "slope_fit_bin_count": mean_spectrum_fit_count,
                "slope_fit_lambda_min_pc": 10.0 * float(pixel_size[index]),
                "slope_fit_lambda_max_pc": mean_spectrum_scale,
                "average_start": float(bounds[0]),
                "average_stop": float(bounds[1]),
            }
        )
    return pd.DataFrame(rows)


def attach_spectrum_diagnostics(data, summary, time_diagnostics):
    """Return an archive copy augmented with per-model spectrum diagnostics."""
    augmented = dict(data)
    columns = (
        "pixel_size_pc",
        "omega",
        "kappa",
        "stellar_midplane_density",
        "qshear",
        "integral_scale_time_mean_pc",
        "integral_scale_time_std_pc",
        "integral_scale_time_median_pc",
        "integral_scale_time_percentile16_pc",
        "integral_scale_time_percentile84_pc",
        "integral_scale_time_count",
        "spectral_slope_alpha_time_mean",
        "spectral_slope_alpha_time_std",
        "spectral_slope_alpha_time_median",
        "spectral_slope_alpha_time_percentile16",
        "spectral_slope_alpha_time_percentile84",
        "spectral_slope_alpha_time_count",
        "integral_scale_pc",
        "spectral_slope_alpha",
        "slope_fit_bin_count",
        "slope_fit_lambda_min_pc",
        "slope_fit_lambda_max_pc",
    )
    for column in columns:
        augmented[column] = summary[column].to_numpy()
    augmented.update(time_diagnostics)
    augmented["diagnostic_time_bounds"] = (
        summary[["average_start", "average_stop"]].iloc[0].to_numpy(dtype=float)
    )
    augmented["integral_scale_definition"] = np.asarray(
        "integral(E(k)*2pi/k dk)/integral(E(k) dk), E(k)=k*P_delta(k)/(2pi)"
    )
    augmented["spectral_slope_definition"] = np.asarray(
        "P_delta(k) proportional to k^-alpha for 10*pixel_size < 2pi/k < L_in"
    )
    augmented["time_diagnostic_statistic_definition"] = np.asarray(
        "mean, population standard deviation, median, and 16th/84th "
        "percentiles of finite instantaneous measurements within "
        "diagnostic_time_bounds"
    )
    augmented["excluded_models"] = np.asarray(sorted(EXCLUDED_MODELS))
    if "box_size_pc" not in augmented:
        augmented["box_size_pc"] = np.asarray(
            2.0 * np.pi / np.asarray(augmented["k_edges"], dtype=float)[0]
        )
    return augmented


def _atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_text(text, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def plot_spectrum_diagnostic_relations(
    summary,
    output,
    *,
    color_field="mean_sfr10",
    cmap_name="plasma",
    color_scale="log",
    colorbar_label=None,
    dpi=180,
):
    """Plot integral scale and spectral slope against mean SFR."""
    color_values = summary[color_field].to_numpy(dtype=float)
    cmap, norm = sfr_colormap(color_values, cmap_name, color_scale)
    x = summary["mean_sfr10"].to_numpy(dtype=float)
    specifications = (
        (
            "integral_scale_time_median_pc",
            "integral_scale_time_percentile16_pc",
            "integral_scale_time_percentile84_pc",
            r"$L_{\rm in}\ [{\rm pc}]$",
        ),
        (
            "spectral_slope_alpha_time_median",
            "spectral_slope_alpha_time_percentile16",
            "spectral_slope_alpha_time_percentile84",
            r"$\alpha\quad(P_\delta\propto k^{-\alpha})$",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharex=True)
    for axis, (field, low_field, high_field, ylabel) in zip(axes, specifications):
        y = summary[field].to_numpy(dtype=float)
        y_low = summary[low_field].to_numpy(dtype=float)
        y_high = summary[high_field].to_numpy(dtype=float)
        valid = (
            np.isfinite(x)
            & (x > 0.0)
            & np.isfinite(y)
            & np.isfinite(y_low)
            & np.isfinite(y_high)
            & (y_low <= y)
            & (y <= y_high)
        )
        _plot_percentile_points(
            axis,
            x[valid],
            y[valid],
            y_low[valid],
            y_high[valid],
            color_values[valid],
            cmap,
            norm,
        )
        axis.set_xscale("log")
        axis.set_xlabel(
            r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
            r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
        )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18, which="both")
        axis.tick_params(direction="in", top=True, right=True)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = fig.add_axes((0.35, 0.13, 0.30, 0.026))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    if colorbar_label is None:
        colorbar_label = DIAGNOSTIC_COLOR_SPECS[0][3]
    colorbar.set_label(colorbar_label)
    fig.suptitle(
        "Gas-column density integral scale and spectral slope "
        "(200--600 Myr instantaneous median and 16th--84th percentiles)",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.31, top=0.88, wspace=0.24)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def _plot_percentile_points(axis, x, median, percentile16, percentile84, color, cmap, norm):
    """Plot colored medians with asymmetric 16th--84th percentile bars."""
    for x_value, center, low, high, color_value in zip(
        x, median, percentile16, percentile84, color
    ):
        axis.errorbar(
            x_value,
            center,
            yerr=np.asarray([[center - low], [high - center]]),
            color=cmap(norm(color_value)),
            alpha=0.58,
            linewidth=1.0,
            capsize=2.0,
            zorder=1,
        )
    axis.scatter(
        x,
        median,
        c=color,
        cmap=cmap,
        norm=norm,
        s=42,
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )


def plot_spectrum_correlations(summary, output, *, dpi=180):
    """Plot spectrum diagnostics against kappa, stellar density, and SFR."""
    color_values = summary["mean_sfr10"].to_numpy(dtype=float)
    cmap, norm = sfr_colormap(color_values, "plasma", "log")
    x_specifications = (
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
    y_specifications = (
        (
            "integral_scale_time_median_pc",
            "integral_scale_time_percentile16_pc",
            "integral_scale_time_percentile84_pc",
            r"$L_{\rm in}\ [{\rm pc}]$",
        ),
        (
            "spectral_slope_alpha_time_median",
            "spectral_slope_alpha_time_percentile16",
            "spectral_slope_alpha_time_percentile84",
            r"$\alpha$",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), sharex="col")
    for column, (x_field, xlabel) in enumerate(x_specifications):
        x = summary[x_field].to_numpy(dtype=float)
        for row, (field, low_field, high_field, ylabel) in enumerate(
            y_specifications
        ):
            axis = axes[row, column]
            median = summary[field].to_numpy(dtype=float)
            percentile16 = summary[low_field].to_numpy(dtype=float)
            percentile84 = summary[high_field].to_numpy(dtype=float)
            valid = (
                np.isfinite(x)
                & (x > 0.0)
                & np.isfinite(median)
                & np.isfinite(percentile16)
                & np.isfinite(percentile84)
                & (percentile16 <= median)
                & (median <= percentile84)
            )
            _plot_percentile_points(
                axis,
                x[valid],
                median[valid],
                percentile16[valid],
                percentile84[valid],
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
    colorbar.set_label(DIAGNOSTIC_COLOR_SPECS[0][3])
    fig.suptitle(
        "Gas-column spectrum correlations: 200--600 Myr median "
        "and 16th--84th percentiles",
        fontsize=13,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.22, top=0.92, hspace=0.32, wspace=0.22
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


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
    box_size,
):
    """Create the two-panel dimensional and dimensionless spectrum figure."""
    box_size = float(box_size)
    if not np.isfinite(box_size) or box_size <= 0.0:
        raise ValueError("box_size must be finite and positive")
    mode = np.asarray(k, dtype=float) * box_size / (2.0 * np.pi)

    def reciprocal_scale(value):
        value = np.asarray(value, dtype=float)
        return np.divide(
            box_size,
            value,
            out=np.full_like(value, np.inf),
            where=value != 0.0,
        )

    fig = plt.figure(figsize=(12.6, 6.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.055), hspace=0.30)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    color_axis = fig.add_subplot(grid[1, :])
    lines = [None] * len(ranked)
    for index in reversed(range(len(ranked))):
        model, mean_sfr = ranked[index]
        color = cmap(norm(mean_sfr))
        valid = np.isfinite(power[index]) & (power[index] > 0.0)
        first = axes[0].plot(
            mode[valid], power[index, valid], color=color, linewidth=1.0, alpha=0.82
        )[0]
        second = axes[1].plot(
            mode[valid],
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
        axis.set_xlim(float(mode[0]), float(mode[-1]))
        axis.set_xlabel(r"dimensionless wavenumber $kL/(2\pi)$")
        axis.grid(alpha=0.18, which="both")
        axis.tick_params(direction="in", top=True, right=True)
        top_axis = axis.secondary_xaxis(
            "top",
            functions=(reciprocal_scale, reciprocal_scale),
        )
        top_axis.set_xlabel(r"wavelength $\lambda=2\pi/k\;[\mathrm{pc}]$")
        wavelength_limits = reciprocal_scale(np.asarray([mode[-1], mode[0]]))
        tick_exponents = np.arange(
            np.ceil(np.log2(wavelength_limits[0])),
            np.floor(np.log2(wavelength_limits[1])) + 1.0,
        )
        wavelength_ticks = 2.0**tick_exponents
        top_axis.set_xticks(wavelength_ticks)
        top_axis.set_xticklabels([f"{tick:g}" for tick in wavelength_ticks])
        top_axis.tick_params(direction="in")
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
    title_text = fig.suptitle(title, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.81, wspace=0.20)
    return fig, axes, lines, title_text


def plot_time_mean_spectrum(data, ranked, output, *, cmap_name=DEFAULT_CMAP, dpi=180):
    """Plot each model's t=200--600 mean overdensity spectrum."""
    k = np.asarray(data["k_centers"])
    power = time_mean_power(data, DEFAULT_SFR_RANGE)
    mean_sfr = np.asarray(data["mean_sfr10"])
    cmap, norm = sfr_colormap(mean_sfr, cmap_name, "log")
    dimensionless = k[None, :] ** 2 * power / (2.0 * np.pi)
    box_size = float(np.asarray(data["box_size_pc"]).flat[0])
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
        box_size=box_size,
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
    box_size = float(np.asarray(data["box_size_pc"]).flat[0])
    mode = k * box_size / (2.0 * np.pi)
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
                    box_size=box_size,
                )
            else:
                for model_index, (first, second) in enumerate(lines):
                    values = current[model_index]
                    valid = np.isfinite(values) & (values > 0.0)
                    first.set_data(mode[valid], values[valid])
                    second.set_data(
                        mode[valid],
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
    models = [
        model
        for model in discover_evolution_models(suite, model_glob, proj_id)
        if model.name not in EXCLUDED_MODELS
    ]
    ranked = rank_models_by_sfr(models, bounds=sfr_bounds, max_rows=10000)
    archive = output_dir / DEFAULT_ARCHIVE_NAME
    removed = ()
    if archive.exists() and not overwrite:
        print(f"Loading existing {archive}", flush=True)
        data = load_spectrum_archive(archive)
        data, removed = exclude_corrupted_archive_models(data)
        if removed:
            print(
                "Excluding corrupted archive models: " + ", ".join(removed),
                flush=True,
            )
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
    parameters = {model.name: model_history_parameters(model) for model, _ in ranked}
    time_diagnostics = spectrum_time_diagnostics(data)
    diagnostic_summary = spectrum_diagnostic_summary(
        data,
        ranked,
        bounds=sfr_bounds,
        model_parameters=parameters,
        time_diagnostics=time_diagnostics,
    )
    data = attach_spectrum_diagnostics(data, diagnostic_summary, time_diagnostics)
    _atomic_savez(archive, **data)
    print(f"Wrote {archive}", flush=True)
    diagnostic_csv = output_dir / f"{DEFAULT_DIAGNOSTIC_NAME}.csv"
    _atomic_csv(diagnostic_summary, diagnostic_csv)
    print(f"Wrote {diagnostic_csv}", flush=True)
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
    for field, scale, parameter_cmap, colorbar_label in DIAGNOSTIC_COLOR_SPECS:
        suffix = "" if field == "mean_sfr10" else f"_color_by_{field}"
        diagnostic_output = output_dir / f"{DEFAULT_DIAGNOSTIC_NAME}{suffix}.png"
        plot_spectrum_diagnostic_relations(
            diagnostic_summary,
            diagnostic_output,
            color_field=field,
            cmap_name=parameter_cmap,
            color_scale=scale,
            colorbar_label=colorbar_label,
            dpi=dpi,
        )
    plot_spectrum_correlations(
        diagnostic_summary,
        output_dir / f"{DEFAULT_CORRELATION_NAME}.png",
        dpi=dpi,
    )
    if movie:
        movie_manifest = output_dir / "density_power_spectrum_movie_models.txt"
        expected_manifest = "axis=kL/(2pi); top=lambda=2pi/k\n" + "\n".join(
            str(name) for name in data["model"]
        )
        movie_stale = (
            not movie_manifest.exists()
            or movie_manifest.read_text() != expected_manifest
        )
        render_spectrum_movie(
            data,
            ranked,
            output_dir,
            cmap_name=cmap_name,
            dpi=max(72, int(round(dpi * 0.78))),
            fps=fps,
            overwrite=overwrite or bool(removed) or movie_stale,
        )
        _atomic_text(expected_manifest, movie_manifest)
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
