#!/usr/bin/env python3
"""Compare PDF widths and power-spectrum diagnostics among projected tracers."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .correlation_parameters import ENVIRONMENT_PARAMETER_SPECS
from .plot_suite_density_spectrum import _atomic_csv
from .plot_suite_hst_evolution import sfr_colormap


DEFAULT_SUITE = Path("/tigress/changgoo/anvil/TIGRESS-NCR-suite")
DEFAULT_OUTPUT_NAME = "tracer_correlations_theta0"
DEFAULT_CMAP = "plasma"


@dataclass(frozen=True)
class TracerSpec:
    key: str
    label: str
    pdf_summary: str
    spectrum_summary: str


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    median: str
    percentile16: str
    percentile84: str
    log: bool = False


TRACERS = (
    TracerSpec(
        "gas",
        "gas",
        "density_pdf_theta0/density_pdf_widths.csv",
        "density_power_spectrum_theta0/"
        "density_power_spectrum_integral_scale_slope.csv",
    ),
    TracerSpec(
        "hi",
        "H I",
        "hi_pdf_theta0/hi_pdf_widths.csv",
        "hi_power_spectrum_theta0/hi_power_spectrum_integral_scale_slope.csv",
    ),
    TracerSpec(
        "em",
        "EM",
        "em_pdf_theta0/em_pdf_widths.csv",
        "em_power_spectrum_theta0/em_power_spectrum_integral_scale_slope.csv",
    ),
)
TRACER_BY_KEY = {tracer.key: tracer for tracer in TRACERS}
TRACER_PAIRS = (("gas", "hi"), ("gas", "em"), ("hi", "em"))
PDF_METRICS = (
    MetricSpec(
        "sigma_delta",
        r"$\sigma_\delta$",
        "std_delta_time_median",
        "std_delta_time_percentile16",
        "std_delta_time_percentile84",
        True,
    ),
    MetricSpec(
        "sigma_s",
        r"$\sigma_s$",
        "std_s_time_median",
        "std_s_time_percentile16",
        "std_s_time_percentile84",
        True,
    ),
)
SPECTRUM_METRICS = (
    MetricSpec(
        "integral_scale",
        r"$L_{\rm in}$ [pc]",
        "integral_scale_time_median_pc",
        "integral_scale_time_percentile16_pc",
        "integral_scale_time_percentile84_pc",
        True,
    ),
    MetricSpec(
        "spectral_slope",
        r"$\alpha$",
        "spectral_slope_alpha_time_median",
        "spectral_slope_alpha_time_percentile16",
        "spectral_slope_alpha_time_percentile84",
    ),
    MetricSpec(
        "quadrupole_amplitude",
        r"$A_{2,64-256}$",
        "anisotropy_band_amplitude_time_median",
        "anisotropy_band_amplitude_time_percentile16",
        "anisotropy_band_amplitude_time_percentile84",
    ),
)


def load_tracer_summaries(suite, family):
    """Load and model-align the three existing tracer summary CSVs."""
    if family not in ("pdf", "spectrum"):
        raise ValueError("family must be 'pdf' or 'spectrum'")
    suite = Path(suite)
    summaries = {}
    for tracer in TRACERS:
        relative = tracer.pdf_summary if family == "pdf" else tracer.spectrum_summary
        path = suite / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing {family} tracer summary: {path}")
        frame = pd.read_csv(path)
        if frame["model"].duplicated().any():
            raise ValueError(f"duplicate models in {path}")
        summaries[tracer.key] = frame

    model_order = summaries["gas"]["model"].astype(str).tolist()
    expected = set(model_order)
    for tracer in TRACERS:
        frame = summaries[tracer.key]
        if set(frame["model"].astype(str)) != expected:
            raise ValueError(f"{tracer.label} {family} model set differs from gas")
        summaries[tracer.key] = (
            frame.set_index("model").loc[model_order].reset_index()
        )
    reference = summaries["gas"]
    for tracer in TRACERS[1:]:
        frame = summaries[tracer.key]
        if not np.allclose(
            reference["mean_sfr10"], frame["mean_sfr10"], rtol=1e-10
        ):
            raise ValueError(f"{tracer.label} {family} SFR values differ from gas")
        for field, _, _, _ in ENVIRONMENT_PARAMETER_SPECS:
            if field == "mean_sfr10":
                continue
            if not np.allclose(reference[field], frame[field], rtol=1e-10):
                raise ValueError(
                    f"{tracer.label} {family} {field} values differ from gas"
                )
        for bound in ("average_start", "average_stop"):
            if bound in reference and bound in frame and not np.allclose(
                reference[bound], frame[bound]
            ):
                raise ValueError(
                    f"{tracer.label} {family} time bounds differ from gas"
                )
    return summaries


def _spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    count = int(np.count_nonzero(valid))
    if count < 3:
        return np.nan, count
    x_rank = pd.Series(np.asarray(x)[valid]).rank(method="average")
    y_rank = pd.Series(np.asarray(y)[valid]).rank(method="average")
    return float(x_rank.corr(y_rank)), count


def tracer_correlation_table(summaries, metrics, family):
    """Return pairwise Spearman coefficients for like tracer diagnostics."""
    rows = []
    for metric in metrics:
        for x_key, y_key in TRACER_PAIRS:
            x = summaries[x_key][metric.median].to_numpy(dtype=float)
            y = summaries[y_key][metric.median].to_numpy(dtype=float)
            coefficient, count = _spearman(x, y)
            rows.append(
                {
                    "family": family,
                    "metric": metric.key,
                    "x_tracer": x_key,
                    "y_tracer": y_key,
                    "spearman_rho": coefficient,
                    "model_count": count,
                }
            )
    return pd.DataFrame(rows)


def parameter_correlation_table(summaries, metrics, family):
    """Return tracer-diagnostic Spearman coefficients for every predictor."""
    rows = []
    for tracer in TRACERS:
        frame = summaries[tracer.key]
        for metric in metrics:
            y = frame[metric.median].to_numpy(dtype=float)
            for parameter, parameter_label, _, _ in ENVIRONMENT_PARAMETER_SPECS:
                coefficient, count = _spearman(
                    frame[parameter].to_numpy(dtype=float), y
                )
                rows.append(
                    {
                        "family": family,
                        "tracer": tracer.key,
                        "tracer_label": tracer.label,
                        "metric": metric.key,
                        "metric_label": metric.label,
                        "parameter": parameter,
                        "parameter_label": parameter_label,
                        "spearman_rho": coefficient,
                        "model_count": count,
                    }
                )
    return pd.DataFrame(rows)


def plot_parameter_correlation_matrix(
    correlations,
    metrics,
    output,
    *,
    title,
    dpi=180,
):
    """Plot an annotated tracer/diagnostic-by-parameter Spearman matrix."""
    parameter_names = [item[0] for item in ENVIRONMENT_PARAMETER_SPECS]
    parameter_labels = [item[1] for item in ENVIRONMENT_PARAMETER_SPECS]
    row_specs = [(tracer, metric) for tracer in TRACERS for metric in metrics]
    matrix = np.full((len(row_specs), len(parameter_names)), np.nan)
    for row, (tracer, metric) in enumerate(row_specs):
        for column, parameter in enumerate(parameter_names):
            match = correlations[
                (correlations["tracer"] == tracer.key)
                & (correlations["metric"] == metric.key)
                & (correlations["parameter"] == parameter)
            ]
            if len(match) == 1:
                matrix[row, column] = match["spearman_rho"].iloc[0]

    height = max(5.0, 0.62 * len(row_specs) + 1.8)
    figure, axis = plt.subplots(figsize=(10.8, height))
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(parameter_labels)), parameter_labels)
    axis.set_yticks(
        np.arange(len(row_specs)),
        [f"{tracer.label}: {metric.label}" for tracer, metric in row_specs],
    )
    axis.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    for row, column in np.ndindex(matrix.shape):
        value = matrix[row, column]
        axis.text(
            column,
            row,
            "--" if not np.isfinite(value) else f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=(
                "0.45"
                if not np.isfinite(value)
                else "white" if abs(value) > 0.55 else "black"
            ),
        )
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.045)
    colorbar.set_label(r"Spearman $\rho_s$ across models")
    figure.suptitle(title, fontsize=14, y=0.985)
    figure.subplots_adjust(left=0.23, right=0.91, bottom=0.07, top=0.86)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Wrote {output}", flush=True)


def _axis_limits(x_low, x_high, y_low, y_high, log):
    values = np.concatenate((x_low, x_high, y_low, y_high))
    values = values[np.isfinite(values)]
    if log:
        values = values[values > 0.0]
        lower, upper = np.min(np.log10(values)), np.max(np.log10(values))
        padding = max(0.05, 0.07 * (upper - lower))
        return 10 ** (lower - padding), 10 ** (upper + padding)
    lower, upper = np.min(values), np.max(values)
    padding = max(1e-6, 0.07 * (upper - lower))
    return lower - padding, upper + padding


def _plot_tracer_grid(
    summaries,
    metrics,
    output,
    *,
    title,
    note=None,
    dpi=180,
):
    color_values = summaries["gas"]["mean_sfr10"].to_numpy(dtype=float)
    cmap, norm = sfr_colormap(color_values, DEFAULT_CMAP, "log")
    figure, axes = plt.subplots(
        len(metrics),
        len(TRACER_PAIRS),
        figsize=(13.8, 4.15 * len(metrics)),
        squeeze=False,
    )
    for row, metric in enumerate(metrics):
        for column, (x_key, y_key) in enumerate(TRACER_PAIRS):
            axis = axes[row, column]
            x_frame = summaries[x_key]
            y_frame = summaries[y_key]
            x = x_frame[metric.median].to_numpy(dtype=float)
            x_low = x_frame[metric.percentile16].to_numpy(dtype=float)
            x_high = x_frame[metric.percentile84].to_numpy(dtype=float)
            y = y_frame[metric.median].to_numpy(dtype=float)
            y_low = y_frame[metric.percentile16].to_numpy(dtype=float)
            y_high = y_frame[metric.percentile84].to_numpy(dtype=float)
            valid = (
                np.isfinite(x)
                & np.isfinite(x_low)
                & np.isfinite(x_high)
                & np.isfinite(y)
                & np.isfinite(y_low)
                & np.isfinite(y_high)
                & (x_low <= x)
                & (x <= x_high)
                & (y_low <= y)
                & (y <= y_high)
            )
            if metric.log:
                valid &= (
                    (x_low > 0.0)
                    & (x > 0.0)
                    & (x_high > 0.0)
                    & (y_low > 0.0)
                    & (y > 0.0)
                    & (y_high > 0.0)
                )
            for index in np.flatnonzero(valid):
                axis.errorbar(
                    x[index],
                    y[index],
                    xerr=np.asarray(
                        [[x[index] - x_low[index]], [x_high[index] - x[index]]]
                    ),
                    yerr=np.asarray(
                        [[y[index] - y_low[index]], [y_high[index] - y[index]]]
                    ),
                    color=cmap(norm(color_values[index])),
                    alpha=0.38,
                    linewidth=0.65,
                    capsize=1.5,
                    zorder=1,
                )
            axis.scatter(
                x[valid],
                y[valid],
                c=color_values[valid],
                cmap=cmap,
                norm=norm,
                s=38,
                edgecolor="black",
                linewidth=0.35,
                zorder=2,
            )
            limits = _axis_limits(
                x_low[valid], x_high[valid], y_low[valid], y_high[valid], metric.log
            )
            axis.plot(limits, limits, color="0.35", linestyle="--", linewidth=0.8)
            axis.set_xlim(limits)
            axis.set_ylim(limits)
            if metric.log:
                axis.set_xscale("log")
                axis.set_yscale("log")
            coefficient, count = _spearman(x[valid], y[valid])
            axis.text(
                0.04,
                0.95,
                rf"$\rho_s={coefficient:+.2f}$ ($N={count}$)",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
            )
            axis.set_xlabel(
                f"{TRACER_BY_KEY[x_key].label}: {metric.label}"
            )
            axis.set_ylabel(
                f"{TRACER_BY_KEY[y_key].label}: {metric.label}"
            )
            axis.grid(alpha=0.16, which="both")
            axis.tick_params(direction="in", top=True, right=True)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = figure.add_axes((0.35, 0.055, 0.30, 0.016))
    colorbar = figure.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )
    figure.suptitle(title, fontsize=14)
    if note:
        figure.text(0.5, 0.095, note, ha="center", fontsize=9)
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.15 if note else 0.125,
        top=0.94,
        hspace=0.30,
        wspace=0.27,
    )
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Wrote {output}", flush=True)


def render_suite_tracer_correlations(
    suite,
    *,
    output_dir=None,
    dpi=180,
):
    """Render cross-tracer PDF and spectrum diagnostic comparisons."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = load_tracer_summaries(suite, "pdf")
    spectrum = load_tracer_summaries(suite, "spectrum")
    pdf_correlations = tracer_correlation_table(pdf, PDF_METRICS, "pdf")
    spectrum_correlations = tracer_correlation_table(
        spectrum, SPECTRUM_METRICS, "spectrum"
    )
    pdf_parameter_correlations = parameter_correlation_table(
        pdf, PDF_METRICS, "pdf"
    )
    spectrum_parameter_correlations = parameter_correlation_table(
        spectrum, SPECTRUM_METRICS, "spectrum"
    )
    correlation_path = output_dir / "tracer_correlation_coefficients.csv"
    _atomic_csv(
        pd.concat((pdf_correlations, spectrum_correlations), ignore_index=True),
        correlation_path,
    )
    print(f"Wrote {correlation_path}", flush=True)
    parameter_path = output_dir / "tracer_parameter_correlation_coefficients.csv"
    _atomic_csv(
        pd.concat(
            (pdf_parameter_correlations, spectrum_parameter_correlations),
            ignore_index=True,
        ),
        parameter_path,
    )
    print(f"Wrote {parameter_path}", flush=True)
    _plot_tracer_grid(
        pdf,
        PDF_METRICS,
        output_dir / "tracer_pdf_width_correlations.png",
        title=(
            "Projected-tracer PDF widths: 200--600 Myr temporal medians "
            "and 16th--84th percentiles"
        ),
        note=(
            "H I sigma_s includes the documented low-column, "
            "low-neutral-fraction component."
        ),
        dpi=dpi,
    )
    _plot_tracer_grid(
        spectrum,
        SPECTRUM_METRICS,
        output_dir / "tracer_power_spectrum_correlations.png",
        title=(
            "Projected-tracer power-spectrum diagnostics: 200--600 Myr "
            "temporal medians and 16th--84th percentiles"
        ),
        dpi=dpi,
    )
    plot_parameter_correlation_matrix(
        pdf_parameter_correlations,
        PDF_METRICS,
        output_dir / "tracer_pdf_width_parameter_correlation_matrix.png",
        title=(
            "Total-gas, H I, and EM PDF-width correlations with suite parameters"
        ),
        dpi=dpi,
    )
    plot_parameter_correlation_matrix(
        spectrum_parameter_correlations,
        SPECTRUM_METRICS,
        output_dir / "tracer_power_spectrum_parameter_correlation_matrix.png",
        title=(
            "Total-gas, H I, and EM power-spectrum correlations with suite parameters"
        ),
        dpi=dpi,
    )
    return pdf_correlations, spectrum_correlations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    render_suite_tracer_correlations(
        args.suite, output_dir=args.output_dir, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
