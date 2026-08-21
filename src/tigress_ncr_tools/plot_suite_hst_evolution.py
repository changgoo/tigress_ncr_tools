#!/usr/bin/env python3
"""Plot parameter-colored time evolution of suite-wide history diagnostics."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pathena.hst_reader import read_hst

from .plot_suite_evolution import (
    DEFAULT_MODEL_GLOB,
    DEFAULT_SFR_RANGE,
    rank_models_by_sfr,
)
from .plot_suite_hst import input_parameter
from .surface_density_stats import read_shear_parameters


DEFAULT_SUITE = Path("/tigress/changgoo/anvil/TIGRESS-NCR-suite")
DEFAULT_OUTPUT_NAME = "hst_evolution"
DEFAULT_FIGURE_NAME = "velocity_dispersions"
DEFAULT_TIME_RANGE = (0.0, 600.0)
DEFAULT_CMAP = "plasma"
HISTORY_PARAMETER_COLOR_SPECS = (
    ("omega", "log", "viridis", r"$\Omega\ [{\rm Myr}^{-1}]$"),
    (
        "stellar_midplane_density",
        "log",
        "cividis",
        r"$\Sigma_*/(2H_*)\ [M_\odot\,{\rm pc}^{-3}]$",
    ),
    ("qshear", "linear", "magma", r"$q$"),
)

# key, panel title, whole-history energy/pressure field, multiplicative factor
SPEED_QUANTITIES = (
    ("sigma_x1", r"$\sigma_1=\sqrt{2K_1/M}$", "x1KE", 2.0),
    ("sigma_x2", r"$\sigma_2=\sqrt{2\,\delta K_2/M}$", "x2dke", 2.0),
    ("sigma_x3", r"$\sigma_3=\sqrt{2K_3/M}$", "x3KE", 2.0),
    ("thermal", r"$c_{\rm th}=\sqrt{P/M}$", "P", 1.0),
    ("alfven_x1", r"$v_{A,1}=\sqrt{2\,\mathrm{ME}_1/M}$", "x1ME", 2.0),
    ("alfven_x2", r"$v_{A,2}=\sqrt{2\,\mathrm{ME}_2/M}$", "x2ME", 2.0),
    ("alfven_x3", r"$v_{A,3}=\sqrt{2\,\mathrm{ME}_3/M}$", "x3ME", 2.0),
)


def primary_history_file(model):
    """Return the single primary history file for a model."""
    files = [
        path
        for path in sorted((Path(model) / "hst").glob("*.hst"))
        if ".phase" not in path.name and ".whole" not in path.name
    ]
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected one primary history under {Path(model) / 'hst'}, found {len(files)}"
        )
    return files[0]


def whole_history_file(model):
    """Return the whole-domain phase-history file for a model."""
    files = sorted((Path(model) / "hst").glob("*.whole.hst"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected one whole history under {Path(model) / 'hst'}, found {len(files)}"
        )
    return files[0]


def discover_history_models(suite, model_glob=DEFAULT_MODEL_GLOB):
    """Return model directories with primary and whole-domain histories."""
    suite = Path(suite)
    models = []
    for model in sorted(suite.glob(model_glob)):
        if not model.is_dir():
            continue
        try:
            primary_history_file(model)
            whole_history_file(model)
        except FileNotFoundError:
            continue
        models.append(model)
    if not models:
        raise FileNotFoundError(
            f"no models under {suite} match {model_glob!r} with primary and whole histories"
        )
    return models


def characteristic_speed(numerator, mass, factor=1.0):
    """Return ``sqrt(factor * numerator / mass)`` with invalid cells masked."""
    numerator = np.asarray(numerator, dtype=float)
    mass = np.asarray(mass, dtype=float)
    if numerator.shape != mass.shape:
        raise ValueError("numerator and mass must have matching shapes")
    result = np.full(numerator.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(numerator) & np.isfinite(mass) & (numerator >= 0.0) & (mass > 0.0)
    )
    result[valid] = np.sqrt(factor * numerator[valid] / mass[valid])
    return result


def history_speeds(history):
    """Derive kinetic, thermal, and Alfvén speeds from a whole history."""
    required = {"mass"} | {item[2] for item in SPEED_QUANTITIES}
    missing = sorted(required - history.keys())
    if missing:
        raise KeyError(f"whole history lacks required fields: {', '.join(missing)}")
    mass = history["mass"]
    return {
        key: characteristic_speed(history[field], mass, factor)
        for key, _, field, factor in SPEED_QUANTITIES
    }


def model_history_parameters(model):
    """Return parameters used for alternate history-track colors."""
    _, qshear, omega = read_shear_parameters(model)
    stellar_surface_density = input_parameter(Path(model), "SurfS")
    stellar_scale_height = input_parameter(Path(model), "zstar")
    values = (omega, stellar_surface_density, stellar_scale_height, qshear)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"nonfinite history color parameters for {model}")
    if stellar_scale_height <= 0.0:
        raise ValueError(f"non-positive stellar scale height for {model}")
    return {
        "omega": float(omega),
        "stellar_midplane_density": float(
            stellar_surface_density / (2.0 * stellar_scale_height)
        ),
        "qshear": float(qshear),
    }


def sfr_colormap(mean_sfr, name=DEFAULT_CMAP, scale="log"):
    """Return a trimmed sequential colormap and normalization for mean SFR."""
    values = np.asarray(mean_sfr, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("mean SFR values must be finite and non-empty")
    if scale == "log" and np.any(values <= 0.0):
        raise ValueError("logarithmic SFR colors require positive values")
    if scale not in ("linear", "log"):
        raise ValueError("SFR color scale must be 'linear' or 'log'")

    base = mpl.colormaps[name]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{name}_trimmed", base(np.linspace(0.06, 0.90, 256))
    )
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        if scale == "log":
            vmin, vmax = vmin / 1.1, vmax * 1.1
        else:
            width = max(abs(vmin) * 0.1, 1.0)
            vmin, vmax = vmin - width, vmax + width
    norm_class = mpl.colors.LogNorm if scale == "log" else mpl.colors.Normalize
    return cmap, norm_class(vmin=vmin, vmax=vmax)


def write_model_colors(ranked, path, cmap, norm, bounds=DEFAULT_SFR_RANGE):
    """Write model ranks, mean SFR values, and plotted colors."""
    path = Path(path)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("rank", "model", "mean_sfr10", "color_hex", "time_start", "time_stop")
        )
        for rank, (model, mean_sfr) in enumerate(ranked, start=1):
            writer.writerow(
                (
                    rank,
                    model.name,
                    f"{mean_sfr:.12g}",
                    mpl.colors.to_hex(cmap(norm(mean_sfr)), keep_alpha=False),
                    bounds[0],
                    bounds[1],
                )
            )


def plot_velocity_evolution(
    ranked,
    output,
    *,
    time_bounds=DEFAULT_TIME_RANGE,
    sfr_bounds=DEFAULT_SFR_RANGE,
    history_samples=4000,
    cmap_name=DEFAULT_CMAP,
    color_scale="log",
    colorbar_label=None,
    yscale="linear",
    dpi=180,
):
    """Plot the seven requested whole-domain characteristic speeds."""
    if len(time_bounds) != 2 or time_bounds[1] <= time_bounds[0]:
        raise ValueError("time bounds must be increasing")
    if history_samples < 2:
        raise ValueError("history_samples must be at least 2")
    if yscale not in ("linear", "log"):
        raise ValueError("yscale must be 'linear' or 'log'")

    mean_sfr = [value for _, value in ranked]
    cmap, norm = sfr_colormap(mean_sfr, cmap_name, color_scale)
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16.5, 8.1),
        squeeze=False,
    )
    plot_axes = list(axes.flat[:7])
    color_axis = axes.flat[7]

    speed_min = np.inf
    speed_max = -np.inf
    # Draw low-value models first so high-value tracks remain visible.
    for model, color_value in reversed(ranked):
        history = read_hst(whole_history_file(model), max_rows=history_samples)
        time = np.asarray(history["time"], dtype=float)
        use = np.isfinite(time) & (time >= time_bounds[0]) & (time <= time_bounds[1])
        speeds = history_speeds(history)
        color = cmap(norm(color_value))
        for axis, (key, _, _, _) in zip(plot_axes, SPEED_QUANTITIES):
            values = speeds[key][use]
            finite = values[np.isfinite(values)]
            if finite.size:
                speed_min = min(speed_min, float(np.min(finite)))
                speed_max = max(speed_max, float(np.max(finite)))
            axis.plot(
                time[use],
                values,
                color=color,
                linewidth=0.85,
                alpha=0.82,
            )

    if not np.isfinite(speed_min) or not np.isfinite(speed_max):
        raise ValueError("no finite characteristic speeds in the requested time range")
    if yscale == "log":
        common_ylim = (speed_min / 1.15, speed_max * 1.15)
    else:
        common_ylim = (0.0, speed_max * 1.05)
    for index, (axis, (_, title, _, _)) in enumerate(zip(plot_axes, SPEED_QUANTITIES)):
        axis.set_title(title, fontsize=11)
        axis.set_xlim(time_bounds)
        axis.set_ylim(common_ylim)
        axis.set_yscale(yscale)
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.tick_params(direction="in", top=True, right=True)
        if index < 4:
            axis.tick_params(labelbottom=False)
        if index % 4:
            axis.tick_params(labelleft=False)
    for axis in axes[-1, :3]:
        axis.set_xlabel(r"simulation time $t$")
    axes[0, 0].set_ylabel(r"speed $[\mathrm{km\,s^{-1}}]$")
    axes[1, 0].set_ylabel(r"speed $[\mathrm{km\,s^{-1}}]$")

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=color_axis)
    if colorbar_label is None:
        colorbar_label = (
            rf"$\langle\Sigma_{{\rm SFR,10}}\rangle_{{{sfr_bounds[0]:g}-{sfr_bounds[1]:g}}}$ "
            r"$[M_\odot\,\mathrm{kpc}^{-2}\,\mathrm{yr}^{-1}]$"
        )
    colorbar.set_label(colorbar_label)
    color_axis.set_title(f"{len(ranked)} models", fontsize=10)
    fig.suptitle("Whole-domain mass-weighted characteristic speeds", fontsize=14)
    fig.text(
        0.5,
        0.012,
        r"The $x_2$ kinetic component uses $\delta K_2$ with the imposed background shear removed.",
        ha="center",
        color="0.35",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.955,
        bottom=0.09,
        top=0.92,
        wspace=0.10,
        hspace=0.16,
    )

    output = Path(output)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    return cmap, norm


def render_history_evolution(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    output_dir=None,
    time_bounds=DEFAULT_TIME_RANGE,
    sfr_bounds=DEFAULT_SFR_RANGE,
    history_samples=4000,
    cmap_name=DEFAULT_CMAP,
    color_scale="log",
    yscale="linear",
    dpi=180,
):
    """Rank models by mean SFR and write all history-evolution products."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_history_models(suite, model_glob)
    ranked = rank_models_by_sfr(models, bounds=sfr_bounds, max_rows=10000)

    png = output_dir / f"{DEFAULT_FIGURE_NAME}.png"
    cmap, norm = plot_velocity_evolution(
        ranked,
        png,
        time_bounds=time_bounds,
        sfr_bounds=sfr_bounds,
        history_samples=history_samples,
        cmap_name=cmap_name,
        color_scale=color_scale,
        yscale=yscale,
        dpi=dpi,
    )
    write_model_colors(
        ranked,
        output_dir / "model_sfr_colors.csv",
        cmap,
        norm,
        bounds=sfr_bounds,
    )
    print(f"Wrote {png}")
    print(f"Wrote {output_dir / 'model_sfr_colors.csv'}")
    parameters = {model.name: model_history_parameters(model) for model, _ in ranked}
    for field, scale, parameter_cmap, colorbar_label in HISTORY_PARAMETER_COLOR_SPECS:
        parameter_ranked = [
            (model, parameters[model.name][field]) for model, _ in ranked
        ]
        parameter_png = output_dir / f"{DEFAULT_FIGURE_NAME}_color_by_{field}.png"
        plot_velocity_evolution(
            parameter_ranked,
            parameter_png,
            time_bounds=time_bounds,
            sfr_bounds=sfr_bounds,
            history_samples=history_samples,
            cmap_name=parameter_cmap,
            color_scale=scale,
            colorbar_label=colorbar_label,
            yscale=yscale,
            dpi=dpi,
        )
        print(f"Wrote {parameter_png}")
    return png, ranked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=float, default=DEFAULT_TIME_RANGE[0])
    parser.add_argument("--stop", type=float, default=DEFAULT_TIME_RANGE[1])
    parser.add_argument("--sfr-start", type=float, default=DEFAULT_SFR_RANGE[0])
    parser.add_argument("--sfr-stop", type=float, default=DEFAULT_SFR_RANGE[1])
    parser.add_argument("--history-samples", type=int, default=4000)
    parser.add_argument("--cmap", default=DEFAULT_CMAP)
    parser.add_argument("--color-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--yscale", choices=("linear", "log"), default="linear")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.stop <= args.start:
        parser.error("--stop must be greater than --start")
    if args.sfr_stop <= args.sfr_start:
        parser.error("--sfr-stop must be greater than --sfr-start")
    if args.history_samples < 2:
        parser.error("--history-samples must be at least 2")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    render_history_evolution(
        args.suite,
        model_glob=args.model_glob,
        output_dir=args.output_dir,
        time_bounds=(args.start, args.stop),
        sfr_bounds=(args.sfr_start, args.sfr_stop),
        history_samples=args.history_samples,
        cmap_name=args.cmap,
        color_scale=args.color_scale,
        yscale=args.yscale,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
