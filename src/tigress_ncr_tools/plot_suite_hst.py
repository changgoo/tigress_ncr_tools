#!/usr/bin/env python3
"""Make quick history summaries for all models in an Athena suite."""

import argparse
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .check_suite import (
    PROBLEM,
    application_exit_code,
    discover_models,
    failure_reason,
    model_status,
    progress_mtime,
    scheduler_jobs,
)
from pathena.hst_reader import read_hst
from pathena.units import star_particle_units, tigress_units

STATUS_COLORS = {
    "RUNNING": "tab:blue",
    "COMPLETING": "tab:blue",
    "PENDING": "tab:orange",
    "CONFIGURING": "tab:orange",
    "REQUEUED": "tab:purple",
    "COMPLETE": "tab:green",
    "CANCELLED": "0.4",
    "UNKNOWN": "0.4",
}

SCATTER_TIME_RANGE_MYR = (200.0, 600.0)
NUMBER_PATTERN = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"


def status_color(status):
    """Return a consistent badge color for a suite status."""
    return "red" if status in PROBLEM else STATUS_COLORS.get(status, "0.4")


def model_history(model):
    files = [path for path in sorted((model / "hst").glob("*.hst"))
             if ".phase" not in path.name and ".whole" not in path.name]
    # ponytail: 2k samples is ample for pixels; raise this if cadence-scale structure matters.
    return read_hst(files[0], max_rows=2000) if files else None


def vertical_size(model, default=4096.0):
    scripts = sorted(model.glob("*.slurm")) + sorted(model.glob("*.pbs"))
    if not scripts:
        return default
    for script in scripts:
        text = script.read_text(errors="replace")
        lo = re.search(r"domain1/x3min=([+\-\d.eE]+)", text)
        hi = re.search(r"domain1/x3max=([+\-\d.eE]+)", text)
        if lo and hi:
            return float(hi.group(1)) - float(lo.group(1))
    return default


def input_parameter(model, name):
    """Read a problem parameter from a generated batch script or athinput."""
    override = re.compile(
        rf"problem/{re.escape(name)}=({NUMBER_PATTERN})(?:\s|$)"
    )
    assignment = re.compile(
        rf"^\s*{re.escape(name)}\s*=\s*({NUMBER_PATTERN})(?:\s|$)",
        re.MULTILINE,
    )
    for pattern, paths in (
        (override, sorted(model.glob("*.slurm")) + sorted(model.glob("*.pbs"))),
        (assignment, sorted(model.glob("athinput*"))),
    ):
        for path in paths:
            match = pattern.search(path.read_text(errors="replace"))
            if match:
                return float(match.group(1))
    return np.nan


def stellar_surface_density(model, msp, surface_unit):
    """Return input old stars plus the time-dependent star-particle column."""
    return input_parameter(model, "SurfS") + np.asarray(msp) * surface_unit


def positive(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan)


def time_range_mask(time, bounds=SCATTER_TIME_RANGE_MYR):
    """Select samples inside an inclusive physical-time interval."""
    time = np.asarray(time, dtype=float)
    return (time >= bounds[0]) & (time <= bounds[1])


def histories(suite, model_glob="*"):
    result = []
    for model in discover_models(suite, model_glob):
        try:
            data = model_history(model)
        except (OSError, ValueError) as error:
            print(f"Skipping {model.name}: {error}")
            data = None
        result.append((model, data))
    return result


def plot_dashboard(models, outfile):
    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(1, 2, width_ratios=(2.2, 1))
    history_grid = grid[0, 0].subgridspec(5, 1)
    scatter_grid = grid[0, 1].subgridspec(3, 1, hspace=0.35)
    axes = [fig.add_subplot(history_grid[0, 0])]
    axes.extend(fig.add_subplot(history_grid[row, 0], sharex=axes[0])
                for row in range(1, 5))
    gas_axis = fig.add_subplot(scatter_grid[0, 0])
    star_axis = fig.add_subplot(scatter_grid[1, 0])
    pressure_axis = fig.add_subplot(scatter_grid[2, 0])
    colors = plt.colormaps["turbo"](np.linspace(0, 1, max(len(models), 1)))
    particle_units = star_particle_units()
    time_unit = particle_units["time_myr"]
    pressure_unit = tigress_units()["pressure_over_kB"]

    for color, (model, hst) in zip(colors, models):
        if not hst or not len(hst.get("time", [])):
            continue
        label = model.name
        time = hst["time"] * time_unit
        sfr = positive(hst["sfr10"])
        pressure = positive((hst["Pth_mid"] + hst["Pturb_mid"]) * pressure_unit)
        surface_unit = particle_units["mass_msun"] * vertical_size(model)
        gas = positive(hst["mass"] * surface_unit)
        sigma_star = positive(stellar_surface_density(model, hst["msp"],
                                                      surface_unit))
        scatter_samples = time_range_mask(time)
        axes[0].plot(time, sfr, color=color, label=label)
        axes[1].plot(time, positive(hst["nmid"]), color=color)
        axes[2].plot(time, pressure, color=color)
        axes[3].plot(time, sigma_star * 1e6, color=color)
        axes[4].plot(time, gas, color=color)
        gas_axis.scatter(gas[scatter_samples], sfr[scatter_samples],
                         color=color, s=2, alpha=0.25)
        star_axis.scatter(sigma_star[scatter_samples], sfr[scatter_samples],
                          color=color, s=2, alpha=0.25)
        pressure_axis.scatter(pressure[scatter_samples], sfr[scatter_samples],
                              color=color, s=2, alpha=0.25)

    labels = [r"$\Sigma_{\rm SFR,10}$ [$M_\odot\,\mathrm{kpc}^{-2}\,\mathrm{yr}^{-1}$]",
              r"$n_{\rm mid}$ [$\mathrm{cm}^{-3}$]",
              r"$(P_{\rm th}+P_{\rm turb})/k_B$ [$\mathrm{K\,cm}^{-3}$]",
              r"$\Sigma_*$ [$M_\odot\,\mathrm{kpc}^{-2}$]",
              r"$\Sigma_{\rm gas}$ [$M_\odot\,\mathrm{pc}^{-2}$]"]
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    for axis in axes:
        axis.set_yscale("log")
    axes[-1].set_xlabel("time [Myr]")
    gas_axis.set(xlabel=r"$\Sigma_{\rm gas}$ [$M_\odot\,\mathrm{pc}^{-2}$]",
                 ylabel=r"$\Sigma_{\rm SFR,10}$ [$M_\odot\,\mathrm{kpc}^{-2}\,\mathrm{yr}^{-1}$]")
    star_axis.set(xlabel=r"$\Sigma_*=\Sigma_{\rm star,input}+\Sigma_{\rm sp}$ [$M_\odot\,\mathrm{pc}^{-2}$]",
                  ylabel=r"$\Sigma_{\rm SFR,10}$ [$M_\odot\,\mathrm{kpc}^{-2}\,\mathrm{yr}^{-1}$]")
    pressure_axis.set(xlabel=r"$(P_{\rm th}+P_{\rm turb})_{\rm mid}/k_B$ [$\mathrm{K\,cm}^{-3}$]",
                      ylabel=r"$\Sigma_{\rm SFR,10}$ [$M_\odot\,\mathrm{kpc}^{-2}\,\mathrm{yr}^{-1}$]")
    for axis in (gas_axis, star_axis, pressure_axis):
        axis.set_title(r"$200 \leq t \leq 600\ \mathrm{Myr}$")
        axis.set_xscale("log")
        axis.set_yscale("log")
        if not np.isfinite(axis.dataLim.get_points()).all():
            axis.set_xlim(1.0, 10.0)
            axis.set_ylim(1.0, 10.0)
        axis.grid(alpha=0.2)
    axes[0].legend(title="model", ncol=4, fontsize=7, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_sfr_grid(models, outfile, jobs=None):
    nmodels = max(len(models), 1)
    ncols = min(8, nmodels)
    nrows = math.ceil(nmodels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(6, 2.25 * ncols), max(3, 2.25 * nrows)),
        sharex=True, sharey=True, squeeze=False,
    )
    time_unit = star_particle_units()["time_myr"]

    jobs = jobs or {}

    def inspect(model):
        job = jobs.get(model.name)
        since = job.get("start") if job else None
        reason = failure_reason(model, since=since)
        return model_status(
            model, job, reason, progress=progress_mtime(model),
            app_exit=application_exit_code(model, since, job),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = dict(zip((model for model, _ in models),
                            pool.map(inspect, (model for model, _ in models))))
    for axis, item in zip(axes.flat, models):
        model, hst = item
        axis.set_title(model.name, fontsize=9)
        if hst and len(hst.get("time", [])):
            time = hst["time"] * time_unit
            for field, style in zip(("sfr10", "sfr40", "sfr100"), ("-", "--", ":")):
                axis.plot(time, np.log10(positive(hst[field])), style, linewidth=0.8, label=field)
        else:
            axis.text(0.5, 0.5, "no history", ha="center", va="center", transform=axis.transAxes)
        status, _ = statuses[model]
        color = status_color(status)
        axis.text(0.03, 0.95, status, color=color, weight="bold", va="top",
                  bbox={"facecolor": "white", "edgecolor": color,
                        "alpha": 0.75, "pad": 1.5},
                  transform=axis.transAxes)
        axis.grid(alpha=0.15)
    for axis in axes.flat[len(models):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("time [Myr]")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$\log_{10}\Sigma_{\rm SFR}$")
    axes.flat[0].legend(fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.95, wspace=0.08, hspace=0.22)
    fig.savefig(outfile, dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", default="/anvil/scratch/x-ckim5/TIGRESS-NCR")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model-glob", default="*",
        help="glob for model directory names (default: discover all run directories)",
    )
    parser.add_argument(
        "--scheduler", choices=("auto", "slurm", "pbs", "none"), default="auto",
        help="batch scheduler used for status badges (default: auto-detect)",
    )
    args = parser.parse_args(argv)
    output = args.output_dir or Path(args.suite)
    output.mkdir(parents=True, exist_ok=True)
    models = histories(args.suite, args.model_glob)
    plot_dashboard(models, output / "hst_summary.png")
    plot_sfr_grid(models, output / "hst_sfr_grid.png", scheduler_jobs(args.scheduler))
    print(f"Wrote {output / 'hst_summary.png'}")
    print(f"Wrote {output / 'hst_sfr_grid.png'}")


if __name__ == "__main__":
    main()
