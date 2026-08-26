#!/usr/bin/env python3
"""Measure six-phase evolution and vertical structure from zprof outputs."""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_suite_density_spectrum import _atomic_csv, _finite_statistics
from .plot_suite_evolution import (
    DEFAULT_MODEL_GLOB,
    DEFAULT_SFR_RANGE,
    rank_models_by_sfr,
)
from .plot_suite_hst_evolution import (
    model_history_parameters,
    primary_history_file,
    sfr_colormap,
    write_model_colors,
)
from .plot_suite_prfm import _mesh_area, _profile_header, _read_profile_columns


DEFAULT_SUITE = Path("/tigress/changgoo/anvil/TIGRESS-NCR-suite")
DEFAULT_OUTPUT_NAME = "phase_evolution_zprof"
DEFAULT_TIME_RANGE = (0.0, 600.0)
DEFAULT_SUMMARY_RANGE = (400.0, 600.0)
DEFAULT_CMAP = "plasma"
REFERENCE_PHASE_INDEX = 5
PROFILE_NAME = re.compile(r"^(?P<problem>.+)\.(?P<dump>\d+)\.phase5\.zprof$")


@dataclass(frozen=True)
class PhaseDefinition:
    """One displayed phase and its Athena zprof component indices."""

    key: str
    label: str
    indices: tuple


PHASES = (
    PhaseDefinition("cold", "CNM+CMM", (7, 11)),
    PhaseDefinition("unm", "UNM", (12,)),
    PhaseDefinition("wnm", "WNM", (13,)),
    PhaseDefinition("wim", "WIM", (9, 10)),
    PhaseDefinition("whim", "WHIM", (5,)),
    PhaseDefinition("him", "HIM", (6,)),
)
PHASE_BY_KEY = {phase.key: phase for phase in PHASES}
REQUIRED_PHASE_INDICES = tuple(
    sorted({index for phase in PHASES for index in phase.indices})
)
WHOLE_FIELDS = ("z", "A", "d", "B1", "B2", "B3")
PROFILE_FIELDS = (
    "z",
    "A",
    "d",
    "M1",
    "dM2",
    "M3",
    "Ek1",
    "dEk2",
    "Ek3",
    "PB1",
    "PB2",
    "PB3",
    "dPB1",
    "dPB2",
    "dPB3",
)
VELOCITY_COMPONENTS = (
    ("x1", "M1", "Ek1"),
    ("x2", "dM2", "dEk2"),
    ("x3", "M3", "Ek3"),
)
MAGNETIC_COMPONENTS = (
    ("x1", "B1", "dPB1"),
    ("x2", "B2", "dPB2"),
    ("x3", "B3", "dPB3"),
)
SUMMARY_FIELDS = (
    "mass_fraction_box",
    "volume_fraction_box",
    "mass_fraction_hgas",
    "volume_fraction_hgas",
    "mass_scale_height_pc",
    "volume_scale_height_pc",
    "sigma_x1_box",
    "sigma_x2_box",
    "sigma_x3_box",
    "sigma_3d_box",
    "sigma_x1_hgas",
    "sigma_x2_hgas",
    "sigma_x3_hgas",
    "sigma_3d_hgas",
    "alfven_mean_3d_box",
    "alfven_perturbed_3d_box",
    "alfven_mean_3d_hgas",
    "alfven_perturbed_3d_hgas",
)


def discover_phase_models(suite, model_glob=DEFAULT_MODEL_GLOB):
    """Return suite models with primary histories and required zprof phases."""
    models = []
    for model in sorted(Path(suite).glob(model_glob)):
        if not model.is_dir():
            continue
        try:
            primary_history_file(model)
        except FileNotFoundError:
            continue
        zprof = model / "zprof"
        required = [
            next(zprof.glob(f"*.phase{index}.zprof"), None)
            for index in REQUIRED_PHASE_INDICES
        ]
        if (
            all(path is not None for path in required)
            and next(zprof.glob("*.whole.zprof"), None) is not None
        ):
            models.append(model)
    if not models:
        raise FileNotFoundError(
            f"no models under {suite} match {model_glob!r} with phase zprof output"
        )
    return models


def cell_width(z):
    """Return and validate the uniform zprof cell width."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 2 or not np.all(np.isfinite(z)):
        raise ValueError("z-profile coordinates must be a finite 1D array")
    spacing = np.diff(z)
    if not np.all(spacing > 0.0) or not np.allclose(spacing, spacing[0]):
        raise ValueError("z-profile coordinates must be strictly uniform")
    return float(spacing[0])


def slab_overlap_weights(z, half_height):
    """Return each z cell's overlap length with |z| <= half_height."""
    z = np.asarray(z, dtype=float)
    dz = cell_width(z)
    half_height = float(half_height)
    if not np.isfinite(half_height) or half_height <= 0.0:
        raise ValueError("slab half-height must be finite and positive")
    lower = z - 0.5 * dz
    upper = z + 0.5 * dz
    return np.clip(
        np.minimum(upper, half_height) - np.maximum(lower, -half_height),
        0.0,
        dz,
    )


def _safe_ratio(numerator, denominator):
    if not np.isfinite(denominator) or denominator <= 0.0:
        return np.nan
    return float(numerator / denominator)


def _rms_height(z, weight, dz):
    denominator = float(np.sum(weight * dz))
    if denominator <= 0.0:
        return np.nan
    return float(np.sqrt(np.sum(weight * z**2 * dz) / denominator))


def _variance_speed(second_moment, mean):
    variance = float(second_moment - mean**2)
    scale = max(abs(float(second_moment)), mean**2, 1.0)
    if variance < -1.0e-10 * scale:
        return np.nan
    return float(np.sqrt(max(variance, 0.0)))


def _energy_speed(energy, mass):
    if mass <= 0.0:
        return np.nan
    scale = max(abs(float(energy)), 1.0)
    if energy < -1.0e-10 * scale:
        return np.nan
    return float(np.sqrt(2.0 * max(float(energy), 0.0) / mass))


def phase_profile_moments(values, weights, mean_field):
    """Return phase velocity and split Alfvén diagnostics in one z region.

    ``mean_field`` is the whole-horizontal mean B at every z. The zprof
    perturbations are defined relative to exactly this mean, not a
    phase-conditional field mean.
    """
    weights = np.asarray(weights, dtype=float)
    mass = float(np.sum(values["d"] * weights))
    result = {"mass_code": mass, "volume_pc3": float(np.sum(values["A"] * weights))}
    velocity = []
    for label, momentum_name, energy_name in VELOCITY_COMPONENTS:
        momentum = float(np.sum(values[momentum_name] * weights))
        energy = float(np.sum(values[energy_name] * weights))
        mean = _safe_ratio(momentum, mass)
        second = _safe_ratio(2.0 * energy, mass)
        sigma = (
            _variance_speed(second, mean)
            if np.isfinite(mean) and np.isfinite(second)
            else np.nan
        )
        result[f"mean_velocity_{label}"] = mean
        result[f"sigma_{label}"] = sigma
        velocity.append(sigma)

    alfven_mean = []
    alfven_perturbed = []
    for label, mean_name, perturbed_name in MAGNETIC_COMPONENTS:
        field = np.asarray(mean_field[mean_name], dtype=float)
        mean_energy = float(np.sum(0.5 * values["A"] * np.square(field) * weights))
        perturbed_energy = float(np.sum(values[perturbed_name] * weights))
        mean_speed = _energy_speed(mean_energy, mass)
        perturbed_speed = _energy_speed(perturbed_energy, mass)
        result[f"alfven_mean_{label}"] = mean_speed
        result[f"alfven_perturbed_{label}"] = perturbed_speed
        alfven_mean.append(mean_speed)
        alfven_perturbed.append(perturbed_speed)

    result["sigma_3d"] = (
        float(np.sqrt(np.sum(np.square(velocity))))
        if np.all(np.isfinite(velocity))
        else np.nan
    )
    result["alfven_mean_3d"] = (
        float(np.sqrt(np.sum(np.square(alfven_mean))))
        if np.all(np.isfinite(alfven_mean))
        else np.nan
    )
    result["alfven_perturbed_3d"] = (
        float(np.sqrt(np.sum(np.square(alfven_perturbed))))
        if np.all(np.isfinite(alfven_perturbed))
        else np.nan
    )
    return result


def _sum_profiles(paths):
    reference_time = reference_z = None
    summed = None
    for path in paths:
        time, values = _read_profile_columns(path, PROFILE_FIELDS)
        z = values.pop("z")
        if reference_time is None:
            reference_time = float(time)
            reference_z = z
            summed = {name: np.zeros_like(value) for name, value in values.items()}
        elif not np.isclose(time, reference_time) or not np.array_equal(z, reference_z):
            raise ValueError(f"phase z-profiles are not aligned at {path}")
        for name, value in values.items():
            summed[name] += value
    return reference_time, reference_z, summed


def reduce_phase_zprof_snapshot(phase_paths, whole_path, horizontal_area):
    """Reduce one aligned dump into six phase rows and one closure row."""
    whole_time, whole = _read_profile_columns(whole_path, WHOLE_FIELDS)
    z = whole.pop("z")
    dz = cell_width(z)
    full_weights = np.full(z.shape, dz)
    whole_mass = float(np.sum(whole["d"] * full_weights))
    whole_volume = float(np.sum(whole["A"] * full_weights))
    if whole_mass <= 0.0 or whole_volume <= 0.0:
        raise ValueError("whole z-profile has non-positive mass or volume")
    expected_volume = float(horizontal_area) * dz * len(z)
    if not np.isclose(whole_volume, expected_volume, rtol=2.0e-5):
        raise ValueError("whole z-profile volume does not match the mesh geometry")
    if np.any(whole["A"] <= 0.0):
        raise ValueError("whole z-profile has non-positive horizontal area")
    mean_field = {name: whole[name] / whole["A"] for name in ("B1", "B2", "B3")}

    gas_scale_height = _rms_height(z, whole["d"], full_weights)
    slab_weights = slab_overlap_weights(z, gas_scale_height)
    whole_mass_hgas = float(np.sum(whole["d"] * slab_weights))
    whole_volume_hgas = float(np.sum(whole["A"] * slab_weights))

    rows = []
    summed_mass_box = summed_volume_box = 0.0
    summed_mass_hgas = summed_volume_hgas = 0.0
    for definition in PHASES:
        time, phase_z, values = _sum_profiles(phase_paths[definition.key])
        if not np.isclose(time, whole_time) or not np.array_equal(phase_z, z):
            raise ValueError(f"{definition.label} and whole z-profiles are not aligned")
        box = phase_profile_moments(values, full_weights, mean_field)
        hgas = phase_profile_moments(values, slab_weights, mean_field)
        summed_mass_box += box["mass_code"]
        summed_volume_box += box["volume_pc3"]
        summed_mass_hgas += hgas["mass_code"]
        summed_volume_hgas += hgas["volume_pc3"]
        row = {
            "time": float(whole_time),
            "phase": definition.key,
            "phase_label": definition.label,
            "zprof_indices": "+".join(str(index) for index in definition.indices),
            "gas_scale_height_pc": gas_scale_height,
            "mass_fraction_box": _safe_ratio(box["mass_code"], whole_mass),
            "volume_fraction_box": _safe_ratio(box["volume_pc3"], whole_volume),
            "mass_fraction_hgas": _safe_ratio(hgas["mass_code"], whole_mass_hgas),
            "volume_fraction_hgas": _safe_ratio(hgas["volume_pc3"], whole_volume_hgas),
            "mass_scale_height_pc": _rms_height(z, values["d"], full_weights),
            "volume_scale_height_pc": _rms_height(z, values["A"], full_weights),
        }
        for region, moments in (("box", box), ("hgas", hgas)):
            for name, value in moments.items():
                row[f"{name}_{region}"] = value
        rows.append(row)

    closure = {
        "time": float(whole_time),
        "gas_scale_height_pc": gas_scale_height,
        "selected_mass_fraction_box": summed_mass_box / whole_mass,
        "selected_volume_fraction_box": summed_volume_box / whole_volume,
        "uim_residual_mass_fraction_box": 1.0 - summed_mass_box / whole_mass,
        "uim_residual_volume_fraction_box": 1.0 - summed_volume_box / whole_volume,
        "selected_mass_fraction_hgas": summed_mass_hgas / whole_mass_hgas,
        "selected_volume_fraction_hgas": summed_volume_hgas / whole_volume_hgas,
        "uim_residual_mass_fraction_hgas": 1.0 - summed_mass_hgas / whole_mass_hgas,
        "uim_residual_volume_fraction_hgas": 1.0
        - summed_volume_hgas / whole_volume_hgas,
    }
    return rows, closure


def phase_zprof_index(model):
    """Index reference phase5 z-profiles by output number."""
    entries = []
    for path in (Path(model) / "zprof").glob("*.phase5.zprof"):
        match = PROFILE_NAME.match(path.name)
        if match is not None:
            entries.append((int(match.group("dump")), match.group("problem"), path))
    if not entries:
        raise FileNotFoundError(f"no phase5 z-profiles under {Path(model) / 'zprof'}")
    return sorted(entries)


def phase_paths_for_dump(reference_path, problem, dump):
    """Return grouped phase paths and the aligned whole path."""
    directory = Path(reference_path).parent
    grouped = {
        phase.key: [
            directory / f"{problem}.{dump:04d}.phase{index}.zprof"
            for index in phase.indices
        ]
        for phase in PHASES
    }
    whole = directory / f"{problem}.{dump:04d}.whole.zprof"
    paths = [path for group in grouped.values() for path in group] + [whole]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing aligned phase z-profiles: "
            + ", ".join(str(path) for path in missing)
        )
    return grouped, whole


def _analyze_phase_model(task):
    model, mean_sfr, start, stop, stride = task
    indexed = []
    for dump, problem, reference in phase_zprof_index(model):
        time, _ = _profile_header(reference)
        if start <= time <= stop:
            indexed.append((dump, problem, reference, time))
    indexed = indexed[:: int(stride)]
    if not indexed:
        raise ValueError(
            f"{model.name} has no phase zprof dumps in {start}--{stop} Myr"
        )
    horizontal_area = _mesh_area(model)
    phase_rows = []
    closure_rows = []
    for item, (dump, problem, reference, _) in enumerate(indexed, start=1):
        grouped, whole = phase_paths_for_dump(reference, problem, dump)
        rows, closure = reduce_phase_zprof_snapshot(grouped, whole, horizontal_area)
        for row in rows:
            row.update(
                {
                    "model": model.name,
                    "mean_sfr10": float(mean_sfr),
                    "dump": int(dump),
                }
            )
        closure.update(
            {
                "model": model.name,
                "mean_sfr10": float(mean_sfr),
                "dump": int(dump),
            }
        )
        phase_rows.extend(rows)
        closure_rows.append(closure)
        if item == 1 or item % 50 == 0 or item == len(indexed):
            print(
                f"{model.name}: {item}/{len(indexed)} phase zprof dumps",
                flush=True,
            )
    return phase_rows, closure_rows


def analyze_suite_phases(
    ranked,
    *,
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    workers=1,
):
    """Reduce the selected zprof interval for every ranked model."""
    if stop < start:
        raise ValueError("phase-analysis stop time must not precede start time")
    if stride <= 0 or workers <= 0:
        raise ValueError("phase-analysis stride and worker count must be positive")
    tasks = [
        (model, mean_sfr, float(start), float(stop), int(stride))
        for model, mean_sfr in ranked
    ]
    if int(workers) == 1:
        results = [_analyze_phase_model(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(_analyze_phase_model, tasks))
    phase_rows = [row for result, _ in results for row in result]
    closure_rows = [row for _, result in results for row in result]
    return pd.DataFrame(phase_rows), pd.DataFrame(closure_rows)


def phase_model_summary(data, ranked, *, bounds=DEFAULT_SUMMARY_RANGE):
    """Return temporal statistics for every model and displayed phase."""
    if bounds[1] <= bounds[0]:
        raise ValueError("summary bounds must be increasing")
    rows = []
    for model, mean_sfr in ranked:
        parameters = model_history_parameters(model)
        omega = float(parameters["omega"])
        qshear = float(parameters["qshear"])
        for phase in PHASES:
            selected = data[
                (data["model"] == model.name)
                & (data["phase"] == phase.key)
                & (data["time"] >= bounds[0])
                & (data["time"] <= bounds[1])
            ]
            if selected.empty:
                raise ValueError(
                    f"{model.name} {phase.label} has no phase samples in {bounds}"
                )
            row = {
                "model": model.name,
                "phase": phase.key,
                "phase_label": phase.label,
                "mean_sfr10": float(mean_sfr),
                "omega": omega,
                "kappa": np.sqrt(2.0 * (2.0 - qshear)) * omega,
                "stellar_midplane_density": float(
                    parameters["stellar_midplane_density"]
                ),
                "qshear": qshear,
                "average_start": float(bounds[0]),
                "average_stop": float(bounds[1]),
            }
            for field in SUMMARY_FIELDS:
                statistics = _finite_statistics(selected[field].to_numpy(dtype=float))
                for suffix, value in zip(
                    ("mean", "std", "median", "percentile16", "percentile84", "count"),
                    statistics,
                ):
                    row[f"{field}_time_{suffix}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _add_sfr_colorbar(fig, ranked, cmap, norm, bounds):
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    axis = fig.add_axes((0.36, 0.035, 0.28, 0.012))
    colorbar = fig.colorbar(scalar, cax=axis, orientation="horizontal")
    colorbar.set_label(
        rf"$\langle\Sigma_{{\rm SFR,10}}\rangle_{{{bounds[0]:g}-{bounds[1]:g}}}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )


def plot_phase_fraction_evolution(
    data,
    ranked,
    output,
    *,
    region="box",
    sfr_bounds=DEFAULT_SFR_RANGE,
    dpi=180,
):
    """Plot mass and volume fraction histories for all six phases."""
    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    fig, axes = plt.subplots(6, 2, figsize=(12.8, 17.0), sharex=True)
    for row_index, phase in enumerate(PHASES):
        phase_data = data[data["phase"] == phase.key]
        for model, mean_sfr in ranked:
            selected = phase_data[phase_data["model"] == model.name].sort_values("time")
            color = cmap(norm(mean_sfr))
            for column, kind in enumerate(("mass", "volume")):
                values = selected[f"{kind}_fraction_{region}"].to_numpy(dtype=float)
                valid = np.isfinite(values) & (values > 0.0)
                axes[row_index, column].plot(
                    selected["time"].to_numpy(dtype=float)[valid],
                    values[valid],
                    color=color,
                    alpha=0.55,
                    linewidth=0.65,
                )
        axes[row_index, 0].set_ylabel(phase.label)
        for column in range(2):
            axes[row_index, column].set_yscale("log")
            axes[row_index, column].grid(alpha=0.15, which="both")
            axes[row_index, column].tick_params(direction="in", top=True, right=True)
    axes[0, 0].set_title("mass fraction")
    axes[0, 1].set_title("volume fraction")
    axes[-1, 0].set_xlabel("time [Myr]")
    axes[-1, 1].set_xlabel("time [Myr]")
    region_label = "whole box" if region == "box" else r"$|z|\leq H_{\rm gas}(t)$"
    fig.suptitle(f"Six-phase fractions in {region_label}", fontsize=14)
    _add_sfr_colorbar(fig, ranked, cmap, norm, sfr_bounds)
    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.075, top=0.96, hspace=0.16, wspace=0.18
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def plot_phase_six_panel(
    data,
    ranked,
    output,
    series,
    *,
    ylabel,
    title,
    sfr_bounds=DEFAULT_SFR_RANGE,
    log=True,
    dpi=180,
):
    """Plot one or more diagnostics in a 3-by-2 six-phase layout."""
    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    fig, axes = plt.subplots(3, 2, figsize=(12.8, 11.4), sharex=True)
    for axis, phase in zip(axes.flat, PHASES):
        phase_data = data[data["phase"] == phase.key]
        for model, mean_sfr in ranked:
            selected = phase_data[phase_data["model"] == model.name].sort_values("time")
            color = cmap(norm(mean_sfr))
            time = selected["time"].to_numpy(dtype=float)
            for field, linestyle in series:
                values = selected[field].to_numpy(dtype=float)
                valid = np.isfinite(values) & ((values > 0.0) if log else True)
                axis.plot(
                    time[valid],
                    values[valid],
                    color=color,
                    linestyle=linestyle,
                    alpha=0.55,
                    linewidth=0.65,
                )
        axis.set_title(phase.label)
        if log:
            axis.set_yscale("log")
        axis.grid(alpha=0.15, which="both")
        axis.tick_params(direction="in", top=True, right=True)
    for axis in axes[-1]:
        axis.set_xlabel("time [Myr]")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14)
    _add_sfr_colorbar(fig, ranked, cmap, norm, sfr_bounds)
    fig.subplots_adjust(
        left=0.08, right=0.99, bottom=0.09, top=0.94, hspace=0.20, wspace=0.16
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def render_suite_phase_evolution(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    output_dir=None,
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    summary_bounds=DEFAULT_SUMMARY_RANGE,
    sfr_bounds=DEFAULT_SFR_RANGE,
    workers=1,
    dpi=180,
    overwrite=False,
):
    """Analyze, cache, summarize, and plot six-phase zprof diagnostics."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_phase_models(suite, model_glob)
    ranked = rank_models_by_sfr(models, bounds=sfr_bounds, max_rows=10000)
    time_path = output_dir / "phase_time_series.csv"
    closure_path = output_dir / "phase_closure_time_series.csv"
    if time_path.exists() and closure_path.exists() and not overwrite:
        print(f"Loading existing {time_path}", flush=True)
        data = pd.read_csv(time_path)
        closure = pd.read_csv(closure_path)
        expected = {model.name for model, _ in ranked}
        if set(data["model"]) != expected or set(closure["model"]) != expected:
            raise ValueError("cached phase model set does not match the suite")
    else:
        data, closure = analyze_suite_phases(
            ranked,
            start=start,
            stop=stop,
            stride=stride,
            workers=workers,
        )
        _atomic_csv(data, time_path)
        _atomic_csv(closure, closure_path)
        print(f"Wrote {time_path}", flush=True)
        print(f"Wrote {closure_path}", flush=True)

    summary = phase_model_summary(data, ranked, bounds=summary_bounds)
    summary_path = output_dir / "phase_model_summary.csv"
    _atomic_csv(summary, summary_path)
    print(f"Wrote {summary_path}", flush=True)

    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    write_model_colors(
        ranked,
        output_dir / "model_sfr_colors.csv",
        cmap,
        norm,
        bounds=sfr_bounds,
    )
    plot_phase_fraction_evolution(
        data,
        ranked,
        output_dir / "phase_fractions_box_evolution.png",
        region="box",
        sfr_bounds=sfr_bounds,
        dpi=dpi,
    )
    plot_phase_fraction_evolution(
        data,
        ranked,
        output_dir / "phase_fractions_hgas_evolution.png",
        region="hgas",
        sfr_bounds=sfr_bounds,
        dpi=dpi,
    )
    plot_phase_six_panel(
        data,
        ranked,
        output_dir / "phase_scale_heights_evolution.png",
        (
            ("mass_scale_height_pc", "-"),
            ("volume_scale_height_pc", "--"),
        ),
        ylabel="RMS height [pc]",
        title="Phase scale heights (solid: mass; dashed: volume)",
        sfr_bounds=sfr_bounds,
        dpi=dpi,
    )
    for region in ("box", "hgas"):
        region_label = "whole box" if region == "box" else r"$|z|\leq H_{\rm gas}(t)$"
        plot_phase_six_panel(
            data,
            ranked,
            output_dir / f"phase_sigma3d_{region}_evolution.png",
            ((f"sigma_3d_{region}", "-"),),
            ylabel=r"$\sigma_{\rm 3D}$ [km s$^{-1}$]",
            title=f"Phase velocity dispersion in {region_label}",
            sfr_bounds=sfr_bounds,
            dpi=dpi,
        )
        plot_phase_six_panel(
            data,
            ranked,
            output_dir / f"phase_alfven3d_{region}_evolution.png",
            (
                (f"alfven_mean_3d_{region}", "-"),
                (f"alfven_perturbed_3d_{region}", "--"),
            ),
            ylabel=r"$v_{\rm A,3D}$ [km s$^{-1}$]",
            title=(
                f"Phase Alfvén speeds in {region_label} "
                "(solid: mean field; dashed: perturbed field)"
            ),
            sfr_bounds=sfr_bounds,
            dpi=dpi,
        )
    return data, closure, summary, ranked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=float, default=DEFAULT_TIME_RANGE[0])
    parser.add_argument("--stop", type=float, default=DEFAULT_TIME_RANGE[1])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--summary-start", type=float, default=DEFAULT_SUMMARY_RANGE[0])
    parser.add_argument("--summary-stop", type=float, default=DEFAULT_SUMMARY_RANGE[1])
    parser.add_argument("--sfr-start", type=float, default=DEFAULT_SFR_RANGE[0])
    parser.add_argument("--sfr-stop", type=float, default=DEFAULT_SFR_RANGE[1])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.start < 0.0 or args.stop < args.start:
        parser.error("require 0 <= --start <= --stop")
    if args.stride <= 0 or args.workers <= 0 or args.dpi <= 0:
        parser.error("--stride, --workers, and --dpi must be positive")
    if args.summary_stop <= args.summary_start:
        parser.error("--summary-stop must exceed --summary-start")
    if args.sfr_stop <= args.sfr_start:
        parser.error("--sfr-stop must exceed --sfr-start")
    render_suite_phase_evolution(
        args.suite,
        model_glob=args.model_glob,
        output_dir=args.output_dir,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        summary_bounds=(args.summary_start, args.summary_stop),
        sfr_bounds=(args.sfr_start, args.sfr_stop),
        workers=args.workers,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
