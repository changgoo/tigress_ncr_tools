#!/usr/bin/env python3
"""Build pressure-regulated feedback diagnostics from suite z-profiles."""

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathena.hst_reader import read_hst
from pathena.units import KB_CGS, KM_CGS, MSUN_CGS, MYR_CGS, PC_CGS

from .plot_suite_evolution import (
    DEFAULT_MODEL_GLOB,
    DEFAULT_SFR_RANGE,
    rank_models_by_sfr,
)
from .plot_suite_hst_evolution import (
    primary_history_file,
    sfr_colormap,
    write_model_colors,
)
from .plot_suite_hst import input_parameter
from .surface_density_stats import read_athinput_section, read_shear_parameters


DEFAULT_SUITE = Path("/tigress/changgoo/anvil/TIGRESS-NCR-suite")
DEFAULT_OUTPUT_NAME = "prfm_diagnostics"
DEFAULT_TIME_SERIES_NAME = "prfm_time_series.csv"
DEFAULT_SUMMARY_NAME = "prfm_model_summary.csv"
DEFAULT_PROFILE_NAME = "prfm_vertical_profiles.csv"
DEFAULT_CMAP = "plasma"
DEFAULT_TIME_RANGE = DEFAULT_SFR_RANGE
DEFAULT_SFR_FIELD = "sfr40"
DEFAULT_TOP_HALF_WIDTH = 10.0
TWO_PHASE_INDICES = (7, 11, 12, 13)
PROFILE_TIME = re.compile(r"t=([+\-0-9.eE]+)")
PROFILE_NAME = re.compile(r"^(?P<problem>.+)\.(?P<dump>\d+)\.phase7\.zprof$")

# The reference pyathena unit convention uses m_H = 1.008 atomic mass units.
AMU_CGS = 1.66053906660e-24
ZPROF_PRESSURE_OVER_KB = 1.4271 * 1.008 * AMU_CGS * KM_CGS**2 / KB_CGS
YIELD_KMS_PER_POK_SFR = (
    KB_CGS * (1000.0 * PC_CGS) ** 2 * (MYR_CGS / 1.0e6) / (MSUN_CGS * KM_CGS)
)

PRESSURE_COMPONENTS = (
    ("pressure_turbulent", r"$P_{\rm turb}$", r"$\Upsilon_{\rm turb}$"),
    ("pressure_thermal", r"$P_{\rm th}$", r"$\Upsilon_{\rm th}$"),
    (
        "pressure_magnetic_turbulent",
        r"$\Pi_{\delta B}$",
        r"$\Upsilon_{\delta B}$",
    ),
    (
        "pressure_magnetic_mean",
        r"$\Pi_{\overline{B}}$",
        r"$\Upsilon_{\overline{B}}$",
    ),
)
PROFILE_FIELDS = (
    "area_two_phase_fraction",
    "density_two_phase",
    "pressure_turbulent",
    "pressure_thermal",
    "pressure_magnetic_turbulent",
    "pressure_magnetic_mean",
    "pressure_total",
)
SUMMARY_FIELDS = (
    "sfr10",
    "sfr40",
    "pressure_turbulent",
    "pressure_thermal",
    "pressure_magnetic_turbulent",
    "pressure_magnetic_mean",
    "pressure_total",
    "pressure_top_turbulent",
    "pressure_top_thermal",
    "pressure_top_magnetic_turbulent",
    "pressure_top_magnetic_mean",
    "pressure_top_total",
    "pressure_delta_turbulent",
    "pressure_delta_thermal",
    "pressure_delta_magnetic_turbulent",
    "pressure_delta_magnetic_mean",
    "pressure_delta_total",
    "weight_external",
    "weight_self_gravity",
    "weight_total",
    "pressure_weight_ratio",
    "pressure_delta_weight_ratio",
    "yield_turbulent",
    "yield_thermal",
    "yield_magnetic_turbulent",
    "yield_magnetic_mean",
    "yield_total",
    "yield_delta_turbulent",
    "yield_delta_thermal",
    "yield_delta_magnetic_turbulent",
    "yield_delta_magnetic_mean",
    "yield_delta_total",
)


def discover_prfm_models(suite, model_glob=DEFAULT_MODEL_GLOB):
    """Return model directories with primary history and required z-profiles."""
    models = []
    for model in sorted(Path(suite).glob(model_glob)):
        if not model.is_dir():
            continue
        try:
            primary_history_file(model)
        except FileNotFoundError:
            continue
        zprof = model / "zprof"
        if (
            next(zprof.glob("*.phase7.zprof"), None) is not None
            and next(zprof.glob("*.whole.zprof"), None) is not None
        ):
            models.append(model)
    if not models:
        raise FileNotFoundError(
            f"no models under {suite} match {model_glob!r} with history and zprof output"
        )
    return models


def _profile_header(path):
    """Return the profile time and comma-separated column names."""
    with Path(path).open(encoding="utf-8") as stream:
        first = stream.readline()
        fields = stream.readline().strip().split(",")
    match = PROFILE_TIME.search(first)
    if match is None:
        raise ValueError(f"could not read zprof time from {path}")
    if not fields:
        raise ValueError(f"could not read zprof fields from {path}")
    return float(match.group(1)), fields


def _read_profile_columns(path, fields):
    """Read selected z-profile fields into a dictionary of float arrays."""
    time, names = _profile_header(path)
    missing = sorted(set(fields) - set(names))
    if missing:
        raise ValueError(f"{path} lacks required fields: {', '.join(missing)}")
    indices = [names.index(field) for field in fields]
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=2,
        usecols=indices,
        ndmin=2,
    )
    return time, {field: values[:, index] for index, field in enumerate(fields)}


def _mesh_area(model):
    """Return horizontal box area from the model's Athena input file."""
    found = []
    for path in sorted(Path(model).glob("athinput*")):
        values = read_athinput_section(path, section="domain1")
        required = ("x1min", "x1max", "x2min", "x2max")
        if all(field in values for field in required):
            area = (values["x1max"] - values["x1min"]) * (
                values["x2max"] - values["x2min"]
            )
            found.append((path, float(area)))
    if not found:
        raise FileNotFoundError(f"could not determine horizontal box area for {model}")
    reference = found[0][1]
    if reference <= 0.0 or any(
        not np.isclose(item[1], reference) for item in found[1:]
    ):
        raise ValueError(f"inconsistent or non-positive horizontal area for {model}")
    return reference


def _boundary_integral(z, values):
    """Integrate a signed force-density profile inward from both boundaries."""
    z = np.asarray(z, dtype=float)
    values = np.asarray(values, dtype=float)
    if z.ndim != 1 or values.shape != z.shape or z.size < 2:
        raise ValueError("weight profiles must be aligned one-dimensional arrays")
    if not np.all(np.diff(z) > 0.0):
        raise ValueError("z-profile coordinates must be strictly increasing")
    dz = float(np.median(np.diff(z)))
    if not np.allclose(np.diff(z), dz):
        raise ValueError("z-profile coordinates must be uniformly spaced")
    result = np.full(z.shape, np.nan)
    lower = z < 0.0
    upper = ~lower
    result[lower] = -np.cumsum(values[lower]) * dz
    result[upper] = np.cumsum(values[upper][::-1])[::-1] * dz
    return result


def _pressure_numerators(values):
    """Return extensive vertical stress components from z-profile fields."""
    turbulent = 2.0 * values["Ek3"]
    thermal = values["P"]
    magnetic_turbulent = values["dPB1"] + values["dPB2"] - values["dPB3"]
    magnetic_all = values["PB1"] + values["PB2"] - values["PB3"]
    return {
        "turbulent": turbulent,
        "thermal": thermal,
        "magnetic_turbulent": magnetic_turbulent,
        "magnetic_mean": magnetic_all - magnetic_turbulent,
    }


def reduce_zprof_snapshot(
    phase_paths,
    whole_path,
    horizontal_area,
    *,
    midplane_half_width=10.0,
    top_half_width=DEFAULT_TOP_HALF_WIDTH,
    return_profile=False,
):
    """Reduce one aligned zprof dump to two-phase pressure and total weight."""
    pressure_fields = (
        "z",
        "A",
        "Ek3",
        "d",
        "P",
        "PB1",
        "PB2",
        "PB3",
        "dPB1",
        "dPB2",
        "dPB3",
    )
    summed = None
    reference_time = None
    reference_z = None
    for path in phase_paths:
        time, values = _read_profile_columns(path, pressure_fields)
        z = values.pop("z")
        if reference_time is None:
            reference_time = time
            reference_z = z
            summed = {field: np.zeros_like(value) for field, value in values.items()}
        elif not np.isclose(time, reference_time) or not np.array_equal(z, reference_z):
            raise ValueError(f"two-phase profiles are not aligned at {path}")
        for field, value in values.items():
            summed[field] += value

    use_midplane = np.abs(reference_z) <= float(midplane_half_width)
    if not np.any(use_midplane):
        raise ValueError("no z-profile cells lie in the requested midplane slab")
    mid = {
        field: float(np.mean(value[use_midplane])) for field, value in summed.items()
    }
    area_two_phase = mid["A"]
    if not np.isfinite(area_two_phase) or area_two_phase <= 0.0:
        raise ValueError("two-phase midplane area is non-positive")
    dz = float(np.median(np.diff(reference_z)))
    if not np.allclose(np.diff(reference_z), dz):
        raise ValueError("z-profile coordinates must be uniformly spaced")
    if top_half_width <= 0.0:
        raise ValueError("top half-width must be positive")
    z_boundary = float(np.max(np.abs(reference_z)) + 0.5 * dz)
    z_top = 0.5 * z_boundary
    use_top = np.abs(np.abs(reference_z) - z_top) <= float(top_half_width)
    if not np.any(use_top):
        raise ValueError("no z-profile cells lie in the requested top slabs")

    top = {field: float(np.mean(value[use_top])) for field, value in summed.items()}
    mid_pressure = _pressure_numerators(mid)
    top_pressure = _pressure_numerators(top)
    pressure_groups = {
        "pressure": mid_pressure,
        "pressure_top": top_pressure,
        "pressure_delta": {
            field: mid_pressure[field] - top_pressure[field] for field in mid_pressure
        },
    }

    weight_time, whole = _read_profile_columns(whole_path, ("z", "dWext", "dWsg"))
    if not np.isclose(weight_time, reference_time):
        raise ValueError(f"whole and phase profile times differ at {whole_path}")
    z_whole = whole["z"]
    if not np.array_equal(z_whole, reference_z):
        raise ValueError(f"whole and phase z grids differ at {whole_path}")
    if not np.isfinite(horizontal_area) or horizontal_area <= 0.0:
        raise ValueError("horizontal area must be finite and positive")
    weight_external_profile = _boundary_integral(z_whole, whole["dWext"])
    weight_self_profile = _boundary_integral(z_whole, whole["dWsg"])
    weight_external = float(np.mean(weight_external_profile[use_midplane]))
    weight_self = float(np.mean(weight_self_profile[use_midplane]))

    conversion = ZPROF_PRESSURE_OVER_KB
    result = {
        "time": float(reference_time),
        "area_two_phase_fraction": area_two_phase / float(horizontal_area),
        "weight_external": weight_external / float(horizontal_area) * conversion,
        "weight_self_gravity": weight_self / float(horizontal_area) * conversion,
    }
    for prefix, values in pressure_groups.items():
        for field, value in values.items():
            result[f"{prefix}_{field}"] = value / area_two_phase * conversion
        result[f"{prefix}_total"] = sum(result[f"{prefix}_{field}"] for field in values)
    result["weight_total"] = result["weight_external"] + result["weight_self_gravity"]
    with np.errstate(divide="ignore", invalid="ignore"):
        result["pressure_weight_ratio"] = (
            result["pressure_total"] / result["weight_total"]
        )
        result["pressure_delta_weight_ratio"] = (
            result["pressure_delta_total"] / result["weight_total"]
        )
    if not return_profile:
        return result

    profile_pressure = _pressure_numerators(summed)
    profile = {
        "z": reference_z,
        "area_two_phase_fraction": summed["A"] / float(horizontal_area),
        "density_two_phase": summed["d"] / float(horizontal_area),
    }
    for field, value in profile_pressure.items():
        profile[f"pressure_{field}"] = value / float(horizontal_area) * conversion
    profile["pressure_total"] = sum(
        profile[f"pressure_{field}"] for field in profile_pressure
    )
    return result, pd.DataFrame(profile)


def _zprof_index(model):
    """Index phase7 z-profiles by integer dump identifier."""
    index = []
    for path in (Path(model) / "zprof").glob("*.phase7.zprof"):
        match = PROFILE_NAME.match(path.name)
        if match:
            index.append((int(match.group("dump")), match.group("problem"), path))
    if not index:
        raise FileNotFoundError(f"no phase7 z-profiles under {Path(model) / 'zprof'}")
    return sorted(index)


def _paths_for_dump(phase7_path, problem_id, dump_id):
    directory = Path(phase7_path).parent
    phase_paths = [
        directory / f"{problem_id}.{dump_id:04d}.phase{phase}.zprof"
        for phase in TWO_PHASE_INDICES
    ]
    whole = directory / f"{problem_id}.{dump_id:04d}.whole.zprof"
    missing = [path for path in phase_paths + [whole] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing aligned zprof files: " + ", ".join(str(path) for path in missing)
        )
    return phase_paths, whole


def _history_sfr(model):
    history = read_hst(primary_history_file(model))
    required = {"time", "sfr10", "sfr40"}
    missing = sorted(required - history.keys())
    if missing:
        raise KeyError(f"{model} history lacks: {', '.join(missing)}")
    return history


def model_prfm_parameters(model):
    """Return Omega, stellar midplane density, and shear parameter."""
    _, qshear, omega = read_shear_parameters(model)
    stellar_surface_density = input_parameter(Path(model), "SurfS")
    stellar_scale_height = input_parameter(Path(model), "zstar")
    values = (omega, stellar_surface_density, stellar_scale_height, qshear)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"nonfinite PRFM model parameters for {model}")
    if stellar_scale_height <= 0.0:
        raise ValueError(f"non-positive stellar scale height for {model}")
    return {
        "omega": float(omega),
        "stellar_midplane_density": float(
            stellar_surface_density / (2.0 * stellar_scale_height)
        ),
        "qshear": float(qshear),
    }


def _summarize_vertical_profiles(profiles, model):
    """Return temporal mean and percentile profiles for one model."""
    if not profiles:
        raise ValueError("at least one vertical profile is required")
    reference_z = profiles[0]["z"].to_numpy(dtype=float)
    if any(
        not np.array_equal(frame["z"].to_numpy(dtype=float), reference_z)
        for frame in profiles[1:]
    ):
        raise ValueError(f"vertical profile grids differ for {model}")
    result = pd.DataFrame({"z": reference_z})
    result.insert(0, "model", Path(model).name)
    result.insert(1, "samples", len(profiles))
    for field in PROFILE_FIELDS:
        values = np.stack([frame[field].to_numpy(dtype=float) for frame in profiles])
        result[f"{field}_mean"] = np.nanmean(values, axis=0)
        result[f"{field}_p16"] = np.nanpercentile(values, 16.0, axis=0)
        result[f"{field}_p50"] = np.nanpercentile(values, 50.0, axis=0)
        result[f"{field}_p84"] = np.nanpercentile(values, 84.0, axis=0)
    return result


def reduce_model_prfm(
    model,
    *,
    time_bounds=DEFAULT_TIME_RANGE,
    stride=1,
    midplane_half_width=10.0,
    top_half_width=DEFAULT_TOP_HALF_WIDTH,
    return_profiles=False,
):
    """Return one model's pressure, weight, SFR, and yield time series."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    indexed = _zprof_index(model)
    selected = []
    for dump_id, problem_id, phase7 in indexed:
        time, _ = _profile_header(phase7)
        if time_bounds[0] <= time <= time_bounds[1]:
            selected.append((dump_id, problem_id, phase7, time))
    selected = selected[::stride]
    if not selected:
        raise ValueError(f"{model} has no zprof dumps in {time_bounds}")

    history = _history_sfr(model)
    history_time = np.asarray(history["time"], dtype=float)
    selected_time = np.asarray([item[3] for item in selected])
    if (
        selected_time.min() < history_time.min()
        or selected_time.max() > history_time.max()
    ):
        raise ValueError(f"history does not cover selected zprof times for {model}")

    horizontal_area = _mesh_area(model)
    rows = []
    profiles = []
    for index, (dump_id, problem_id, phase7, _) in enumerate(selected, start=1):
        phase_paths, whole = _paths_for_dump(phase7, problem_id, dump_id)
        reduced = reduce_zprof_snapshot(
            phase_paths,
            whole,
            horizontal_area,
            midplane_half_width=midplane_half_width,
            top_half_width=top_half_width,
            return_profile=return_profiles,
        )
        if return_profiles:
            row, profile = reduced
            profiles.append(profile)
        else:
            row = reduced
        row.update(
            {
                "model": Path(model).name,
                "dump_id": dump_id,
                "sfr10": float(np.interp(row["time"], history_time, history["sfr10"])),
                "sfr40": float(np.interp(row["time"], history_time, history["sfr40"])),
            }
        )
        sfr = row[DEFAULT_SFR_FIELD]
        for pressure, _, _ in PRESSURE_COMPONENTS:
            suffix = pressure.removeprefix("pressure_")
            row[f"yield_{suffix}"] = (
                row[pressure] / sfr * YIELD_KMS_PER_POK_SFR
                if np.isfinite(sfr) and sfr > 0.0
                else np.nan
            )
            delta_pressure = f"pressure_delta_{suffix}"
            row[f"yield_delta_{suffix}"] = (
                row[delta_pressure] / sfr * YIELD_KMS_PER_POK_SFR
                if np.isfinite(sfr) and sfr > 0.0
                else np.nan
            )
        row["yield_total"] = (
            row["pressure_total"] / sfr * YIELD_KMS_PER_POK_SFR
            if np.isfinite(sfr) and sfr > 0.0
            else np.nan
        )
        row["yield_delta_total"] = (
            row["pressure_delta_total"] / sfr * YIELD_KMS_PER_POK_SFR
            if np.isfinite(sfr) and sfr > 0.0
            else np.nan
        )
        rows.append(row)
        if index == 1 or index % 50 == 0 or index == len(selected):
            print(
                f"{Path(model).name}: {index}/{len(selected)} zprof dumps",
                flush=True,
            )
    time_series = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    if return_profiles:
        return time_series, _summarize_vertical_profiles(profiles, model)
    return time_series


def reduce_suite_prfm(
    ranked,
    *,
    time_bounds=DEFAULT_TIME_RANGE,
    stride=1,
    midplane_half_width=10.0,
    top_half_width=DEFAULT_TOP_HALF_WIDTH,
    return_profiles=False,
):
    """Reduce every ranked model to one concatenated time-series table."""
    frames = []
    profile_frames = []
    for model_index, (model, _) in enumerate(ranked, start=1):
        print(
            f"Reducing {model.name} ({model_index}/{len(ranked)} models)",
            flush=True,
        )
        reduced = reduce_model_prfm(
            model,
            time_bounds=time_bounds,
            stride=stride,
            midplane_half_width=midplane_half_width,
            top_half_width=top_half_width,
            return_profiles=return_profiles,
        )
        if return_profiles:
            time_series, profiles = reduced
            frames.append(time_series)
            profile_frames.append(profiles)
        else:
            frames.append(reduced)
    suite_time_series = pd.concat(frames, ignore_index=True)
    if return_profiles:
        return suite_time_series, pd.concat(profile_frames, ignore_index=True)
    return suite_time_series


def summarize_prfm(time_series, ranked, model_parameters=None):
    """Return per-model temporal means, medians, and 16--84 percentiles."""
    mean_sfr = {model.name: value for model, value in ranked}
    model_parameters = model_parameters or {}
    rows = []
    for model, frame in time_series.groupby("model", sort=False):
        row = {
            "model": model,
            "samples": len(frame),
            "time_min": float(frame["time"].min()),
            "time_max": float(frame["time"].max()),
            "mean_sfr10_color": mean_sfr[model],
        }
        row.update(model_parameters.get(model, {}))
        for field in SUMMARY_FIELDS:
            values = np.asarray(frame[field], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                row[f"{field}_mean"] = float(np.mean(finite))
                row[f"{field}_p16"] = float(np.percentile(finite, 16.0))
                row[f"{field}_p50"] = float(np.percentile(finite, 50.0))
                row[f"{field}_p84"] = float(np.percentile(finite, 84.0))
            else:
                for statistic in ("mean", "p16", "p50", "p84"):
                    row[f"{field}_{statistic}"] = np.nan
        rows.append(row)
    order = {model.name: index for index, (model, _) in enumerate(ranked)}
    return (
        pd.DataFrame(rows)
        .sort_values("model", key=lambda values: values.map(order))
        .reset_index(drop=True)
    )


def _atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _draw_summary_points(axis, summary, xfield, yfield, cmap, norm):
    """Draw temporal means with independent 16--84 percentile bars."""
    for _, row in summary.iterrows():
        x = row[f"{xfield}_mean"]
        y = row[f"{yfield}_mean"]
        color = cmap(norm(row["mean_sfr10_color"]))
        if not np.isfinite(x) or not np.isfinite(y) or x <= 0.0 or y <= 0.0:
            continue
        x16, x84 = row[f"{xfield}_p16"], row[f"{xfield}_p84"]
        y16, y84 = row[f"{yfield}_p16"], row[f"{yfield}_p84"]
        if np.isfinite(x16) and np.isfinite(x84) and x16 > 0.0:
            axis.hlines(y, x16, x84, color=color, alpha=0.42, linewidth=0.8)
        if np.isfinite(y16) and np.isfinite(y84) and y16 > 0.0:
            axis.vlines(x, y16, y84, color=color, alpha=0.42, linewidth=0.8)
        axis.scatter(
            x,
            y,
            s=34,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            zorder=3,
        )


def _decorate_relation_axis(axis, xlabel, ylabel):
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.18, which="both")
    axis.tick_params(direction="in", top=True, right=True)


def plot_prfm_balance(
    summary,
    output,
    *,
    cmap,
    norm,
    sfr_field=DEFAULT_SFR_FIELD,
    pressure_field="pressure_total",
    pressure_symbol=r"P_{\rm tot,2p}",
    colorbar_label=None,
    title="PRFM pressure, weight, and star-formation relations",
    dpi=180,
):
    """Plot total pressure/weight balance and their SFR relations."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    pressure_label = (
        rf"$\langle {pressure_symbol}\rangle\ [k_B\,{{\rm cm}}^{{-3}}\,{{\rm K}}]$"
    )
    if colorbar_label is None:
        colorbar_label = (
            r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
            r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
        )
    _draw_summary_points(axes[0], summary, "weight_total", pressure_field, cmap, norm)
    _draw_summary_points(axes[1], summary, sfr_field, pressure_field, cmap, norm)
    _draw_summary_points(axes[2], summary, sfr_field, "weight_total", cmap, norm)

    _decorate_relation_axis(
        axes[0],
        r"$\langle\mathcal{W}\rangle\ [k_B\,{\rm cm}^{-3}\,{\rm K}]$",
        pressure_label,
    )
    limits = np.asarray(axes[0].get_xlim() + axes[0].get_ylim())
    positive = limits[np.isfinite(limits) & (limits > 0.0)]
    equality = (positive.min(), positive.max())
    axes[0].plot(equality, equality, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(equality)
    axes[0].set_ylim(equality)
    axes[0].text(
        0.05,
        0.93,
        rf"${pressure_symbol}=\mathcal{{W}}$",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )

    sfr_label = (
        r"$\langle\Sigma_{\rm SFR,40}\rangle\ "
        r"[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )
    _decorate_relation_axis(
        axes[1],
        sfr_label,
        pressure_label,
    )
    _decorate_relation_axis(
        axes[2],
        sfr_label,
        r"$\langle\mathcal{W}\rangle\ [k_B\,{\rm cm}^{-3}\,{\rm K}]$",
    )
    for axis, panel in zip(axes, ("a", "b", "c")):
        axis.text(0.97, 0.05, f"({panel})", transform=axis.transAxes, ha="right")

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = fig.add_axes((0.33, 0.13, 0.34, 0.035))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(colorbar_label)
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.34, top=0.88, wspace=0.27)
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_prfm_components(
    summary,
    output,
    *,
    cmap,
    norm,
    sfr_field=DEFAULT_SFR_FIELD,
    colorbar_label=None,
    title="Two-phase midplane pressure components and feedback yields",
    dpi=180,
):
    """Plot four pressure components and their feedback yields versus SFR."""
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.2), sharex=True)
    if colorbar_label is None:
        colorbar_label = (
            r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
            r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
        )
    sfr_label = (
        r"$\langle\Sigma_{\rm SFR,40}\rangle\ "
        r"[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )
    for column, (pressure, pressure_label, yield_label) in enumerate(
        PRESSURE_COMPONENTS
    ):
        suffix = pressure.removeprefix("pressure_")
        _draw_summary_points(axes[0, column], summary, sfr_field, pressure, cmap, norm)
        _draw_summary_points(
            axes[1, column], summary, sfr_field, f"yield_{suffix}", cmap, norm
        )
        _decorate_relation_axis(
            axes[0, column],
            "",
            pressure_label + r"$\ [k_B\,{\rm cm}^{-3}\,{\rm K}]$",
        )
        _decorate_relation_axis(
            axes[1, column],
            sfr_label,
            yield_label + r"$\ [{\rm km\,s}^{-1}]$",
        )
        axes[0, column].set_title(pressure_label, fontsize=12)
        axes[0, column].tick_params(labelbottom=False)
        for row in range(2):
            axes[row, column].text(
                0.96,
                0.05,
                f"({chr(ord('a') + row * 4 + column)})",
                transform=axes[row, column].transAxes,
                ha="right",
                fontsize=9,
            )

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = fig.add_axes((0.34, 0.10, 0.32, 0.025))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(colorbar_label)
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(
        left=0.065, right=0.99, bottom=0.27, top=0.91, wspace=0.25, hspace=0.12
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_prfm_vertical_profiles(
    profile_summary, summary, output, *, cmap, norm, dpi=180
):
    """Plot suite-mean two-phase density and vertical stress profiles."""
    specifications = (
        (
            "density_two_phase",
            r"$\langle n_{\rm H,2p}\rangle\ [{\rm cm}^{-3}]$",
            "log",
        ),
        (
            "pressure_turbulent",
            r"$P_{\rm turb}/k_B\ [{\rm cm}^{-3}\,{\rm K}]$",
            "symlog",
        ),
        ("pressure_thermal", r"$P_{\rm th}/k_B\ [{\rm cm}^{-3}\,{\rm K}]$", "symlog"),
        (
            "pressure_magnetic_turbulent",
            r"$\Pi_{\delta B}/k_B\ [{\rm cm}^{-3}\,{\rm K}]$",
            "symlog",
        ),
        (
            "pressure_magnetic_mean",
            r"$\Pi_{\overline{B}}/k_B\ [{\rm cm}^{-3}\,{\rm K}]$",
            "symlog",
        ),
        ("pressure_total", r"$P_{\rm tot,2p}/k_B\ [{\rm cm}^{-3}\,{\rm K}]$", "symlog"),
    )
    colors = {
        row["model"]: cmap(norm(row["mean_sfr10_color"]))
        for _, row in summary.iterrows()
    }
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=True)
    for axis, (field, label, scale) in zip(axes.flat, specifications):
        for model, frame in profile_summary.groupby("model", sort=False):
            color = colors[model]
            z = frame["z"].to_numpy(dtype=float) / 1000.0
            mean = frame[f"{field}_mean"].to_numpy(dtype=float)
            p16 = frame[f"{field}_p16"].to_numpy(dtype=float)
            p84 = frame[f"{field}_p84"].to_numpy(dtype=float)
            axis.plot(z, mean, color=color, linewidth=0.8, alpha=0.9)
            axis.fill_between(z, p16, p84, color=color, alpha=0.06, linewidth=0)
        if scale == "log":
            axis.set_yscale("log")
        else:
            axis.set_yscale("symlog", linthresh=10.0)
            axis.axhline(0.0, color="0.55", linewidth=0.6)
        axis.set_xlabel(r"$z\ [{\rm kpc}]$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.16, which="both")
        axis.tick_params(direction="in", top=True, right=True)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    color_axis = fig.add_axes((0.35, 0.06, 0.30, 0.022))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )
    fig.suptitle(
        "Time-averaged two-phase vertical density and pressure profiles",
        fontsize=14,
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.17, top=0.91, wspace=0.28, hspace=0.20
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)


def render_suite_prfm(
    suite,
    *,
    model_glob=DEFAULT_MODEL_GLOB,
    output_dir=None,
    time_bounds=DEFAULT_TIME_RANGE,
    stride=1,
    midplane_half_width=10.0,
    top_half_width=DEFAULT_TOP_HALF_WIDTH,
    cmap_name=DEFAULT_CMAP,
    dpi=180,
    overwrite=False,
):
    """Reduce, cache, and plot full-suite PRFM diagnostics."""
    suite = Path(suite).expanduser()
    output_dir = Path(output_dir) if output_dir else suite / DEFAULT_OUTPUT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    models = discover_prfm_models(suite, model_glob)
    ranked = rank_models_by_sfr(models, bounds=time_bounds, max_rows=10000)

    time_series_path = output_dir / DEFAULT_TIME_SERIES_NAME
    profile_path = output_dir / DEFAULT_PROFILE_NAME
    cache_ready = time_series_path.exists() and profile_path.exists()
    if cache_ready:
        cached_columns = pd.read_csv(time_series_path, nrows=0).columns
        cache_ready = {"pressure_delta_total", "yield_delta_total"}.issubset(
            cached_columns
        )
    if cache_ready and not overwrite:
        print(f"Loading existing {time_series_path}", flush=True)
        time_series = pd.read_csv(time_series_path)
        profile_summary = pd.read_csv(profile_path)
        expected = [model.name for model, _ in ranked]
        present = list(dict.fromkeys(time_series["model"]))
        if present != expected:
            raise ValueError(
                "cached PRFM model ordering does not match current ranking"
            )
    else:
        time_series, profile_summary = reduce_suite_prfm(
            ranked,
            time_bounds=time_bounds,
            stride=stride,
            midplane_half_width=midplane_half_width,
            top_half_width=top_half_width,
            return_profiles=True,
        )
        _atomic_csv(time_series, time_series_path)
        _atomic_csv(profile_summary, profile_path)
        print(f"Wrote {profile_path}", flush=True)
        print(f"Wrote {time_series_path}", flush=True)

    model_parameters = {model.name: model_prfm_parameters(model) for model, _ in ranked}
    summary = summarize_prfm(time_series, ranked, model_parameters)
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    _atomic_csv(summary, summary_path)
    print(f"Wrote {summary_path}", flush=True)

    mean_sfr = np.asarray([value for _, value in ranked])
    cmap, norm = sfr_colormap(mean_sfr, cmap_name, "log")
    write_model_colors(
        ranked,
        output_dir / "model_sfr_colors.csv",
        cmap,
        norm,
        bounds=time_bounds,
    )
    balance = output_dir / "prfm_pressure_weight_relations.png"
    components = output_dir / "prfm_pressure_components_yields.png"
    plot_prfm_balance(summary, balance, cmap=cmap, norm=norm, dpi=dpi)
    plot_prfm_components(summary, components, cmap=cmap, norm=norm, dpi=dpi)
    delta_balance = output_dir / "prfm_delta_pressure_weight_relations.png"
    vertical = output_dir / "prfm_vertical_profiles.png"
    plot_prfm_balance(
        summary,
        delta_balance,
        cmap=cmap,
        norm=norm,
        pressure_field="pressure_delta_total",
        pressure_symbol=r"\Delta P_{\rm tot,2p}",
        title="PRFM pressure-drop, weight, and star-formation relations",
        dpi=dpi,
    )
    plot_prfm_vertical_profiles(
        profile_summary, summary, vertical, cmap=cmap, norm=norm, dpi=dpi
    )
    print(f"Wrote {delta_balance}", flush=True)
    print(f"Wrote {vertical}", flush=True)
    print(f"Wrote {balance}", flush=True)
    print(f"Wrote {components}", flush=True)
    color_specs = (
        ("omega", "log", r"$\Omega\ [{\rm Myr}^{-1}]$"),
        (
            "stellar_midplane_density",
            "log",
            r"$\Sigma_*/(2H_*)\ [M_\odot\,{\rm pc}^{-3}]$",
        ),
        ("qshear", "linear", r"$q$"),
    )
    for field, scale, colorbar_label in color_specs:
        values = summary[field].to_numpy(dtype=float)
        parameter_cmap, parameter_norm = sfr_colormap(values, cmap_name, scale)
        colored = summary.copy()
        colored["mean_sfr10_color"] = values
        suffix = f"_color_by_{field}"
        parameter_balance = output_dir / f"prfm_pressure_weight_relations{suffix}.png"
        parameter_delta = (
            output_dir / f"prfm_delta_pressure_weight_relations{suffix}.png"
        )
        parameter_components = (
            output_dir / f"prfm_pressure_components_yields{suffix}.png"
        )
        plot_prfm_balance(
            colored,
            parameter_balance,
            cmap=parameter_cmap,
            norm=parameter_norm,
            colorbar_label=colorbar_label,
            dpi=dpi,
        )
        plot_prfm_balance(
            colored,
            parameter_delta,
            cmap=parameter_cmap,
            norm=parameter_norm,
            pressure_field="pressure_delta_total",
            pressure_symbol=r"\Delta P_{\rm tot,2p}",
            colorbar_label=colorbar_label,
            title="PRFM pressure-drop, weight, and star-formation relations",
            dpi=dpi,
        )
        plot_prfm_components(
            colored,
            parameter_components,
            cmap=parameter_cmap,
            norm=parameter_norm,
            colorbar_label=colorbar_label,
            dpi=dpi,
        )
        for output in (parameter_balance, parameter_delta, parameter_components):
            print(f"Wrote {output}", flush=True)

    return time_series, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start", type=float, default=DEFAULT_TIME_RANGE[0])
    parser.add_argument("--stop", type=float, default=DEFAULT_TIME_RANGE[1])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--midplane-half-width", type=float, default=10.0)
    parser.add_argument("--top-half-width", type=float, default=DEFAULT_TOP_HALF_WIDTH)
    parser.add_argument("--cmap", default=DEFAULT_CMAP)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.stop <= args.start:
        parser.error("--stop must be greater than --start")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.midplane_half_width <= 0.0:
        parser.error("--midplane-half-width must be positive")
    if args.top_half_width <= 0.0:
        parser.error("--top-half-width must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    render_suite_prfm(
        args.suite,
        model_glob=args.model_glob,
        output_dir=args.output_dir,
        time_bounds=(args.start, args.stop),
        stride=args.stride,
        midplane_half_width=args.midplane_half_width,
        top_half_width=args.top_half_width,
        cmap_name=args.cmap,
        dpi=args.dpi,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
