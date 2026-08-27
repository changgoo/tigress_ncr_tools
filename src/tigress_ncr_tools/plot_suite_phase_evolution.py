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
DEFAULT_EVOLUTION_LIMIT_RANGE = (200.0, 600.0)
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
REDUCED_PHASES = (
    ("neutral", "Neutral", ("cold", "unm", "wnm")),
    ("ionized", "Ionized", ("wim", "whim", "him")),
)
CORRELATION_PHASES = PHASES + tuple(
    PhaseDefinition(key, label, ()) for key, label, _ in REDUCED_PHASES
) + (PhaseDefinition("whole", "Whole", ()),)
REQUIRED_PHASE_INDICES = tuple(
    sorted({index for phase in PHASES for index in phase.indices})
)
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
    "P",
    "PB1",
    "PB2",
    "PB3",
    "dPB1",
    "dPB2",
    "dPB3",
)
WHOLE_FIELDS = ("z", "A", "d", "B1", "B2", "B3")
WHOLE_MOMENT_FIELDS = PROFILE_FIELDS + ("B1", "B2", "B3")
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
    "sigma_eff_z_box",
    "sigma_eff_z_hgas",
    "alfven_mean_3d_box",
    "alfven_perturbed_3d_box",
    "alfven_mean_3d_hgas",
    "alfven_perturbed_3d_hgas",
)
FRACTION_FIELDS = (
    "mass_fraction_box",
    "volume_fraction_box",
    "mass_fraction_hgas",
    "volume_fraction_hgas",
)

PHASE_PARAMETER_SPECS = (
    ("stellar_surface_density", r"$\Sigma_*$"),
    ("stellar_scale_height", r"$H_*$"),
    ("omega", r"$\Omega$"),
    ("qshear", r"$q$"),
    ("kappa", r"$\kappa$"),
    ("stellar_midplane_density", r"$\rho_*$"),
    ("mean_sfr10", r"$\langle\Sigma_{\rm SFR,10}\rangle$"),
)
PHASE_PARAMETER_AXIS_SPECS = (
    ("stellar_surface_density", r"$\Sigma_*\ [M_\odot\,{\rm pc}^{-2}]$", "log"),
    ("stellar_scale_height", r"$H_*\ [{\rm pc}]$", "log"),
    ("omega", r"$\Omega\ [{\rm Myr}^{-1}]$", "log"),
    ("qshear", r"$q$", "linear"),
    (
        "kappa",
        r"$\kappa=\sqrt{2(2-q)}\,\Omega\ [{\rm Myr}^{-1}]$",
        "log",
    ),
    (
        "stellar_midplane_density",
        r"$\rho_*=\Sigma_*/(2H_*)\ [M_\odot\,{\rm pc}^{-3}]$",
        "log",
    ),
    (
        "mean_sfr10",
        r"$\langle\Sigma_{\rm SFR,10}\rangle$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$",
        "log",
    ),
)
PHASE_SUMMARY_LABELS = {
    "mass_fraction_box": "mass fraction: box",
    "volume_fraction_box": "volume fraction: box",
    "mass_fraction_hgas": "mass fraction: Hgas",
    "volume_fraction_hgas": "volume fraction: Hgas",
    "mass_scale_height_pc": "mass RMS height",
    "volume_scale_height_pc": "volume RMS height",
    "sigma_3d_box": "sigma3D: box",
    "sigma_3d_hgas": "sigma3D: Hgas",
    "sigma_eff_z_box": "sigma_eff,z: box",
    "sigma_eff_z_hgas": "sigma_eff,z: Hgas",
    "alfven_mean_3d_box": "mean-field vA: box",
    "alfven_perturbed_3d_box": "perturbed vA: box",
    "alfven_mean_3d_hgas": "mean-field vA: Hgas",
    "alfven_perturbed_3d_hgas": "perturbed vA: Hgas",
}
PLOTTED_PHASE_SUMMARY_FIELDS = tuple(PHASE_SUMMARY_LABELS)
FRACTION_CORRELATION_FIELDS = (
    "mass_fraction_box",
    "volume_fraction_box",
    "mass_fraction_hgas",
    "volume_fraction_hgas",
)
SCALE_HEIGHT_CORRELATION_FIELDS = (
    "mass_scale_height_pc",
    "volume_scale_height_pc",
)
DYNAMIC_CORRELATION_FIELDS = (
    "sigma_3d_box",
    "sigma_3d_hgas",
    "sigma_eff_z_box",
    "sigma_eff_z_hgas",
    "alfven_mean_3d_box",
    "alfven_mean_3d_hgas",
    "alfven_perturbed_3d_box",
    "alfven_perturbed_3d_hgas",
)
PHASE_CORRELATION_FAMILIES = (
    (
        "fractions",
        FRACTION_CORRELATION_FIELDS,
        "Phase-fraction correlations with environmental parameters",
    ),
    (
        "scale_heights",
        SCALE_HEIGHT_CORRELATION_FIELDS,
        "Phase scale-height correlations with environmental parameters",
    ),
    (
        "dynamics",
        DYNAMIC_CORRELATION_FIELDS,
        "Phase speed correlations with environmental parameters",
    ),
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


def _support_speed(support, mass):
    """Return sqrt(integrated vertical support / integrated mass)."""
    if mass <= 0.0:
        return np.nan
    scale = max(abs(float(support)), 1.0)
    if support < -1.0e-10 * scale:
        return np.nan
    return float(np.sqrt(max(float(support), 0.0) / mass))


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
    vertical_support = float(
        np.sum(
            (
                2.0 * values["Ek3"]
                + values["P"]
                + values["PB1"]
                + values["PB2"]
                - values["PB3"]
            )
            * weights
        )
    )
    result["sigma_eff_z"] = _support_speed(vertical_support, mass)
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


def reduce_whole_zprof_snapshot(whole_path, horizontal_area):
    """Reduce one whole-gas zprof for aggregate correlation diagnostics."""
    time, values = _read_profile_columns(whole_path, WHOLE_MOMENT_FIELDS)
    z = values.pop("z")
    dz = cell_width(z)
    full_weights = np.full(z.shape, dz)
    mass = float(np.sum(values["d"] * full_weights))
    volume = float(np.sum(values["A"] * full_weights))
    expected_volume = float(horizontal_area) * dz * len(z)
    if mass <= 0.0 or not np.isclose(volume, expected_volume, rtol=2.0e-5):
        raise ValueError(f"invalid whole-gas mass or volume in {whole_path}")
    mean_field = {
        name: values[name] / values["A"] for name in ("B1", "B2", "B3")
    }
    gas_scale_height = _rms_height(z, values["d"], full_weights)
    slab_weights = slab_overlap_weights(z, gas_scale_height)
    box = phase_profile_moments(values, full_weights, mean_field)
    hgas = phase_profile_moments(values, slab_weights, mean_field)
    row = {
        "time": float(time),
        "phase": "whole",
        "phase_label": "Whole",
        "zprof_indices": "whole",
        "gas_scale_height_pc": gas_scale_height,
        "mass_fraction_box": 1.0,
        "volume_fraction_box": 1.0,
        "mass_fraction_hgas": 1.0,
        "volume_fraction_hgas": 1.0,
        "mass_scale_height_pc": gas_scale_height,
        "volume_scale_height_pc": _rms_height(z, values["A"], full_weights),
    }
    for region, moments in (("box", box), ("hgas", hgas)):
        for name, value in moments.items():
            row[f"{name}_{region}"] = value
    return row


def _aggregate_phase_snapshot(snapshot, key, label, phase_keys):
    """Combine component phase moments into one exact reduced-phase row."""
    selected = snapshot[snapshot["phase"].isin(phase_keys)]
    if set(selected["phase"]) != set(phase_keys):
        raise ValueError(f"{label} snapshot is missing one or more component phases")
    first = selected.iloc[0]
    row = {
        "time": float(first["time"]),
        "phase": key,
        "phase_label": label,
        "zprof_indices": "+".join(phase_keys),
        "gas_scale_height_pc": float(first["gas_scale_height_pc"]),
        "model": first["model"],
        "mean_sfr10": float(first["mean_sfr10"]),
        "dump": int(first["dump"]),
    }
    for field in FRACTION_FIELDS:
        row[field] = float(selected[field].sum(min_count=len(phase_keys)))
    for region in ("box", "hgas"):
        mass = selected[f"mass_code_{region}"].to_numpy(dtype=float)
        volume = selected[f"volume_pc3_{region}"].to_numpy(dtype=float)
        row[f"mass_code_{region}"] = float(np.sum(mass))
        row[f"volume_pc3_{region}"] = float(np.sum(volume))
        if region == "box":
            for prefix, weight in (("mass", mass), ("volume", volume)):
                height = selected[f"{prefix}_scale_height_pc"].to_numpy(
                    dtype=float
                )
                valid = np.isfinite(weight) & (weight >= 0.0) & np.isfinite(height)
                if np.all(valid) and np.sum(weight) > 0.0:
                    value = np.sqrt(np.sum(weight * height**2) / np.sum(weight))
                else:
                    value = np.nan
                row[f"{prefix}_scale_height_pc"] = float(value)
        sigmas = []
        alfven_mean = []
        alfven_perturbed = []
        for component in ("x1", "x2", "x3"):
            mean = selected[f"mean_velocity_{component}_{region}"].to_numpy(
                dtype=float
            )
            sigma = selected[f"sigma_{component}_{region}"].to_numpy(dtype=float)
            valid = (
                np.isfinite(mass)
                & (mass >= 0.0)
                & np.isfinite(mean)
                & np.isfinite(sigma)
            )
            if np.all(valid) and np.sum(mass) > 0.0:
                total_mass = np.sum(mass)
                combined_mean = float(np.sum(mass * mean) / total_mass)
                second = float(np.sum(mass * (sigma**2 + mean**2)) / total_mass)
                combined_sigma = _variance_speed(second, combined_mean)
            else:
                combined_mean = combined_sigma = np.nan
            row[f"mean_velocity_{component}_{region}"] = combined_mean
            row[f"sigma_{component}_{region}"] = combined_sigma
            sigmas.append(combined_sigma)
            for prefix, collection in (
                ("alfven_mean", alfven_mean),
                ("alfven_perturbed", alfven_perturbed),
            ):
                speed = selected[f"{prefix}_{component}_{region}"].to_numpy(
                    dtype=float
                )
                valid_speed = np.isfinite(mass) & (mass >= 0.0) & np.isfinite(speed)
                if np.all(valid_speed) and np.sum(mass) > 0.0:
                    combined = np.sqrt(np.sum(mass * speed**2) / np.sum(mass))
                else:
                    combined = np.nan
                row[f"{prefix}_{component}_{region}"] = float(combined)
                collection.append(combined)
        for prefix, values_3d in (
            ("sigma", sigmas),
            ("alfven_mean", alfven_mean),
            ("alfven_perturbed", alfven_perturbed),
        ):
            row[f"{prefix}_3d_{region}"] = (
                float(np.sqrt(np.sum(np.square(values_3d))))
                if np.all(np.isfinite(values_3d))
                else np.nan
            )
        effective_speed = selected[f"sigma_eff_z_{region}"].to_numpy(dtype=float)
        valid_effective = (
            np.isfinite(mass) & (mass >= 0.0) & np.isfinite(effective_speed)
        )
        if np.all(valid_effective) and np.sum(mass) > 0.0:
            row[f"sigma_eff_z_{region}"] = float(
                np.sqrt(np.sum(mass * effective_speed**2) / np.sum(mass))
            )
        else:
            row[f"sigma_eff_z_{region}"] = np.nan
    return row


def aggregate_reduced_phase_time_series(data):
    """Construct exact neutral and ionized moment histories from six phases."""
    base = data[data["phase"].isin([phase.key for phase in PHASES])]
    rows = []
    for _, snapshot in base.groupby(["model", "dump"], sort=False):
        for key, label, phase_keys in REDUCED_PHASES:
            rows.append(_aggregate_phase_snapshot(snapshot, key, label, phase_keys))
    return pd.DataFrame(rows)


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


def _analyze_whole_model(task):
    model, mean_sfr, dump_times = task
    indexed = {
        dump: (problem, reference)
        for dump, problem, reference in phase_zprof_index(model)
    }
    horizontal_area = _mesh_area(model)
    rows = []
    for item, (dump, expected_time) in enumerate(dump_times, start=1):
        if dump not in indexed:
            raise FileNotFoundError(f"{model.name} has no zprof dump {dump}")
        problem, reference = indexed[dump]
        whole_path = reference.parent / f"{problem}.{dump:04d}.whole.zprof"
        row = reduce_whole_zprof_snapshot(whole_path, horizontal_area)
        if not np.isclose(row["time"], expected_time):
            raise ValueError(f"cached and whole-zprof times differ for {whole_path}")
        row.update(
            {
                "model": model.name,
                "mean_sfr10": float(mean_sfr),
                "dump": int(dump),
            }
        )
        rows.append(row)
        if item == 1 or item % 100 == 0 or item == len(dump_times):
            print(
                f"{model.name}: {item}/{len(dump_times)} whole zprof dumps",
                flush=True,
            )
    return rows


def analyze_suite_whole_from_cache(data, ranked, *, workers=1):
    """Reduce true whole-gas moments at the dumps present in a phase cache."""
    tasks = []
    for model, mean_sfr in ranked:
        selected = data[
            (data["model"] == model.name)
            & (data["phase"].isin([phase.key for phase in PHASES]))
        ][["dump", "time"]].drop_duplicates()
        dump_times = [
            (int(row.dump), float(row.time))
            for row in selected.sort_values("dump").itertuples(index=False)
        ]
        if not dump_times:
            raise ValueError(f"{model.name} has no cached phase dumps")
        tasks.append((model, mean_sfr, dump_times))
    if int(workers) == 1:
        results = [_analyze_whole_model(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            results = list(executor.map(_analyze_whole_model, tasks))
    return pd.DataFrame([row for result in results for row in result])


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


def phase_model_summary(
    data,
    ranked,
    *,
    bounds=DEFAULT_SUMMARY_RANGE,
    phases=PHASES,
):
    """Return temporal statistics for every model and displayed phase."""
    if bounds[1] <= bounds[0]:
        raise ValueError("summary bounds must be increasing")
    rows = []
    for model, mean_sfr in ranked:
        parameters = model_history_parameters(model)
        omega = float(parameters["omega"])
        qshear = float(parameters["qshear"])
        for phase in phases:
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
                "stellar_surface_density": float(parameters["stellar_surface_density"]),
                "stellar_scale_height": float(parameters["stellar_scale_height"]),
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


def reduced_phase_fraction_summary(data, ranked, *, bounds=DEFAULT_SUMMARY_RANGE):
    """Summarize neutral and ionized sums without renormalizing the UIM residual."""
    if bounds[1] <= bounds[0]:
        raise ValueError("summary bounds must be increasing")
    rows = []
    for model, mean_sfr in ranked:
        parameters = model_history_parameters(model)
        omega = float(parameters["omega"])
        qshear = float(parameters["qshear"])
        model_data = data[data["model"] == model.name]
        for key, label, phase_keys in REDUCED_PHASES:
            selected = model_data[model_data["phase"].isin(phase_keys)]
            present = set(selected["phase"].unique())
            if present != set(phase_keys):
                raise ValueError(
                    f"{model.name} {label} is missing phases: "
                    f"{sorted(set(phase_keys) - present)}"
                )
            grouped = (
                selected.groupby("time", sort=True)[list(FRACTION_FIELDS)]
                .sum(min_count=len(phase_keys))
                .reset_index()
            )
            grouped = grouped[
                (grouped["time"] >= bounds[0])
                & (grouped["time"] <= bounds[1])
            ]
            if grouped.empty:
                raise ValueError(
                    f"{model.name} {label} has no samples in {bounds}"
                )
            row = {
                "model": model.name,
                "phase": key,
                "phase_label": label,
                "component_phases": "+".join(phase_keys),
                "mean_sfr10": float(mean_sfr),
                "omega": omega,
                "kappa": np.sqrt(2.0 * (2.0 - qshear)) * omega,
                "stellar_surface_density": float(
                    parameters["stellar_surface_density"]
                ),
                "stellar_scale_height": float(
                    parameters["stellar_scale_height"]
                ),
                "stellar_midplane_density": float(
                    parameters["stellar_midplane_density"]
                ),
                "qshear": qshear,
                "average_start": float(bounds[0]),
                "average_stop": float(bounds[1]),
            }
            for field in FRACTION_FIELDS:
                statistics = _finite_statistics(
                    grouped[field].to_numpy(dtype=float)
                )
                for suffix, value in zip(
                    ("mean", "std", "median", "percentile16", "percentile84", "count"),
                    statistics,
                ):
                    row[f"{field}_time_{suffix}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _spearman_coefficient(x, y):
    """Return a finite-pair Spearman coefficient and sample count."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    count = int(np.count_nonzero(valid))
    if count < 3:
        return np.nan, count
    x_rank = pd.Series(x[valid]).rank(method="average")
    y_rank = pd.Series(y[valid]).rank(method="average")
    return float(x_rank.corr(y_rank)), count


def reference_axis_limits(values, *, log=True, padding=0.08):
    """Return padded limits determined only from finite reference values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if log:
        values = values[values > 0.0]
    if values.size == 0:
        return None
    transformed = np.log10(values) if log else values
    lower = float(np.min(transformed))
    upper = float(np.max(transformed))
    span = upper - lower
    if span == 0.0:
        if log:
            pad = padding
        else:
            pad = max(1.0, abs(lower)) * padding
    else:
        pad = padding * span
    limits = (lower - pad, upper + pad)
    if log:
        return 10.0 ** limits[0], 10.0 ** limits[1]
    return limits


def _set_reference_ylim(axis, phase_data, fields, bounds, *, log):
    reference = phase_data[
        (phase_data["time"] >= bounds[0]) & (phase_data["time"] <= bounds[1])
    ]
    values = [reference[field].to_numpy(dtype=float) for field in fields]
    limits = reference_axis_limits(
        np.concatenate(values) if values else np.asarray([]), log=log
    )
    if limits is not None:
        axis.set_ylim(limits)


def _plot_parameter_relation_series(
    axis,
    summary,
    ranked,
    phase_key,
    field,
    parameter,
    parameter_scale,
    cmap,
    norm,
    *,
    marker="o",
    x_offset=0.0,
):
    model_names = [model.name for model, _ in ranked]
    selected = summary[summary["phase"] == phase_key].set_index("model")
    selected = selected.reindex(model_names)
    x = selected[parameter].to_numpy(dtype=float)
    median = selected[f"{field}_time_median"].to_numpy(dtype=float)
    low = selected[f"{field}_time_percentile16"].to_numpy(dtype=float)
    high = selected[f"{field}_time_percentile84"].to_numpy(dtype=float)
    color_values = selected["mean_sfr10"].to_numpy(dtype=float)
    valid = (
        np.isfinite(x)
        & np.isfinite(median)
        & np.isfinite(low)
        & np.isfinite(high)
        & np.isfinite(color_values)
        & (median > 0.0)
        & (low > 0.0)
        & (high > 0.0)
        & (low <= median)
        & (median <= high)
    )
    if parameter_scale == "log":
        valid &= x > 0.0
        plotted_x = x * (10.0**x_offset)
    else:
        finite_x = x[np.isfinite(x)]
        span = np.ptp(finite_x) if finite_x.size else 0.0
        plotted_x = x + x_offset * span
    for xv, center, lower, upper, color_value in zip(
        plotted_x[valid],
        median[valid],
        low[valid],
        high[valid],
        color_values[valid],
    ):
        axis.errorbar(
            xv,
            center,
            yerr=np.asarray([[center - lower], [upper - center]]),
            color=cmap(norm(color_value)),
            alpha=0.42,
            linewidth=0.65,
            zorder=1,
        )
    axis.scatter(
        plotted_x[valid],
        median[valid],
        c=color_values[valid],
        cmap=cmap,
        norm=norm,
        marker=marker,
        s=22,
        edgecolor="black",
        linewidth=0.25,
        alpha=0.86,
        zorder=2,
    )
    coefficient, count = _spearman_coefficient(x[valid], median[valid])
    return coefficient, count


def plot_phase_fraction_parameter_relations(
    summary,
    ranked,
    output_dir,
    *,
    bounds=DEFAULT_SUMMARY_RANGE,
    sfr_bounds=DEFAULT_SFR_RANGE,
    dpi=180,
):
    """Plot each six-phase fraction directly against every parameter."""
    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    fraction_specs = (
        ("mass_fraction_box", "mass fraction: whole box"),
        ("volume_fraction_box", "volume fraction: whole box"),
        ("mass_fraction_hgas", r"mass fraction: $|z|\leq H_{\rm gas}$"),
        ("volume_fraction_hgas", r"volume fraction: $|z|\leq H_{\rm gas}$"),
    )
    for field, field_label in fraction_specs:
        figure, axes = plt.subplots(
            len(PHASES),
            len(PHASE_PARAMETER_AXIS_SPECS),
            figsize=(22.0, 17.2),
            sharex="col",
            sharey="row",
        )
        for row, phase in enumerate(PHASES):
            for column, (parameter, parameter_label, scale) in enumerate(
                PHASE_PARAMETER_AXIS_SPECS
            ):
                axis = axes[row, column]
                coefficient, count = _plot_parameter_relation_series(
                    axis,
                    summary,
                    ranked,
                    phase.key,
                    field,
                    parameter,
                    scale,
                    cmap,
                    norm,
                )
                if scale == "log":
                    axis.set_xscale("log")
                axis.set_yscale("log")
                axis.grid(alpha=0.16, which="both")
                axis.tick_params(direction="in", top=True, right=True)
                axis.text(
                    0.04,
                    0.94,
                    rf"$\rho_s={coefficient:+.2f}$ ($N={count}$)",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.5,
                    bbox={"facecolor": "white", "alpha": 0.62, "edgecolor": "none"},
                )
                if row == 0:
                    axis.set_title(parameter_label, fontsize=10)
                if row == len(PHASES) - 1:
                    axis.set_xlabel(parameter_label, fontsize=9)
            axes[row, 0].set_ylabel(f"{phase.label}\n{field_label}")
        figure.suptitle(
            f"Six-phase {field_label}: {bounds[0]:g}--{bounds[1]:g} Myr",
            fontsize=14,
        )
        _add_sfr_colorbar(figure, ranked, cmap, norm, sfr_bounds)
        figure.subplots_adjust(
            left=0.075,
            right=0.99,
            bottom=0.085,
            top=0.945,
            hspace=0.12,
            wspace=0.12,
        )
        output = Path(output_dir) / f"phase_{field}_parameter_relations.png"
        figure.savefig(output, dpi=dpi, facecolor="white")
        plt.close(figure)
        print(f"Wrote {output}", flush=True)


def plot_reduced_phase_fraction_parameter_relations(
    summary,
    ranked,
    output_dir,
    *,
    bounds=DEFAULT_SUMMARY_RANGE,
    sfr_bounds=DEFAULT_SFR_RANGE,
    dpi=180,
):
    """Plot neutral and ionized summaries in separate parameter figures."""
    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    fraction_specs = (
        ("mass_fraction_box", "mass fraction: whole box"),
        ("volume_fraction_box", "volume fraction: whole box"),
        ("mass_fraction_hgas", r"mass fraction: $|z|\leq H_{\rm gas}$"),
        ("volume_fraction_hgas", r"volume fraction: $|z|\leq H_{\rm gas}$"),
    )
    markers = {"neutral": "o", "ionized": "^"}
    for phase_key, phase_label, _ in REDUCED_PHASES:
        figure, axes = plt.subplots(
            len(fraction_specs),
            len(PHASE_PARAMETER_AXIS_SPECS),
            figsize=(22.0, 12.4),
            sharex="col",
            sharey="row",
        )
        for row, (field, field_label) in enumerate(fraction_specs):
            for column, (parameter, parameter_label, scale) in enumerate(
                PHASE_PARAMETER_AXIS_SPECS
            ):
                axis = axes[row, column]
                coefficient, count = _plot_parameter_relation_series(
                    axis,
                    summary,
                    ranked,
                    phase_key,
                    field,
                    parameter,
                    scale,
                    cmap,
                    norm,
                    marker=markers[phase_key],
                )
                axis.text(
                    0.04,
                    0.95,
                    rf"$\rho_s={coefficient:+.2f}$ ($N={count}$)",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.5,
                    bbox={"facecolor": "white", "alpha": 0.62, "edgecolor": "none"},
                )
                if scale == "log":
                    axis.set_xscale("log")
                axis.set_yscale("log")
                axis.grid(alpha=0.16, which="both")
                axis.tick_params(direction="in", top=True, right=True)
                if row == 0:
                    axis.set_title(parameter_label, fontsize=10)
                if row == len(fraction_specs) - 1:
                    axis.set_xlabel(parameter_label, fontsize=9)
            axes[row, 0].set_ylabel(field_label)
        figure.suptitle(
            f"Reduced {phase_label.lower()} phase fractions: "
            f"{bounds[0]:g}--{bounds[1]:g} Myr",
            fontsize=14,
        )
        _add_sfr_colorbar(figure, ranked, cmap, norm, sfr_bounds)
        figure.subplots_adjust(
            left=0.075,
            right=0.99,
            bottom=0.105,
            top=0.93,
            hspace=0.15,
            wspace=0.12,
        )
        output = Path(output_dir) / f"phase_{phase_key}_fraction_parameter_relations.png"
        figure.savefig(output, dpi=dpi, facecolor="white")
        plt.close(figure)
        print(f"Wrote {output}", flush=True)


def plot_phase_structure_dynamics_parameter_relations(
    summary,
    ranked,
    output_dir,
    *,
    phases=CORRELATION_PHASES,
    bounds=DEFAULT_SUMMARY_RANGE,
    sfr_bounds=DEFAULT_SFR_RANGE,
    dpi=180,
):
    """Plot full model scatter behind structure and dynamics matrices."""
    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    fields = SCALE_HEIGHT_CORRELATION_FIELDS + DYNAMIC_CORRELATION_FIELDS
    for field in fields:
        figure, axes = plt.subplots(
            len(phases),
            len(PHASE_PARAMETER_AXIS_SPECS),
            figsize=(22.0, 23.5),
            sharex="col",
            sharey="row",
        )
        for row, phase in enumerate(phases):
            for column, (parameter, parameter_label, scale) in enumerate(
                PHASE_PARAMETER_AXIS_SPECS
            ):
                axis = axes[row, column]
                coefficient, count = _plot_parameter_relation_series(
                    axis,
                    summary,
                    ranked,
                    phase.key,
                    field,
                    parameter,
                    scale,
                    cmap,
                    norm,
                )
                if scale == "log":
                    axis.set_xscale("log")
                axis.set_yscale("log")
                axis.grid(alpha=0.16, which="both")
                axis.tick_params(direction="in", top=True, right=True)
                axis.text(
                    0.04,
                    0.94,
                    rf"$\rho_s={coefficient:+.2f}$ ($N={count}$)",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.5,
                    bbox={"facecolor": "white", "alpha": 0.62, "edgecolor": "none"},
                )
                if row == 0:
                    axis.set_title(parameter_label, fontsize=10)
                if row == len(phases) - 1:
                    axis.set_xlabel(parameter_label, fontsize=9)
            if field in SCALE_HEIGHT_CORRELATION_FIELDS:
                unit = "[pc]"
            else:
                unit = r"[km s$^{-1}$]"
            axes[row, 0].set_ylabel(f"{phase.label}\n{unit}")
        figure.suptitle(
            f"Direct model scatter for {PHASE_SUMMARY_LABELS[field]}: "
            f"{bounds[0]:g}--{bounds[1]:g} Myr",
            fontsize=14,
        )
        _add_sfr_colorbar(figure, ranked, cmap, norm, sfr_bounds)
        figure.subplots_adjust(
            left=0.075,
            right=0.99,
            bottom=0.075,
            top=0.955,
            hspace=0.12,
            wspace=0.12,
        )
        output = Path(output_dir) / f"phase_{field}_parameter_relations.png"
        figure.savefig(output, dpi=dpi, facecolor="white")
        plt.close(figure)
        print(f"Wrote {output}", flush=True)


def _add_sfr_colorbar(fig, ranked, cmap, norm, bounds):
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    axis = fig.add_axes((0.36, 0.035, 0.28, 0.012))
    colorbar = fig.colorbar(scalar, cax=axis, orientation="horizontal")
    colorbar.set_label(
        rf"$\langle\Sigma_{{\rm SFR,10}}\rangle_{{{bounds[0]:g}-{bounds[1]:g}}}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$"
    )


def _plot_phase_summary_axis(
    axis,
    summary,
    ranked,
    series,
    cmap,
    norm,
    *,
    ylabel,
    log=True,
):
    """Plot model temporal medians and percentile ranges by phase."""
    model_names = [model.name for model, _ in ranked]
    model_offset = {
        name: offset
        for name, offset in zip(
            model_names, np.linspace(-0.25, 0.25, len(model_names))
        )
    }
    for series_index, (field, label, marker) in enumerate(series):
        centers = []
        series_shift = (series_index - 0.5 * (len(series) - 1)) * 0.055
        for phase_index, phase in enumerate(PHASES):
            selected = summary[summary["phase"] == phase.key].set_index("model")
            selected = selected.reindex(model_names)
            median = selected[f"{field}_time_median"].to_numpy(dtype=float)
            low = selected[f"{field}_time_percentile16"].to_numpy(dtype=float)
            high = selected[f"{field}_time_percentile84"].to_numpy(dtype=float)
            color_values = selected["mean_sfr10"].to_numpy(dtype=float)
            x = np.asarray(
                [
                    phase_index + model_offset[name] + series_shift
                    for name in model_names
                ]
            )
            valid = (
                np.isfinite(median)
                & np.isfinite(low)
                & np.isfinite(high)
                & np.isfinite(color_values)
                & (low <= median)
                & (median <= high)
            )
            if log:
                valid &= (median > 0.0) & (low > 0.0) & (high > 0.0)
            for xv, center, lower, upper, color_value in zip(
                x[valid],
                median[valid],
                low[valid],
                high[valid],
                color_values[valid],
            ):
                axis.errorbar(
                    xv,
                    center,
                    yerr=np.asarray([[center - lower], [upper - center]]),
                    color=cmap(norm(color_value)),
                    alpha=0.38,
                    linewidth=0.55,
                    zorder=1,
                )
            axis.scatter(
                x[valid],
                median[valid],
                c=color_values[valid],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=18,
                edgecolor="black",
                linewidth=0.18,
                alpha=0.82,
                zorder=2,
            )
            centers.append(np.nanmedian(median[valid]))
        axis.plot(
            np.arange(len(PHASES)) + series_shift,
            centers,
            color="black",
            marker=marker,
            markersize=5,
            linewidth=1.0,
            label=label,
            zorder=3,
        )
    axis.set_xticks(np.arange(len(PHASES)), [phase.label for phase in PHASES])
    axis.tick_params(axis="x", labelrotation=28)
    axis.set_ylabel(ylabel)
    if log:
        axis.set_yscale("log")
    axis.grid(alpha=0.16, which="both")
    axis.tick_params(direction="in", top=True, right=True)
    if len(series) > 1:
        axis.legend(fontsize=8, loc="best")


def plot_phase_model_summaries(
    summary,
    ranked,
    output_dir,
    *,
    bounds=DEFAULT_SUMMARY_RANGE,
    sfr_bounds=DEFAULT_SFR_RANGE,
    dpi=180,
):
    """Plot temporal phase summaries across the model ensemble."""
    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    fraction_specs = (
        ("mass_fraction_box", "mass fraction: whole box"),
        ("volume_fraction_box", "volume fraction: whole box"),
        ("mass_fraction_hgas", r"mass fraction: $|z|\leq H_{\rm gas}$"),
        ("volume_fraction_hgas", r"volume fraction: $|z|\leq H_{\rm gas}$"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    for axis, (field, ylabel) in zip(axes.flat, fraction_specs):
        _plot_phase_summary_axis(
            axis,
            summary,
            ranked,
            ((field, "temporal median", "o"),),
            cmap,
            norm,
            ylabel=ylabel,
        )
    figure.suptitle(
        f"Six-phase fraction summaries: {bounds[0]:g}--{bounds[1]:g} Myr",
        fontsize=14,
    )
    _add_sfr_colorbar(figure, ranked, cmap, norm, sfr_bounds)
    figure.subplots_adjust(
        left=0.08, right=0.99, bottom=0.13, top=0.93, hspace=0.31, wspace=0.24
    )
    output = Path(output_dir) / "phase_fraction_model_summary.png"
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Wrote {output}", flush=True)

    dynamic_specs = (
        (
            (("mass_scale_height_pc", "mass weighted", "o"),),
            "mass RMS height [pc]",
        ),
        (
            (("volume_scale_height_pc", "volume weighted", "o"),),
            "volume RMS height [pc]",
        ),
        ((("sigma_3d_box", "whole box", "o"),), r"$\sigma_{\rm 3D}$ [km/s]"),
        (
            (("sigma_3d_hgas", r"$|z|\leq H_{\rm gas}$", "o"),),
            r"$\sigma_{\rm 3D}$ [km/s]",
        ),
        (
            (("sigma_eff_z_box", "whole box", "o"),),
            r"$\sigma_{{\rm eff},z}$ [km/s]",
        ),
        (
            (("sigma_eff_z_hgas", r"$|z|\leq H_{\rm gas}$", "o"),),
            r"$\sigma_{{\rm eff},z}$ [km/s]",
        ),
        (
            (
                ("alfven_mean_3d_box", "mean field", "o"),
                ("alfven_perturbed_3d_box", "perturbed field", "^"),
            ),
            r"$v_{\rm A,3D}$: whole box [km/s]",
        ),
        (
            (
                ("alfven_mean_3d_hgas", "mean field", "o"),
                ("alfven_perturbed_3d_hgas", "perturbed field", "^"),
            ),
            r"$v_{\rm A,3D}$: $|z|\leq H_{\rm gas}$ [km/s]",
        ),
    )
    dynamic_titles = (
        "mass-weighted scale height",
        "volume-weighted scale height",
        "velocity dispersion: whole box",
        r"velocity dispersion: $|z|\leq H_{\rm gas}$",
        "effective vertical support speed: whole box",
        r"effective vertical support speed: $|z|\leq H_{\rm gas}$",
        "Alfvén speeds: whole box",
        r"Alfvén speeds: $|z|\leq H_{\rm gas}$",
    )
    figure, axes = plt.subplots(4, 2, figsize=(13.2, 16.5))
    for axis, (series, ylabel), panel_title in zip(
        axes.flat, dynamic_specs, dynamic_titles
    ):
        _plot_phase_summary_axis(
            axis, summary, ranked, series, cmap, norm, ylabel=ylabel
        )
        axis.set_title(panel_title)
    figure.suptitle(
        f"Six-phase structure and speed summaries: "
        f"{bounds[0]:g}--{bounds[1]:g} Myr",
        fontsize=14,
    )
    _add_sfr_colorbar(figure, ranked, cmap, norm, sfr_bounds)
    figure.subplots_adjust(
        left=0.08, right=0.99, bottom=0.10, top=0.945, hspace=0.34, wspace=0.24
    )
    output = Path(output_dir) / "phase_structure_speed_model_summary.png"
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Wrote {output}", flush=True)


def phase_parameter_correlations(summary, *, phases=PHASES):
    """Return Spearman model correlations for plotted phase summaries."""
    rows = []
    for phase in phases:
        selected = summary[summary["phase"] == phase.key]
        for field in PLOTTED_PHASE_SUMMARY_FIELDS:
            y = selected[f"{field}_time_median"].to_numpy(dtype=float)
            for parameter, parameter_label in PHASE_PARAMETER_SPECS:
                x = selected[parameter].to_numpy(dtype=float)
                coefficient, count = _spearman_coefficient(x, y)
                rows.append(
                    {
                        "phase": phase.key,
                        "phase_label": phase.label,
                        "quantity": field,
                        "quantity_label": PHASE_SUMMARY_LABELS[field],
                        "parameter": parameter,
                        "parameter_label": parameter_label,
                        "spearman_rho": coefficient,
                        "model_count": count,
                    }
                )
    return pd.DataFrame(rows)


def plot_phase_parameter_correlation_heatmaps(
    correlations,
    output,
    fields,
    *,
    title,
    dpi=180,
    phases=PHASES,
):
    """Plot phase-by-parameter Spearman matrices for related quantities."""
    ncols = 2
    nrows = int(np.ceil(len(fields) / ncols))
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(12.8, 3.9 * nrows + 0.5), squeeze=False
    )
    parameter_names = [name for name, _ in PHASE_PARAMETER_SPECS]
    parameter_labels = [label for _, label in PHASE_PARAMETER_SPECS]
    for axis, field in zip(axes.flat, fields):
        selected = correlations[correlations["quantity"] == field]
        matrix = np.full((len(phases), len(parameter_names)), np.nan)
        for row, phase in enumerate(phases):
            for column, parameter in enumerate(parameter_names):
                match = selected[
                    (selected["phase"] == phase.key)
                    & (selected["parameter"] == parameter)
                ]
                if len(match) == 1:
                    matrix[row, column] = match["spearman_rho"].iloc[0]
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
            np.arange(len(phases)), [phase.label for phase in phases]
        )
        axis.set_title(PHASE_SUMMARY_LABELS[field])
        for row, column in np.ndindex(matrix.shape):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.55 else "black",
                )
            else:
                axis.text(
                    column,
                    row,
                    "--",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.45",
                )
    for axis in axes.flat[len(fields) :]:
        axis.set_visible(False)
    top = 0.80 if nrows == 1 else 0.91
    color_axis = figure.add_axes((0.92, 0.15, 0.018, top - 0.15))
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label(r"Spearman $\rho_s$ across models")
    figure.suptitle(title, fontsize=14, y=0.975)
    figure.subplots_adjust(
        left=0.09, right=0.88, bottom=0.08, top=top, hspace=0.34, wspace=0.26
    )
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Wrote {output}", flush=True)


def plot_phase_fraction_evolution(
    data,
    ranked,
    output,
    *,
    region="box",
    sfr_bounds=DEFAULT_SFR_RANGE,
    limit_bounds=DEFAULT_EVOLUTION_LIMIT_RANGE,
    phases=CORRELATION_PHASES,
    dpi=180,
):
    """Plot fraction histories for displayed and aggregate phases."""
    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    height = 2.5 * len(phases) + 2.0
    fig, axes = plt.subplots(len(phases), 2, figsize=(12.8, height), sharex=True)
    for row_index, phase in enumerate(phases):
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
            kind = ("mass", "volume")[column]
            _set_reference_ylim(
                axes[row_index, column],
                phase_data,
                (f"{kind}_fraction_{region}",),
                limit_bounds,
                log=True,
            )
    axes[0, 0].set_title("mass fraction")
    axes[0, 1].set_title("volume fraction")
    axes[-1, 0].set_xlabel("time [Myr]")
    axes[-1, 1].set_xlabel("time [Myr]")
    region_label = "whole box" if region == "box" else r"$|z|\leq H_{\rm gas}(t)$"
    fig.suptitle(f"Phase fractions in {region_label}", fontsize=14)
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
    limit_bounds=DEFAULT_EVOLUTION_LIMIT_RANGE,
    phases=CORRELATION_PHASES,
    log=True,
    dpi=180,
):
    """Plot diagnostics for displayed and aggregate phases."""
    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    ncols = 3 if len(phases) > 6 else 2
    nrows = int(np.ceil(len(phases) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols, 3.7 * nrows + 1.0), sharex=True
    )
    for axis, phase in zip(axes.flat, phases):
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
        _set_reference_ylim(
            axis,
            phase_data,
            tuple(field for field, _ in series),
            limit_bounds,
            log=log,
        )
    for axis in axes.flat[len(phases) :]:
        axis.set_visible(False)
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

    cached_phases = set(data["phase"].unique())
    aggregate_changed = False
    reduced_keys = {key for key, _, _ in REDUCED_PHASES}
    if not reduced_keys.issubset(cached_phases):
        data = pd.concat(
            [data, aggregate_reduced_phase_time_series(data)],
            ignore_index=True,
            sort=False,
        )
        aggregate_changed = True
    if "whole" not in cached_phases:
        whole = analyze_suite_whole_from_cache(data, ranked, workers=workers)
        data = pd.concat([data, whole], ignore_index=True, sort=False)
        aggregate_changed = True
    if aggregate_changed:
        _atomic_csv(data, time_path)
        print(f"Augmented {time_path} with aggregate phase moments", flush=True)

    summary = phase_model_summary(data, ranked, bounds=summary_bounds)
    summary_path = output_dir / "phase_model_summary.csv"
    _atomic_csv(summary, summary_path)
    print(f"Wrote {summary_path}", flush=True)

    reduced_summary = reduced_phase_fraction_summary(
        data, ranked, bounds=summary_bounds
    )
    correlation_summary = phase_model_summary(
        data, ranked, bounds=summary_bounds, phases=CORRELATION_PHASES
    )
    correlation_summary_path = output_dir / "phase_correlation_model_summary.csv"
    _atomic_csv(correlation_summary, correlation_summary_path)
    print(f"Wrote {correlation_summary_path}", flush=True)

    reduced_summary_path = output_dir / "phase_two_phase_fraction_summary.csv"
    _atomic_csv(reduced_summary, reduced_summary_path)
    print(f"Wrote {reduced_summary_path}", flush=True)

    cmap, norm = sfr_colormap([value for _, value in ranked], DEFAULT_CMAP, "log")
    write_model_colors(
        ranked,
        output_dir / "model_sfr_colors.csv",
        cmap,
        norm,
        bounds=sfr_bounds,
    )
    plot_phase_model_summaries(
        summary,
        ranked,
        output_dir,
        bounds=summary_bounds,
        sfr_bounds=sfr_bounds,
        dpi=dpi,
    )
    plot_phase_fraction_parameter_relations(
        summary,
        ranked,
        output_dir,
        bounds=summary_bounds,
        sfr_bounds=sfr_bounds,
        dpi=dpi,
    )
    plot_reduced_phase_fraction_parameter_relations(
        reduced_summary,
        ranked,
        output_dir,
        bounds=summary_bounds,
        sfr_bounds=sfr_bounds,
        dpi=dpi,
    )
    plot_phase_structure_dynamics_parameter_relations(
        correlation_summary,
        ranked,
        output_dir,
        phases=CORRELATION_PHASES,
        bounds=summary_bounds,
        sfr_bounds=sfr_bounds,
        dpi=dpi,
    )
    correlations = phase_parameter_correlations(
        correlation_summary, phases=CORRELATION_PHASES
    )
    correlation_path = output_dir / "phase_parameter_correlations.csv"
    _atomic_csv(correlations, correlation_path)
    print(f"Wrote {correlation_path}", flush=True)
    for slug, fields, title in PHASE_CORRELATION_FAMILIES:
        plot_phase_parameter_correlation_heatmaps(
            correlations,
            output_dir / f"phase_{slug}_parameter_correlations.png",
            fields,
            title=(
                f"{title}: {summary_bounds[0]:g}--"
                f"{summary_bounds[1]:g} Myr medians"
            ),
            phases=CORRELATION_PHASES,
            dpi=dpi,
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
            output_dir / f"phase_sigma_eff_z_{region}_evolution.png",
            ((f"sigma_eff_z_{region}", "-"),),
            ylabel=r"$\sigma_{{\rm eff},z}$ [km s$^{-1}$]",
            title=f"Phase effective vertical support speed in {region_label}",
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
