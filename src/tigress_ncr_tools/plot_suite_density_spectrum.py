#!/usr/bin/env python3
"""Build shear-aware gas-column overdensity spectra for an NCR suite."""

import argparse
from concurrent.futures import ProcessPoolExecutor
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
    short_model_name,
)
from .plot_suite_hst_evolution import (
    HISTORY_PARAMETER_COLOR_SPECS,
    model_history_parameters,
    sfr_colormap,
    write_model_colors,
)
from .projected_quantities import (
    PROJECTED_QUANTITIES,
    archive_projected_quantity,
    projected_quantity,
)
from .surface_density_stats import (
    annular_average_power_2d,
    annular_power_statistics_2d,
    band_power_quadrupole_2d,
    default_k_edges,
    power_spectral_density_2d,
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
DEFAULT_POWER2D_MEAN_ARCHIVE = "density_power_2d_time_mean.npz"
DEFAULT_POWER2D_MEAN_FIGURE = "density_power_2d_time_mean.png"
DEFAULT_POWER2D_MODE_LIMIT = 16.0
DEFAULT_SPECTRAL_BAND_PC = (64.0, 256.0)
POWER2D_DIRECTORY = "density_power_2d"
POWER2D_ARCHIVE_NAME = "density_power_2d.npz"
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


def overdensity_power_2d(
    frame,
    qshear,
    omega,
    *,
    field="nH",
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
):
    """Return the shear-remapped 2D PSD of ``Sigma/<Sigma> - 1``.

    The map is first transformed to periodic shearing coordinates. Fourier
    modes are then assigned their instantaneous physical wavenumbers through
    ``kx = kx0 + q Omega t_remap ky`` before annular averaging.
    """
    if not np.isclose(frame["theta"], 0.0):
        raise ValueError("density power spectra currently require theta0")
    sigma = np.asarray(frame["fields"][field], dtype=float)
    if sigma.ndim != 2 or not np.all(np.isfinite(sigma)):
        raise ValueError("surface density must be a finite 2D array")

    x = np.asarray(frame["x_centers"], dtype=float)
    lx = float(frame["x_edges"][-1] - frame["x_edges"][0])
    ly = float(frame["y_edges"][-1] - frame["y_edges"][0])
    remap_time, shear = residual_shear(frame["time"], qshear, omega, lx, ly)
    remapped = shear_remap_periodic(sigma, x, frame["y_spacing"], shear)
    mean_sigma = float(np.mean(remapped))
    if not np.isfinite(mean_sigma) or mean_sigma <= 0.0:
        raise ValueError("surface-density mean must be finite and positive")
    delta = remapped / mean_sigma - 1.0
    power_2d, kx0, ky = power_spectral_density_2d(
        delta,
        frame["x_spacing"],
        frame["y_spacing"],
        window=window,
        tukey_alpha=tukey_alpha,
        pad_factor=pad_factor,
    )
    return {
        "power_2d": power_2d,
        "kx0": kx0,
        "ky": ky,
        "mean_sigma": mean_sigma,
        "remap_time": remap_time,
        "shear": shear,
        "has_negative_sigma": bool(np.any(sigma < 0.0)),
        "box_size": min(lx, ly),
        "box_size_xy": np.asarray([lx, ly]),
        "pixel_size_xy": np.asarray(
            [float(frame["x_spacing"]), float(frame["y_spacing"])]
        ),
    }


def overdensity_power(
    frame,
    qshear,
    omega,
    *,
    field="nH",
    k_edges=None,
    k_bins=DEFAULT_K_BINS,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
):
    """Return a shear-aware annular spectrum through the explicit 2D PSD."""
    result = overdensity_power_2d(
        frame,
        qshear,
        omega,
        field=field,
        window=window,
        tukey_alpha=tukey_alpha,
        pad_factor=pad_factor,
    )
    if k_edges is None:
        k_edges = default_k_edges(frame["x_centers"], frame["y_centers"], bins=k_bins)
    power, count = annular_average_power_2d(
        result["power_2d"], result["kx0"], result["ky"], result["shear"], k_edges
    )
    result.update(
        {
            "power": power,
            "mode_count": count,
            "k_edges": np.asarray(k_edges),
        }
    )
    return result


def _atomic_savez(path, **data):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(path)


def model_power2d_archive(model, proj_id="theta0", quantity="gas"):
    """Return the per-model archive path for one quantity's 2D PSD series."""
    quantity_spec = projected_quantity(quantity)
    directory = f"{quantity_spec.slug}_power_2d"
    return Path(model) / "proj2d" / proj_id / directory / POWER2D_ARCHIVE_NAME


def _power2d_archive_matches(
    data, targets, window, tukey_alpha, pad_factor, field="nH"
):
    """Return whether a cached 2D series matches the requested configuration."""
    return (
        np.array_equal(np.asarray(data.get("target_time")), np.asarray(targets))
        and str(np.asarray(data.get("field", "nH")).item()) == str(field)
        and str(np.asarray(data.get("window", "")).item()) == str(window)
        and np.isclose(float(np.asarray(data.get("tukey_alpha", np.nan))), tukey_alpha)
        and np.isclose(float(np.asarray(data.get("pad_factor", np.nan))), pad_factor)
    )


def _power2d_file_matches(path, targets, window, tukey_alpha, pad_factor, field="nH"):
    """Check 2D-cache metadata without loading the large power array."""
    try:
        with np.load(path) as saved:
            metadata = {
                key: saved[key]
                for key in (
                    "target_time",
                    "field",
                    "window",
                    "tukey_alpha",
                    "pad_factor",
                )
            }
    except (OSError, KeyError, ValueError):
        return False
    return _power2d_archive_matches(
        metadata, targets, window, tukey_alpha, pad_factor, field=field
    )


def generate_model_power2d(
    model,
    *,
    quantity="gas",
    proj_id,
    targets,
    window,
    tukey_alpha,
    pad_factor,
    output,
):
    """Read projection maps and write one model's complete 2D PSD series."""
    quantity_spec = projected_quantity(quantity)
    parameter_path, qshear, omega = read_shear_parameters(model)
    index = projection_number_index(model, proj_id)
    guess_offset = 0
    rows = []
    common_kx0 = common_ky = None
    pixel_size_xy = box_size_xy = None
    for frame_index, target in enumerate(targets, start=1):
        path, stored_time, guess_offset = nearest_indexed_projection(
            index, float(target), guess_offset, tolerance=0.05
        )
        frame = read_proj2d(path, fields=quantity_spec.field)
        result = overdensity_power_2d(
            frame,
            qshear,
            omega,
            field=quantity_spec.field,
            window=window,
            tukey_alpha=tukey_alpha,
            pad_factor=pad_factor,
        )
        if common_kx0 is None:
            common_kx0 = result["kx0"]
            common_ky = result["ky"]
            pixel_size_xy = result["pixel_size_xy"]
            box_size_xy = result["box_size_xy"]
        elif not (
            np.array_equal(common_kx0, result["kx0"])
            and np.array_equal(common_ky, result["ky"])
            and np.allclose(pixel_size_xy, result["pixel_size_xy"])
            and np.allclose(box_size_xy, result["box_size_xy"])
        ):
            raise ValueError(f"2D spectrum grid changes within {model}")
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
        "projection_id": np.asarray(proj_id),
        "quantity": np.asarray(quantity_spec.key),
        "field": np.asarray(quantity_spec.field),
        "quantity_label": np.asarray(quantity_spec.label),
        "quantity_symbol": np.asarray(quantity_spec.symbol),
        "projected_physical_unit": np.asarray(quantity_spec.physical_unit),
        "delta_definition": np.asarray("map/<map>-1"),
        "coordinate_remap": np.asarray("g(x,y)=map(x,y-shear*x)"),
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
        "parameter_source": np.asarray(str(parameter_path)),
        "window": np.asarray(window),
        "tukey_alpha": np.asarray(tukey_alpha),
        "pad_factor": np.asarray(pad_factor),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(output, **data)
    print(f"Wrote {output}", flush=True)
    return data


def load_or_generate_model_power2d(
    model,
    *,
    quantity="gas",
    proj_id,
    targets,
    window,
    tukey_alpha,
    pad_factor,
    overwrite=False,
):
    """Load a compatible per-model 2D PSD series or generate it from maps."""
    quantity_spec = projected_quantity(quantity)
    output = model_power2d_archive(model, proj_id, quantity_spec.key)
    if output.exists() and not overwrite:
        if _power2d_file_matches(
            output,
            targets,
            window,
            tukey_alpha,
            pad_factor,
            field=quantity_spec.field,
        ):
            print(f"Loading existing {output}", flush=True)
            return load_spectrum_archive(output)
        print(f"Regenerating incompatible {output}", flush=True)
    return generate_model_power2d(
        model,
        quantity=quantity_spec.key,
        proj_id=proj_id,
        targets=targets,
        window=window,
        tukey_alpha=tukey_alpha,
        pad_factor=pad_factor,
        output=output,
    )


def _ensure_model_power2d(task):
    """Worker entry point that creates a cache if it is absent or stale."""
    (
        model,
        quantity,
        proj_id,
        targets,
        window,
        tukey_alpha,
        pad_factor,
        overwrite,
    ) = task
    quantity_spec = projected_quantity(quantity)
    output = model_power2d_archive(model, proj_id, quantity_spec.key)
    if (
        output.exists()
        and not overwrite
        and _power2d_file_matches(
            output,
            targets,
            window,
            tukey_alpha,
            pad_factor,
            field=quantity_spec.field,
        )
    ):
        print(f"Keeping compatible {output}", flush=True)
        return str(output)
    generate_model_power2d(
        model,
        quantity=quantity_spec.key,
        proj_id=proj_id,
        targets=targets,
        window=window,
        tukey_alpha=tukey_alpha,
        pad_factor=pad_factor,
        output=output,
    )
    return str(output)


def analyze_suite_power(
    ranked,
    *,
    quantity="gas",
    proj_id="theta0",
    start=DEFAULT_TIME_RANGE[0],
    stop=DEFAULT_TIME_RANGE[1],
    stride=1,
    k_bins=DEFAULT_K_BINS,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
    overwrite_2d=False,
    workers=1,
    output=None,
):
    """Generate/load 2D PSD series, then annularly reduce every model."""
    quantity_spec = projected_quantity(quantity)
    if proj_id != "theta0":
        raise ValueError("shear-aware density spectra currently require theta0")
    targets = np.arange(int(start), int(stop) + 1, int(stride), dtype=int)
    if targets.size == 0:
        raise ValueError("the requested time range contains no outputs")

    generation_tasks = [
        (
            model,
            quantity_spec.key,
            proj_id,
            targets,
            window,
            tukey_alpha,
            pad_factor,
            overwrite_2d,
        )
        for model, _ in ranked
    ]
    if int(workers) == 1:
        generated_paths = [_ensure_model_power2d(task) for task in generation_tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            generated_paths = list(
                executor.map(_ensure_model_power2d, generation_tasks)
            )

    model_names = []
    mean_sfr = []
    all_time = []
    all_power = []
    all_count = []
    all_q2 = []
    all_band_q2 = []
    all_band_mode_count = []
    all_mean_sigma = []
    all_remap_time = []
    all_shear = []
    all_negative = []
    all_qshear = []
    all_omega = []
    all_pixel_size = []
    all_box_size = []
    parameter_sources = []
    power2d_archives = []
    common_k_edges = None

    for model_index, ((model, model_sfr), generated_path) in enumerate(
        zip(ranked, generated_paths), start=1
    ):
        two_dimensional = load_spectrum_archive(generated_path)
        pixel_size_xy = np.asarray(two_dimensional["pixel_size_xy_pc"], dtype=float)
        box_size_xy = np.asarray(two_dimensional["box_size_xy_pc"], dtype=float)
        if common_k_edges is None:
            kmin = 2.0 * np.pi / np.min(box_size_xy)
            kmax = np.pi / np.max(pixel_size_xy)
            common_k_edges = np.geomspace(kmin, kmax, int(k_bins) + 1)
        powers = []
        counts = []
        quadrupoles = []
        band_quadrupoles = []
        band_mode_counts = []
        for time_index, shear in enumerate(two_dimensional["shear"]):
            power_2d = two_dimensional["power_2d"][time_index]
            power, count, q2 = annular_power_statistics_2d(
                power_2d,
                two_dimensional["kx0"],
                two_dimensional["ky"],
                shear,
                common_k_edges,
            )
            band_q2, band_mode_count = band_power_quadrupole_2d(
                power_2d,
                two_dimensional["kx0"],
                two_dimensional["ky"],
                shear,
                DEFAULT_SPECTRAL_BAND_PC,
            )
            powers.append(power)
            counts.append(count)
            quadrupoles.append(q2)
            band_quadrupoles.append(band_q2)
            band_mode_counts.append(band_mode_count)
        print(
            f"{model.name}: reduced {len(targets)} 2D spectra "
            f"({model_index}/{len(ranked)} models)",
            flush=True,
        )

        model_names.append(model.name)
        mean_sfr.append(model_sfr)
        all_time.append(two_dimensional["time"])
        all_power.append(powers)
        all_count.append(counts)
        all_q2.append(quadrupoles)
        all_band_q2.append(band_quadrupoles)
        all_band_mode_count.append(band_mode_counts)
        all_mean_sigma.append(two_dimensional["mean_sigma_code"])
        all_remap_time.append(two_dimensional["remap_time"])
        all_shear.append(two_dimensional["shear"])
        all_negative.append(two_dimensional["has_negative_sigma"])
        all_qshear.append(float(two_dimensional["qshear"]))
        all_omega.append(float(two_dimensional["omega_kms_per_pc"]))
        all_pixel_size.append(float(np.max(pixel_size_xy)))
        all_box_size.append(float(np.min(box_size_xy)))
        parameter_sources.append(
            str(np.asarray(two_dimensional["parameter_source"]).item())
        )
        power2d_archives.append(
            str(model_power2d_archive(model, proj_id, quantity_spec.key))
        )

    time = np.asarray(all_time)
    if np.any(np.ptp(time, axis=0) > 0.05):
        bad = np.flatnonzero(np.ptp(time, axis=0) > 0.05)
        raise ValueError(
            f"models are not time-aligned at target indices {bad.tolist()}"
        )
    power = np.asarray(all_power)
    count = np.asarray(all_count)
    q2 = np.asarray(all_q2)
    band_q2 = np.asarray(all_band_q2)
    data = {
        "model": np.asarray(model_names),
        "mean_sfr10": np.asarray(mean_sfr),
        "sfr_time_bounds": np.asarray(DEFAULT_SFR_RANGE),
        "projection_id": np.asarray(proj_id),
        "quantity": np.asarray(quantity_spec.key),
        "field": np.asarray(quantity_spec.field),
        "quantity_label": np.asarray(quantity_spec.label),
        "quantity_symbol": np.asarray(quantity_spec.symbol),
        "projected_physical_unit": np.asarray(quantity_spec.physical_unit),
        "delta_definition": np.asarray("map/<map>-1"),
        "coordinate_remap": np.asarray("g(x,y)=map(x,y-shear*x)"),
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
        "q2_real": q2.real,
        "q2_imaginary": q2.imag,
        "anisotropy_amplitude": np.abs(q2),
        "anisotropy_angle_rad": 0.5 * np.angle(q2),
        "anisotropy_definition": np.asarray(
            "Q2=sum(P*exp(2i*phi))/sum(P); A2=abs(Q2); phi2=arg(Q2)/2"
        ),
        "anisotropy_band_wavelength_pc": np.asarray(DEFAULT_SPECTRAL_BAND_PC),
        "anisotropy_band_q2_real_time": band_q2.real,
        "anisotropy_band_q2_imaginary_time": band_q2.imag,
        "anisotropy_band_amplitude_time": np.abs(band_q2),
        "anisotropy_band_angle_rad_time": 0.5 * np.angle(band_q2),
        "anisotropy_band_mode_count_time": np.asarray(all_band_mode_count),
        "mean_sigma_code": np.asarray(all_mean_sigma),
        "remap_time": np.asarray(all_remap_time),
        "shear": np.asarray(all_shear),
        "has_negative_sigma": np.asarray(all_negative),
        "qshear": np.asarray(all_qshear),
        "omega_kms_per_pc": np.asarray(all_omega),
        "pixel_size_pc": np.asarray(all_pixel_size),
        "box_size_pc": np.asarray(all_box_size),
        "parameter_source": np.asarray(parameter_sources),
        "power2d_archive": np.asarray(power2d_archives),
        "power2d_storage_dtype": np.asarray("float32"),
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


def _uniform_center_edges(centers):
    """Return bin edges for uniformly spaced cell centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError(
            "centers must be a one-dimensional array with at least 2 cells"
        )
    spacing = float(np.median(np.diff(centers)))
    if spacing <= 0.0 or not np.allclose(np.diff(centers), spacing):
        raise ValueError("centers must be uniformly increasing")
    return np.concatenate(([centers[0] - 0.5 * spacing], centers + 0.5 * spacing))


def physical_time_mean_power_2d(
    power_2d,
    kx0,
    ky,
    shear,
    box_size_xy,
    use,
    *,
    chunk_size=50,
):
    """Deposit selected spectra on a fixed physical-k grid and average them."""
    power_2d = np.asarray(power_2d, dtype=float)
    kx0 = np.asarray(kx0, dtype=float)
    ky = np.asarray(ky, dtype=float)
    shear = np.asarray(shear, dtype=float)
    box_size_xy = np.asarray(box_size_xy, dtype=float)
    use = np.asarray(use, dtype=bool)
    if power_2d.ndim != 3 or power_2d.shape[1:] != (ky.size, kx0.size):
        raise ValueError("power_2d must have shape (time, ky, kx0)")
    if shear.shape != (power_2d.shape[0],) or use.shape != shear.shape:
        raise ValueError("shear and use must match the power time axis")
    if box_size_xy.shape != (2,) or np.any(box_size_xy <= 0.0):
        raise ValueError("box_size_xy must contain positive Lx and Ly")
    if chunk_size <= 0 or not np.any(use):
        raise ValueError("chunk_size must be positive and use must select snapshots")

    lx, ly = box_size_xy
    x_mode = kx0 * lx / (2.0 * np.pi)
    y_mode = ky * ly / (2.0 * np.pi)
    x_order = np.argsort(x_mode)
    y_order = np.argsort(y_mode)
    x_mode = x_mode[x_order]
    y_mode = y_mode[y_order]
    x_edges = _uniform_center_edges(x_mode)
    y_edges = _uniform_center_edges(y_mode)
    power_sum = np.zeros((y_mode.size, x_mode.size), dtype=float)
    sample_count = np.zeros(power_sum.shape, dtype=np.int64)
    selected_indices = np.flatnonzero(use)
    for start in range(0, selected_indices.size, int(chunk_size)):
        indices = selected_indices[start : start + int(chunk_size)]
        selected = power_2d[indices][:, y_order][:, :, x_order]
        x_physical = x_mode[None, None, :] + (
            shear[indices, None, None] * y_mode[None, :, None] * (lx / ly)
        )
        x_physical = np.broadcast_to(x_physical, selected.shape)
        y_physical = np.broadcast_to(y_mode[None, :, None], selected.shape)
        finite = np.isfinite(selected) & (selected >= 0.0)
        weighted, _, _ = np.histogram2d(
            y_physical[finite],
            x_physical[finite],
            bins=(y_edges, x_edges),
            weights=selected[finite],
        )
        counts, _, _ = np.histogram2d(
            y_physical[finite],
            x_physical[finite],
            bins=(y_edges, x_edges),
        )
        power_sum += weighted
        sample_count += counts.astype(np.int64)
    mean_power = np.full(power_sum.shape, np.nan)
    np.divide(
        power_sum,
        sample_count,
        out=mean_power,
        where=sample_count > 0,
    )
    return mean_power, sample_count, x_mode, y_mode


def _time_mean_power2d_archive(task):
    """Worker entry point for one model's physical-grid time mean."""
    model, archive_path, bounds = task
    archive = load_spectrum_archive(archive_path)
    stored_model = str(np.asarray(archive["model"]).item())
    if stored_model != model:
        raise ValueError(f"2D archive model {stored_model} does not match {model}")
    selection_time = np.asarray(
        archive.get("target_time", archive["time"]), dtype=float
    )
    use = (selection_time >= bounds[0]) & (selection_time <= bounds[1])
    mean_power, sample_count, x_mode, y_mode = physical_time_mean_power_2d(
        archive["power_2d"],
        archive["kx0"],
        archive["ky"],
        archive["shear"],
        archive["box_size_xy_pc"],
        use,
    )
    return {
        "model": model,
        "mean_power_2d": mean_power.astype(np.float32),
        "mode_sample_count": sample_count.astype(np.int32),
        "kx_mode": x_mode,
        "ky_mode": y_mode,
        "time_sample_count": int(np.count_nonzero(use)),
        "box_size_xy_pc": np.asarray(archive["box_size_xy_pc"], dtype=float),
    }


def calculate_suite_time_mean_power_2d(
    data,
    *,
    bounds=DEFAULT_SFR_RANGE,
    workers=1,
    output=None,
):
    """Build physical-grid time-mean 2D spectra from every model cache."""
    quantity_spec = archive_projected_quantity(data)
    model_names = np.asarray(data["model"]).astype(str)
    archives = np.asarray(data["power2d_archive"]).astype(str)
    if archives.shape != model_names.shape:
        raise ValueError("power2d archive paths must match the model axis")
    if workers <= 0:
        raise ValueError("workers must be positive")
    tasks = [
        (model, archive_path, tuple(bounds))
        for model, archive_path in zip(model_names, archives)
    ]
    if int(workers) == 1:
        model_results = [_time_mean_power2d_archive(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            model_results = list(executor.map(_time_mean_power2d_archive, tasks))
    means = []
    counts = []
    time_counts = []
    common_x_mode = common_y_mode = None
    box_sizes = []
    for result in model_results:
        model = result["model"]
        x_mode = result["kx_mode"]
        y_mode = result["ky_mode"]
        if common_x_mode is None:
            common_x_mode = x_mode
            common_y_mode = y_mode
        elif not (
            np.allclose(common_x_mode, x_mode) and np.allclose(common_y_mode, y_mode)
        ):
            raise ValueError("2D Fourier grids differ between suite models")
        means.append(result["mean_power_2d"])
        counts.append(result["mode_sample_count"])
        time_counts.append(result["time_sample_count"])
        box_sizes.append(result["box_size_xy_pc"])
        print(f"Averaged physical 2D spectra for {model}", flush=True)
    result = {
        "model": model_names,
        "mean_sfr10": np.asarray(data["mean_sfr10"], dtype=float),
        "quantity": np.asarray(quantity_spec.key),
        "field": np.asarray(quantity_spec.field),
        "quantity_label": np.asarray(quantity_spec.label),
        "quantity_symbol": np.asarray(quantity_spec.symbol),
        "projected_physical_unit": np.asarray(quantity_spec.physical_unit),
        "time_bounds": np.asarray(bounds, dtype=float),
        "time_sample_count": np.asarray(time_counts, dtype=int),
        "mean_power_2d": np.asarray(means),
        "mode_sample_count": np.asarray(counts),
        "kx_mode": common_x_mode,
        "ky_mode": common_y_mode,
        "box_size_xy_pc": np.asarray(box_sizes),
        "power2d_archive": archives,
        "power_unit": np.asarray("pc^2"),
        "coordinate_definition": np.asarray("kx_mode=kx*Lx/(2pi), ky_mode=ky*Ly/(2pi)"),
        "physical_k_mapping": np.asarray("kx=kx0+shear*ky"),
        "deposition": np.asarray(
            "arithmetic mean after Cartesian nearest-bin deposition"
        ),
        "excluded_models": np.asarray(sorted(EXCLUDED_MODELS)),
    }
    if output is not None:
        _atomic_savez(output, **result)
        print(f"Wrote {output}", flush=True)
    return result


def plot_suite_time_mean_power_2d(
    data,
    ranked,
    output,
    *,
    mode_limit=DEFAULT_POWER2D_MODE_LIMIT,
    cmap_name=DEFAULT_CMAP,
    dpi=180,
):
    """Plot the SFR-ranked 4-by-8 suite of physical time-mean 2D spectra."""
    quantity_spec = archive_projected_quantity(data)
    if list(np.asarray(data["model"]).astype(str)) != [
        model.name for model, _ in ranked
    ]:
        raise ValueError("2D mean archive and ranked model ordering differ")
    x_mode = np.asarray(data["kx_mode"], dtype=float)
    y_mode = np.asarray(data["ky_mode"], dtype=float)
    power = np.asarray(data["mean_power_2d"], dtype=float).copy()
    if mode_limit <= 0.0:
        raise ValueError("mode_limit must be positive")
    x_use = np.abs(x_mode) <= mode_limit
    y_use = np.abs(y_mode) <= mode_limit
    if np.count_nonzero(x_use) < 2 or np.count_nonzero(y_use) < 2:
        raise ValueError("mode_limit selects fewer than two Fourier cells")
    origin_x = int(np.argmin(np.abs(x_mode)))
    origin_y = int(np.argmin(np.abs(y_mode)))
    power[:, origin_y, origin_x] = np.nan
    displayed = power[:, y_use][:, :, x_use]
    limits = _positive_limits(displayed, percentiles=(2.0, 99.8))
    norm = mpl.colors.LogNorm(vmin=limits[0], vmax=limits[1])
    colormap = mpl.colormaps[cmap_name].copy()
    colormap.set_bad("white")
    x_display = x_mode[x_use]
    y_display = y_mode[y_use]
    x_edges = _uniform_center_edges(x_display)
    y_edges = _uniform_center_edges(y_display)
    fig, axes = plt.subplots(
        4,
        8,
        figsize=(18.2, 10.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for index, axis in enumerate(axes.flat):
        if index >= len(ranked):
            axis.axis("off")
            continue
        model, _ = ranked[index]
        image = axis.imshow(
            displayed[index],
            origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            cmap=colormap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axis.axhline(0.0, color="white", alpha=0.16, linewidth=0.45)
        axis.axvline(0.0, color="white", alpha=0.16, linewidth=0.45)
        axis.set_title(f"{index + 1}. {short_model_name(model)}", fontsize=7.5)
        axis.tick_params(direction="in", labelsize=7, top=True, right=True)
    if image is None:
        raise ValueError("no models available for the 2D spectrum figure")
    bounds = np.asarray(data["time_bounds"], dtype=float)
    fig.supxlabel(r"physical $k_xL_x/(2\pi)$")
    fig.supylabel(r"physical $k_yL_y/(2\pi)$")
    fig.suptitle(
        f"Shear-aware {quantity_spec.label} 2D power spectra: "
        f"{bounds[0]:g}--{bounds[1]:g} Myr arithmetic means",
        fontsize=13,
    )
    color_axis = fig.add_axes((0.36, 0.045, 0.28, 0.018))
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(r"$\langle P_{\delta,2{\rm D}}\rangle_t\ [{\rm pc}^2]$")
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.09,
        top=0.93,
        hspace=0.28,
        wspace=0.08,
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


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


def spectral_slope_alpha(k, power, wavelength_bounds=DEFAULT_SPECTRAL_BAND_PC):
    """Fit ``P_delta proportional to k^-alpha`` in a wavelength band."""
    k = np.asarray(k, dtype=float)
    power = np.asarray(power, dtype=float)
    lower, upper = (float(value) for value in wavelength_bounds)
    if not np.isfinite(lower + upper) or not 0.0 < lower < upper:
        raise ValueError("wavelength bounds must be finite, positive, and increasing")
    wavelength = 2.0 * np.pi / k
    valid = (
        np.isfinite(k)
        & (k > 0.0)
        & np.isfinite(power)
        & (power > 0.0)
        & (wavelength > lower)
        & (wavelength < upper)
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
    scales = np.full(power.shape[:2], np.nan)
    slopes = np.full(power.shape[:2], np.nan)
    fit_counts = np.zeros(power.shape[:2], dtype=int)
    for model_index in range(power.shape[0]):
        for time_index in range(power.shape[1]):
            scale = integral_scale(k_edges, power[model_index, time_index])
            slope, fit_count = spectral_slope_alpha(
                k,
                power[model_index, time_index],
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


def _axial_angle_statistics(values):
    """Return circular summaries for angles equivalent modulo pi."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (np.nan,) * 5 + (0,)
    vector = np.mean(np.exp(2j * finite))
    center = 0.5 * np.angle(vector)
    residual = 0.5 * np.angle(np.exp(2j * (finite - center)))
    percentile16, median_residual, percentile84 = np.percentile(
        residual, [16.0, 50.0, 84.0]
    )
    low = min(center, center + percentile16)
    high = max(center, center + percentile84)
    return (
        float(center),
        float(np.abs(vector)),
        float(center + median_residual),
        float(low),
        float(high),
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
            k, mean_power[index]
        )
        use = (time[index] >= bounds[0]) & (time[index] <= bounds[1])
        (
            scale_mean,
            scale_std,
            scale_median,
            scale_percentile16,
            scale_percentile84,
            scale_count,
        ) = _finite_statistics(time_diagnostics["integral_scale_time_pc"][index, use])
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
        (
            amplitude_mean,
            amplitude_std,
            amplitude_median,
            amplitude_percentile16,
            amplitude_percentile84,
            amplitude_count,
        ) = _finite_statistics(
            np.asarray(data["anisotropy_band_amplitude_time"])[index, use]
        )
        (
            angle_circular_mean,
            angle_resultant_length,
            angle_median,
            angle_percentile16,
            angle_percentile84,
            angle_count,
        ) = _axial_angle_statistics(
            np.asarray(data["anisotropy_band_angle_rad_time"])[index, use]
        )
        angle_factor = 180.0 / np.pi
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
                "anisotropy_band_amplitude_time_mean": amplitude_mean,
                "anisotropy_band_amplitude_time_std": amplitude_std,
                "anisotropy_band_amplitude_time_median": amplitude_median,
                "anisotropy_band_amplitude_time_percentile16": (amplitude_percentile16),
                "anisotropy_band_amplitude_time_percentile84": (amplitude_percentile84),
                "anisotropy_band_amplitude_time_count": amplitude_count,
                "anisotropy_band_angle_circular_mean_deg": (
                    angle_circular_mean * angle_factor
                ),
                "anisotropy_band_angle_resultant_length": angle_resultant_length,
                "anisotropy_band_angle_time_median_deg": (angle_median * angle_factor),
                "anisotropy_band_angle_time_percentile16_deg": (
                    angle_percentile16 * angle_factor
                ),
                "anisotropy_band_angle_time_percentile84_deg": (
                    angle_percentile84 * angle_factor
                ),
                "anisotropy_band_angle_time_count": angle_count,
                "integral_scale_pc": mean_spectrum_scale,
                "spectral_slope_alpha": mean_spectrum_alpha,
                "slope_fit_bin_count": mean_spectrum_fit_count,
                "slope_fit_lambda_min_pc": DEFAULT_SPECTRAL_BAND_PC[0],
                "slope_fit_lambda_max_pc": DEFAULT_SPECTRAL_BAND_PC[1],
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
        "stellar_surface_density",
        "stellar_scale_height",
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
        "anisotropy_band_amplitude_time_mean",
        "anisotropy_band_amplitude_time_std",
        "anisotropy_band_amplitude_time_median",
        "anisotropy_band_amplitude_time_percentile16",
        "anisotropy_band_amplitude_time_percentile84",
        "anisotropy_band_amplitude_time_count",
        "anisotropy_band_angle_circular_mean_deg",
        "anisotropy_band_angle_resultant_length",
        "anisotropy_band_angle_time_median_deg",
        "anisotropy_band_angle_time_percentile16_deg",
        "anisotropy_band_angle_time_percentile84_deg",
        "anisotropy_band_angle_time_count",
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
        "P_delta(k) proportional to k^-alpha for 64 pc < 2pi/k < 256 pc"
    )
    augmented["anisotropy_band_definition"] = np.asarray(
        "power-weighted Q2 over 64 pc < 2pi/k < 256 pc"
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


def _plot_percentile_points(
    axis, x, median, percentile16, percentile84, color, cmap, norm
):
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
            r"$\phi_{2,64-256}\ [{\rm deg}]$",
        ),
    )
    fig, axes = plt.subplots(4, 3, figsize=(14.2, 14.2), sharex="col")
    for column, (x_field, xlabel) in enumerate(x_specifications):
        x = summary[x_field].to_numpy(dtype=float)
        for row, (field, low_field, high_field, ylabel) in enumerate(y_specifications):
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
            if row == 2:
                axis.set_ylim(0.0, 1.0)
            if row == 3:
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
        "Gas-column spectrum and 64--256 pc quadrupole correlations: "
        "200--600 Myr temporal summaries",
        fontsize=13,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.14, top=0.94, hspace=0.26, wspace=0.22
    )
    fig.savefig(output, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}", flush=True)


def create_spectrum_figure(
    k,
    power,
    anisotropy_amplitude,
    ranked,
    *,
    title,
    cmap,
    norm,
    power_limits,
    anisotropy_limits,
    box_size,
):
    """Create the two-panel angle-averaged power and anisotropy figure."""
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
        anisotropy_valid = np.isfinite(anisotropy_amplitude[index])
        second = axes[1].plot(
            mode[anisotropy_valid],
            anisotropy_amplitude[index, anisotropy_valid],
            color=color,
            linewidth=1.0,
            alpha=0.82,
        )[0]
        lines[index] = (first, second)
    axes[0].set_ylabel(r"$P_\delta(k)\;[\mathrm{pc}^2]$")
    axes[1].set_ylabel(r"$A_2(k)=|Q_2(k)|$")
    for axis in axes:
        axis.set_xscale("log")
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
    axes[0].set_yscale("log")
    axes[0].set_ylim(power_limits)
    axes[1].set_ylim(anisotropy_limits)
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
    """Plot mean power and median anisotropy over t=200--600."""
    quantity_spec = archive_projected_quantity(data)
    k = np.asarray(data["k_centers"])
    power = time_mean_power(data, DEFAULT_SFR_RANGE)
    time = np.asarray(data["time"], dtype=float)
    anisotropy_time = np.asarray(data["anisotropy_amplitude"], dtype=float)
    anisotropy = np.full((power.shape[0], power.shape[1]), np.nan)
    for index in range(power.shape[0]):
        use = (time[index] >= DEFAULT_SFR_RANGE[0]) & (
            time[index] <= DEFAULT_SFR_RANGE[1]
        )
        anisotropy[index] = np.nanmedian(anisotropy_time[index, use], axis=0)
    mean_sfr = np.asarray(data["mean_sfr10"])
    cmap, norm = sfr_colormap(mean_sfr, cmap_name, "log")
    box_size = float(np.asarray(data["box_size_pc"]).flat[0])
    fig, _, _, _ = create_spectrum_figure(
        k,
        power,
        anisotropy,
        ranked,
        title=(
            f"Shear-aware {quantity_spec.label} overdensity spectrum: "
            r"mean $P_\delta$ and median $A_2$ over $200\leq t\leq600$"
        ),
        cmap=cmap,
        norm=norm,
        power_limits=_positive_limits(power),
        anisotropy_limits=(0.0, 1.0),
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
    quantity_spec = archive_projected_quantity(data)
    frame_prefix = f"{quantity_spec.slug}_power_spectrum"
    k = np.asarray(data["k_centers"])
    power = np.asarray(data["power_delta"])
    anisotropy = np.asarray(data["anisotropy_amplitude"])
    box_size = float(np.asarray(data["box_size_pc"]).flat[0])
    mode = k * box_size / (2.0 * np.pi)
    cmap, norm = sfr_colormap(data["mean_sfr10"], cmap_name, "log")
    power_limits = _positive_limits(power)
    fig = axes = lines = title_text = None
    try:
        for time_index, target in enumerate(np.asarray(data["target_time"], dtype=int)):
            output = output_dir / f"{frame_prefix}.{target:04d}.png"
            if output.exists() and not overwrite:
                print(f"Skipping existing {output}", flush=True)
                continue
            current = power[:, time_index, :]
            if fig is None:
                fig, axes, lines, title_text = create_spectrum_figure(
                    k,
                    current,
                    anisotropy[:, time_index, :],
                    ranked,
                    title=(
                        f"Shear-aware {quantity_spec.label} overdensity spectrum: "
                        rf"$t={target:g}$"
                    ),
                    cmap=cmap,
                    norm=norm,
                    power_limits=power_limits,
                    anisotropy_limits=(0.0, 1.0),
                    box_size=box_size,
                )
            else:
                for model_index, (first, second) in enumerate(lines):
                    values = current[model_index]
                    valid = np.isfinite(values) & (values > 0.0)
                    first.set_data(mode[valid], values[valid])
                    anisotropy_values = anisotropy[model_index, time_index]
                    anisotropy_valid = np.isfinite(anisotropy_values)
                    second.set_data(
                        mode[anisotropy_valid], anisotropy_values[anisotropy_valid]
                    )
                title_text.set_text(
                    f"Shear-aware {quantity_spec.label} overdensity spectrum: "
                    rf"$t={target:g}$"
                )
            fig.savefig(output, dpi=dpi, facecolor="white")
            print(f"Wrote {output}", flush=True)
    finally:
        if fig is not None:
            plt.close(fig)
    movie_path = output_dir / f"{frame_prefix}_evolution.mp4"
    make_grid_movie(output_dir, frame_prefix, movie_path, fps)
    return movie_path


def render_suite_density_spectrum(
    suite,
    *,
    quantity="gas",
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
    power2d_mode_limit=DEFAULT_POWER2D_MODE_LIMIT,
    dpi=180,
    overwrite=False,
    overwrite_2d=False,
    workers=1,
    movie=False,
    fps=30.0,
):
    """Analyze, cache, and plot suite projected-quantity spectra."""
    suite = Path(suite).expanduser()
    quantity_spec = projected_quantity(quantity)
    output_dir = (
        Path(output_dir)
        if output_dir
        else suite / f"{quantity_spec.slug}_power_spectrum_theta0"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [
        model
        for model in discover_evolution_models(suite, model_glob, proj_id)
        if model.name not in EXCLUDED_MODELS
    ]
    ranked = rank_models_by_sfr(models, bounds=sfr_bounds, max_rows=10000)
    archive = output_dir / f"{quantity_spec.slug}_power_spectra.npz"
    missing_power2d = [
        model_power2d_archive(model, proj_id, quantity_spec.key)
        for model, _ in ranked
        if not model_power2d_archive(model, proj_id, quantity_spec.key).exists()
    ]
    rebuild_reduction = overwrite or overwrite_2d or bool(missing_power2d)
    if archive.exists() and not rebuild_reduction:
        with np.load(archive) as saved:
            required_quadrupole_fields = {
                "q2_real",
                "q2_imaginary",
                "anisotropy_amplitude",
                "anisotropy_angle_rad",
                "anisotropy_band_amplitude_time",
                "anisotropy_band_angle_rad_time",
            }
            if not required_quadrupole_fields.issubset(saved.files):
                rebuild_reduction = True
                print(
                    "Rebuilding annular reduction to add Q2 anisotropy diagnostics",
                    flush=True,
                )
    if missing_power2d:
        print(
            f"Generating {len(missing_power2d)} missing per-model 2D PSD archives",
            flush=True,
        )
    removed = ()
    if archive.exists() and not rebuild_reduction:
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
        if str(np.asarray(data.get("field", "nH")).item()) != quantity_spec.field:
            raise ValueError("cached spectrum field does not match requested quantity")
    else:
        data = analyze_suite_power(
            ranked,
            quantity=quantity_spec.key,
            proj_id=proj_id,
            start=start,
            stop=stop,
            stride=stride,
            k_bins=k_bins,
            window=window,
            tukey_alpha=tukey_alpha,
            pad_factor=pad_factor,
            overwrite_2d=overwrite_2d,
            workers=workers,
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
    diagnostic_name = f"{quantity_spec.slug}_power_spectrum_integral_scale_slope"
    correlation_name = f"{quantity_spec.slug}_power_spectrum_correlations"
    diagnostic_csv = output_dir / f"{diagnostic_name}.csv"
    _atomic_csv(diagnostic_summary, diagnostic_csv)
    print(f"Wrote {diagnostic_csv}", flush=True)
    summary = output_dir / f"{quantity_spec.slug}_power_spectrum_time_mean.png"
    cmap, norm = plot_time_mean_spectrum(
        data, ranked, summary, cmap_name=cmap_name, dpi=dpi
    )
    mean_power2d_path = output_dir / f"{quantity_spec.slug}_power_2d_time_mean.npz"
    mean_power2d = None
    if mean_power2d_path.exists() and not (overwrite or overwrite_2d):
        candidate = load_spectrum_archive(mean_power2d_path)
        candidate_bounds = np.asarray(candidate.get("time_bounds", ()), dtype=float)
        if (
            list(np.asarray(candidate.get("model", ())).astype(str))
            == [model.name for model, _ in ranked]
            and candidate_bounds.shape == (2,)
            and np.allclose(candidate_bounds, sfr_bounds)
            and str(np.asarray(candidate.get("field", "nH")).item())
            == quantity_spec.field
        ):
            mean_power2d = candidate
            print(f"Loading existing {mean_power2d_path}", flush=True)
    if mean_power2d is None:
        mean_power2d = calculate_suite_time_mean_power_2d(
            data,
            bounds=sfr_bounds,
            workers=workers,
            output=mean_power2d_path,
        )
    plot_suite_time_mean_power_2d(
        mean_power2d,
        ranked,
        output_dir / f"{quantity_spec.slug}_power_2d_time_mean.png",
        mode_limit=power2d_mode_limit,
        cmap_name=cmap_name,
        dpi=dpi,
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
        diagnostic_output = output_dir / f"{diagnostic_name}{suffix}.png"
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
        output_dir / f"{correlation_name}.png",
        dpi=dpi,
    )
    if movie:
        movie_manifest = (
            output_dir / f"{quantity_spec.slug}_power_spectrum_movie_models.txt"
        )
        expected_manifest = (
            "axis=kL/(2pi); top=lambda=2pi/k; second_panel=A2; "
            f"field={quantity_spec.field}; version=3\n"
            + "\n".join(str(name) for name in data["model"])
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
            overwrite=overwrite or overwrite_2d or bool(removed) or movie_stale,
        )
        _atomic_text(expected_manifest, movie_manifest)
    return data, ranked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB)
    parser.add_argument("--projection", default="theta0")
    parser.add_argument(
        "--quantity", choices=tuple(PROJECTED_QUANTITIES), default="gas"
    )
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
    parser.add_argument(
        "--power2d-mode-limit", type=float, default=DEFAULT_POWER2D_MODE_LIMIT
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-2d", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--movie", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.start < 0 or args.stop < args.start:
        parser.error("require 0 <= --start <= --stop")
    if args.stride <= 0 or args.k_bins <= 0 or args.workers <= 0:
        parser.error("--stride, --k-bins, and --workers must be positive")
    if args.sfr_stop <= args.sfr_start:
        parser.error("--sfr-stop must be greater than --sfr-start")
    if not 0.0 <= args.tukey_alpha <= 1.0:
        parser.error("--tukey-alpha must be between zero and one")
    if args.pad_factor < 1.0:
        parser.error("--pad-factor must be at least one")
    if args.power2d_mode_limit <= 0.0:
        parser.error("--power2d-mode-limit must be positive")
    if args.dpi <= 0 or args.fps <= 0:
        parser.error("--dpi and --fps must be positive")
    render_suite_density_spectrum(
        args.suite,
        quantity=args.quantity,
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
        power2d_mode_limit=args.power2d_mode_limit,
        dpi=args.dpi,
        overwrite=args.overwrite,
        overwrite_2d=args.overwrite_2d,
        workers=args.workers,
        movie=args.movie,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
