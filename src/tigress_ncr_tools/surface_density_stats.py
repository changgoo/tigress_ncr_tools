#!/usr/bin/env python3
"""PDFs and shear-aware power spectra of theta0 gas surface-density maps."""

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pathena.proj2d_reader import read_proj2d
from pathena.units import star_particle_units
from tigress_ncr_tools.plot_suite_projections import (
    DEFAULT_SUITE,
    discover_late_models,
    model_size,
    projection_series,
)

DEFAULT_PDF_RANGE = (-2.0, 3.0)
DEFAULT_DELTA_RANGE = (-1.0, 30.0)
DEFAULT_S_RANGE = (-6.0, 4.0)
DEFAULT_PDF_BINS = 100
DEFAULT_K_BINS = 40
STATISTICS_NAME = "surface_density_statistics.npz"
NUMBER_PATTERN = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+\-]?\d+)?"


def logarithmic_bin_edges(start, stop, bins):
    """Return reproducible positive log-spaced bin edges.

    NumPy 2 changed ``geomspace`` by a few ulps relative to NumPy 1. Fourier
    modes can lie exactly on these edges, so those harmless edge changes can
    move modes between neighboring annuli. Scalar powers plus a one-ulp
    inward bias give the boundary a stable, explicit upper-bin convention.
    """
    start = float(start)
    stop = float(stop)
    bins = int(bins)
    if not 0.0 < start < stop:
        raise ValueError("logarithmic bin bounds must satisfy 0 < start < stop")
    if bins < 1:
        raise ValueError("logarithmic bins must be positive")
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    edges = np.asarray(
        [
            10.0 ** (log_start + (log_stop - log_start) * index / bins)
            for index in range(bins + 1)
        ]
    )
    edges[0] = start
    edges[-1] = stop
    edges[1:-1] = np.nextafter(edges[1:-1], 0.0)
    return edges


def read_athinput_section(path, section="problem"):
    """Return numeric assignments from one Athena input-file section."""
    values = {}
    current = None
    section = section.lower()
    with Path(path).open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.split("#", 1)[0].strip()
            match = re.fullmatch(r"<\s*([^>]+?)\s*>", line)
            if match:
                current = match.group(1).strip().lower()
                continue
            if current != section or "=" not in line:
                continue
            key, raw_value = (part.strip() for part in line.split("=", 1))
            token = raw_value.split()[0].replace("D", "E").replace("d", "e")
            try:
                values[key] = float(token)
            except ValueError:
                continue
    return values


def _batch_problem_parameters(path):
    """Return numeric ``problem/name=value`` overrides from a batch script."""
    text = Path(path).read_text(errors="replace")
    pattern = re.compile(
        rf"(?:^|\s)problem/([A-Za-z_][A-Za-z0-9_]*)=({NUMBER_PATTERN})(?=\s|$)"
    )
    return {
        match.group(1): float(match.group(2).replace("D", "E").replace("d", "e"))
        for match in pattern.finditer(text)
    }


def _consistent_shear_parameters(found, model):
    """Validate parameter sources and return the first consistent tuple."""
    reference = found[0][1:]
    inconsistent = [item for item in found[1:] if not np.allclose(item[1:], reference)]
    if inconsistent:
        details = ", ".join(
            f"{path.name}: q={q:g}, Omega={omega:g}"
            for path, q, omega in found
        )
        raise ValueError(f"inconsistent shear parameters in {model}: {details}")
    return found[0]


def read_shear_parameters(model):
    """Read effective ``qshear`` and ``Omega``, honoring runtime overrides."""
    model = Path(model)
    scripts = sorted(model.glob("*.slurm")) + sorted(model.glob("*.pbs"))
    overridden = []
    for path in scripts:
        values = _batch_problem_parameters(path)
        if "qshear" in values and "Omega" in values:
            overridden.append((path, values["qshear"], values["Omega"]))
    if overridden:
        return _consistent_shear_parameters(overridden, model)

    paths = sorted(model.glob("athinput*"))
    if not paths:
        raise FileNotFoundError(f"no batch override or athinput* found in {model}")
    found = []
    for path in paths:
        values = read_athinput_section(path)
        if "qshear" in values and "Omega" in values:
            found.append((path, values["qshear"], values["Omega"]))
    if not found:
        raise KeyError(
            f"qshear and Omega were not found in runtime overrides or <problem> in {model}"
        )
    return _consistent_shear_parameters(found, model)


def residual_shear(time, qshear, omega, lx, ly):
    """Return Athena's residual remap time and dimensionless shear slope."""
    qomega_lx = float(qshear) * float(omega) * float(lx)
    if qomega_lx == 0.0:
        return float(time), 0.0
    periods = np.trunc(qomega_lx * float(time) / float(ly))
    remap_time = float(time) - periods * float(ly) / qomega_lx
    return remap_time, float(qshear) * float(omega) * remap_time


def shear_remap_periodic(values, x_centers, dy, shear):
    """Remap ``f`` to ``g(x,y)=f(x,y-shear*x)`` using Fourier shifts in y."""
    values = np.asarray(values, dtype=float)
    x_centers = np.asarray(x_centers, dtype=float)
    if values.ndim != 2 or values.shape[1] != x_centers.size:
        raise ValueError("values must have shape (ny, len(x_centers))")
    ky = 2.0 * np.pi * np.fft.fftfreq(values.shape[0], d=float(dy))
    transformed = np.fft.fft(values, axis=0)
    phase = np.exp(-1j * ky[:, None] * float(shear) * x_centers[None, :])
    remapped = np.fft.ifft(transformed * phase, axis=0)
    return remapped.real


def normalized_pdf(values, edges, weights=None):
    """Return a PDF normalized by all samples, including out-of-range tails."""
    values = np.asarray(values, dtype=float).ravel()
    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)
    if edges.ndim != 1 or edges.size < 2 or np.any(widths <= 0.0):
        raise ValueError("PDF edges must be a strictly increasing 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("PDF values must be finite")
    if weights is None:
        normalization = values.size
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.shape != values.shape or not np.all(np.isfinite(weights)):
            raise ValueError("PDF weights must be finite and match values")
        normalization = weights.sum()
    if normalization <= 0.0:
        raise ValueError("PDF normalization must be positive")
    histogram = np.histogram(values, bins=edges, weights=weights)[0]
    return histogram / (normalization * widths)


def pdfs_log10_sigma(sigma, edges):
    """Return area- and mass-weighted PDFs per dex of surface density."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("surface density must be a finite, positive 2D array")
    log_sigma = np.log10(sigma.ravel())
    area_pdf = normalized_pdf(log_sigma, edges)
    mass_pdf = normalized_pdf(log_sigma, edges, weights=sigma.ravel())
    return area_pdf, mass_pdf


def pdfs_fluctuations(sigma, delta_edges, s_edges):
    """Return area/mass PDFs of delta and s using the area-mean Sigma."""
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("surface density must be a finite, positive 2D array")
    ratio = sigma.ravel() / sigma.mean()
    delta = ratio - 1.0
    log_fluctuation = np.log(ratio)
    weights = sigma.ravel()
    return {
        "pdf_delta_area": normalized_pdf(delta, delta_edges),
        "pdf_delta_mass": normalized_pdf(delta, delta_edges, weights=weights),
        "pdf_s_area": normalized_pdf(log_fluctuation, s_edges),
        "pdf_s_mass": normalized_pdf(
            log_fluctuation, s_edges, weights=weights
        ),
    }


def window_1d(size, kind="none", tukey_alpha=0.25):
    """Return a one-dimensional none, Hann, or Tukey window."""
    if size < 2:
        raise ValueError("window size must be at least two")
    if kind == "none":
        return np.ones(size)
    if kind == "hann":
        return np.hanning(size)
    if kind != "tukey":
        raise ValueError("window must be one of: none, hann, tukey")
    alpha = float(tukey_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("tukey_alpha must lie between zero and one")
    if alpha == 0.0:
        return np.ones(size)
    if alpha == 1.0:
        return np.hanning(size)
    position = np.arange(size, dtype=float) / (size - 1)
    result = np.ones(size)
    left = position < alpha / 2.0
    right = position >= 1.0 - alpha / 2.0
    result[left] = 0.5 * (
        1.0 + np.cos(np.pi * (2.0 * position[left] / alpha - 1.0))
    )
    result[right] = 0.5 * (
        1.0 + np.cos(np.pi * (2.0 * position[right] / alpha - 2.0 / alpha + 1.0))
    )
    return result


def centered_subregion(values, x_centers, y_centers, size=None):
    """Extract a centered square and return it with its coordinate arrays."""
    values = np.asarray(values)
    x_centers = np.asarray(x_centers)
    y_centers = np.asarray(y_centers)
    if size is None:
        return values, x_centers, y_centers
    dx = float(np.median(np.diff(x_centers)))
    dy = float(np.median(np.diff(y_centers)))
    nx = int(np.rint(float(size) / dx))
    ny = int(np.rint(float(size) / dy))
    if nx < 2 or ny < 2:
        raise ValueError("subregion size must span at least two cells")
    if nx > x_centers.size or ny > y_centers.size:
        raise ValueError("subregion size cannot exceed the projection map")
    ix0 = (x_centers.size - nx) // 2
    iy0 = (y_centers.size - ny) // 2
    return (
        values[iy0:iy0 + ny, ix0:ix0 + nx],
        x_centers[ix0:ix0 + nx],
        y_centers[iy0:iy0 + ny],
    )


def _padded_shape(shape, pad_factor):
    factor = float(pad_factor)
    if factor < 1.0:
        raise ValueError("pad_factor must be at least one")
    return tuple(max(old, int(np.ceil(old * factor))) for old in shape)


def power_spectral_density_2d(
    fluctuation,
    dx,
    dy,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
):
    """Return the normalized 2D periodogram and unsheared FFT coordinates.

    The two-dimensional normalization is
    ``P_2D = |dx dy FFT(w f)|^2 / (dx dy sum(w^2))``. Thus P has units
    of area for a dimensionless fluctuation.
    """
    fluctuation = np.asarray(fluctuation, dtype=float)
    if fluctuation.ndim != 2 or not np.all(np.isfinite(fluctuation)):
        raise ValueError("fluctuation must be a finite 2D array")
    ny, nx = fluctuation.shape
    wx = window_1d(nx, window, tukey_alpha)
    wy = window_1d(ny, window, tukey_alpha)
    weights = wy[:, None] * wx[None, :]
    centered = fluctuation - np.mean(fluctuation)
    weighted = centered * weights
    padded_shape = _padded_shape(weighted.shape, pad_factor)
    transformed = np.fft.fft2(weighted, s=padded_shape)
    effective_area = float(dx) * float(dy) * np.sum(weights**2)
    power_2d = np.abs(float(dx) * float(dy) * transformed) ** 2 / effective_area

    ky = 2.0 * np.pi * np.fft.fftfreq(padded_shape[0], d=float(dy))
    kx0 = 2.0 * np.pi * np.fft.fftfreq(padded_shape[1], d=float(dx))
    return power_2d, kx0, ky


def _physical_wavevector_geometry(power_2d, kx0, ky, shear):
    """Validate a 2D PSD and return physical k and double-angle factors."""
    power_2d = np.asarray(power_2d, dtype=float)
    kx0 = np.asarray(kx0, dtype=float)
    ky = np.asarray(ky, dtype=float)
    if power_2d.shape != (ky.size, kx0.size):
        raise ValueError("power_2d shape must match ky and kx0")
    if not np.all(np.isfinite(power_2d)) or np.any(power_2d < 0.0):
        raise ValueError("power_2d must be finite and nonnegative")
    kx_physical = kx0[None, :] + float(shear) * ky[:, None]
    ky_grid = np.broadcast_to(ky[:, None], power_2d.shape)
    k_squared = kx_physical**2 + ky_grid**2
    kmag = np.sqrt(k_squared)
    cosine_2phi = np.divide(
        kx_physical**2 - ky_grid**2,
        k_squared,
        out=np.zeros_like(k_squared),
        where=k_squared > 0.0,
    )
    sine_2phi = np.divide(
        2.0 * kx_physical * ky_grid,
        k_squared,
        out=np.zeros_like(k_squared),
        where=k_squared > 0.0,
    )
    return power_2d, kmag, cosine_2phi, sine_2phi


def annular_power_statistics_2d(power_2d, kx0, ky, shear, k_edges):
    """Return annular mean power, mode count, and complex quadrupole Q2.

    The quadrupole in each annulus is the power-weighted angular moment
    ``sum(P exp(2 i phi)) / sum(P)`` evaluated with the instantaneous physical
    shearing-wave coordinates.
    """
    power_2d, kmag, cosine_2phi, sine_2phi = _physical_wavevector_geometry(
        power_2d, kx0, ky, shear
    )

    edges = np.asarray(k_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("k_edges must be a strictly increasing 1D array")
    indices = np.digitize(kmag.ravel(), edges) - 1
    valid = (
        (indices >= 0)
        & (indices < edges.size - 1)
        & (kmag.ravel() > 0.0)
    )
    count = np.bincount(indices[valid], minlength=edges.size - 1)
    total = np.bincount(
        indices[valid], weights=power_2d.ravel()[valid], minlength=edges.size - 1
    )
    radial = np.full(edges.size - 1, np.nan)
    np.divide(total, count, out=radial, where=count > 0)
    quadrupole_real_total = np.bincount(
        indices[valid],
        weights=(power_2d * cosine_2phi).ravel()[valid],
        minlength=edges.size - 1,
    )
    quadrupole_imaginary_total = np.bincount(
        indices[valid],
        weights=(power_2d * sine_2phi).ravel()[valid],
        minlength=edges.size - 1,
    )
    q2_real = np.full(edges.size - 1, np.nan)
    q2_imaginary = np.full(edges.size - 1, np.nan)
    np.divide(quadrupole_real_total, total, out=q2_real, where=total > 0.0)
    np.divide(
        quadrupole_imaginary_total,
        total,
        out=q2_imaginary,
        where=total > 0.0,
    )
    return radial, count, q2_real + 1j * q2_imaginary


def annular_average_power_2d(power_2d, kx0, ky, shear, k_edges):
    """Annularly average a 2D PSD using physical shearing-wave wavenumbers."""
    radial, count, _ = annular_power_statistics_2d(
        power_2d, kx0, ky, shear, k_edges
    )
    return radial, count


def band_power_quadrupole_2d(
    power_2d, kx0, ky, shear, wavelength_bounds
):
    """Return the power-weighted Q2 over a physical wavelength band."""
    lower, upper = (float(value) for value in wavelength_bounds)
    if not np.isfinite(lower + upper) or not 0.0 < lower < upper:
        raise ValueError(
            "wavelength bounds must be finite, positive, and increasing"
        )
    power_2d, kmag, cosine_2phi, sine_2phi = _physical_wavevector_geometry(
        power_2d, kx0, ky, shear
    )
    wavelength = np.divide(
        2.0 * np.pi,
        kmag,
        out=np.full_like(kmag, np.inf),
        where=kmag > 0.0,
    )
    use = (wavelength > lower) & (wavelength < upper) & (power_2d > 0.0)
    total = float(np.sum(power_2d[use]))
    if not np.isfinite(total) or total <= 0.0:
        return complex(np.nan, np.nan), 0
    real = float(np.sum(power_2d[use] * cosine_2phi[use]) / total)
    imaginary = float(np.sum(power_2d[use] * sine_2phi[use]) / total)
    return complex(real, imaginary), int(np.count_nonzero(use))


def angle_averaged_power(
    fluctuation,
    dx,
    dy,
    shear,
    k_edges,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
):
    """Calculate an annular PSD using physical shearing-wave wavenumbers."""
    power_2d, kx0, ky = power_spectral_density_2d(
        fluctuation,
        dx,
        dy,
        window=window,
        tukey_alpha=tukey_alpha,
        pad_factor=pad_factor,
    )
    return annular_average_power_2d(power_2d, kx0, ky, shear, k_edges)


def default_k_edges(x_centers, y_centers, bins=DEFAULT_K_BINS):
    """Return logarithmic bins from the region scale through grid Nyquist."""
    dx = float(np.median(np.diff(x_centers)))
    dy = float(np.median(np.diff(y_centers)))
    lx = x_centers.size * dx
    ly = y_centers.size * dy
    kmin = 2.0 * np.pi / min(lx, ly)
    kmax = np.pi / max(dx, dy)
    if kmax <= kmin:
        raise ValueError("map is too small to define power-spectrum bins")
    return logarithmic_bin_edges(kmin, kmax, bins)


def frame_statistics(
    frame,
    qshear,
    omega,
    pdf_edges,
    delta_pdf_edges,
    s_pdf_edges,
    k_edges=None,
    k_bins=DEFAULT_K_BINS,
    subregion_size=None,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
):
    """Calculate all requested one- and two-point statistics for one frame."""
    if not np.isclose(frame["theta"], 0.0):
        raise ValueError("surface-density statistics currently require theta0")
    sigma = (
        np.asarray(frame["fields"]["nH"], dtype=float)
        * star_particle_units()["mass_msun"]
    )
    pdf_mean_sigma = float(np.mean(sigma))
    area_pdf, mass_pdf = pdfs_log10_sigma(sigma, pdf_edges)
    fluctuation_pdfs = pdfs_fluctuations(
        sigma, delta_pdf_edges, s_pdf_edges
    )

    x = np.asarray(frame["x_centers"], dtype=float)
    y = np.asarray(frame["y_centers"], dtype=float)
    lx = float(frame["x_edges"][-1] - frame["x_edges"][0])
    ly = float(frame["y_edges"][-1] - frame["y_edges"][0])
    remap_time, shear = residual_shear(
        frame["time"], qshear, omega, lx, ly
    )
    remapped = shear_remap_periodic(sigma, x, frame["y_spacing"], shear)
    local, local_x, local_y = centered_subregion(
        remapped, x, y, subregion_size
    )
    spectrum_mean_sigma = float(np.mean(local))
    if not np.isfinite(spectrum_mean_sigma) or spectrum_mean_sigma <= 0.0:
        raise ValueError(
            f"{frame.get('path', '')}: selected region has a non-positive "
            "mean surface density"
        )
    delta = local / spectrum_mean_sigma - 1.0
    s_original = np.log(sigma / spectrum_mean_sigma)
    s_remapped = shear_remap_periodic(
        s_original, x, frame["y_spacing"], shear
    )
    log_fluctuation, _, _ = centered_subregion(
        s_remapped, x, y, subregion_size
    )
    if k_edges is None:
        k_edges = default_k_edges(local_x, local_y, bins=k_bins)
    kwargs = {
        "dx": frame["x_spacing"],
        "dy": frame["y_spacing"],
        "shear": shear,
        "k_edges": k_edges,
        "window": window,
        "tukey_alpha": tukey_alpha,
        "pad_factor": pad_factor,
    }
    power_delta, count = angle_averaged_power(delta, **kwargs)
    power_s, count_s = angle_averaged_power(log_fluctuation, **kwargs)
    if not np.array_equal(count, count_s):
        raise RuntimeError("inconsistent Fourier mode counts")
    result = {
        "pdf_log10_sigma_area": area_pdf,
        "pdf_log10_sigma_mass": mass_pdf,
        "power_delta": power_delta,
        "power_s": power_s,
        "mode_count": count,
        "mean_sigma_pdf": pdf_mean_sigma,
        "mean_sigma_spectrum": spectrum_mean_sigma,
        "remap_time": remap_time,
        "shear": shear,
        "k_edges": np.asarray(k_edges),
        "local_shape": np.asarray(local.shape),
        "local_size": np.asarray([
            local_x.size * frame["x_spacing"],
            local_y.size * frame["y_spacing"],
        ]),
    }
    result.update(fluctuation_pdfs)
    return result


def analyze_model(
    model,
    pdf_edges,
    delta_pdf_edges,
    s_pdf_edges,
    k_bins=DEFAULT_K_BINS,
    subregion_size=None,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
    time_tolerance=0.01,
    output_name=STATISTICS_NAME,
):
    """Analyze every deduplicated theta0 projection in one late run."""
    model = Path(model)
    parameter_path, qshear, omega = read_shear_parameters(model)
    series = projection_series(model, "theta0", time_tolerance=time_tolerance)
    output_dir = model / "proj2d" / "theta0"
    rows = []
    k_edges = None
    for index, (stored_time, path) in enumerate(series):
        frame = read_proj2d(path, fields="nH")
        if not np.isclose(frame["time"], stored_time):
            raise ValueError(f"metadata time changed while reading {path}")
        row = frame_statistics(
            frame,
            qshear,
            omega,
            pdf_edges,
            delta_pdf_edges,
            s_pdf_edges,
            k_edges=k_edges,
            k_bins=k_bins,
            subregion_size=subregion_size,
            window=window,
            tukey_alpha=tukey_alpha,
            pad_factor=pad_factor,
        )
        if k_edges is None:
            k_edges = row["k_edges"]
        row["time"] = frame["time"]
        row["num"] = frame["num"] or ""
        rows.append(row)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(series):
            print(f"{model.name}: {index + 1}/{len(series)}")

    time = np.asarray([row["time"] for row in rows])
    time_myr = time * star_particle_units()["time_myr"]
    data = {
        "model": np.asarray(model.name),
        "projection_id": np.asarray("theta0"),
        "field": np.asarray("nH"),
        "surface_density_unit": np.asarray("Msun pc^-2"),
        "power_unit": np.asarray("pc^2"),
        "delta_definition": np.asarray("Sigma/<Sigma>-1"),
        "pdf_mean_definition": np.asarray("full-domain area mean"),
        "power_mean_definition": np.asarray(
            "selected-region area mean after shear remap"
        ),
        "s_definition": np.asarray("ln(Sigma/<Sigma>)"),
        "athinput": np.asarray(str(parameter_path)),
        "qshear": np.asarray(qshear),
        "omega_kms_per_pc": np.asarray(omega),
        "time": time,
        "time_myr": time_myr,
        "num": np.asarray([row["num"] for row in rows]),
        "mean_sigma_pdf": np.asarray(
            [row["mean_sigma_pdf"] for row in rows]
        ),
        "mean_sigma_spectrum": np.asarray(
            [row["mean_sigma_spectrum"] for row in rows]
        ),
        "remap_time": np.asarray([row["remap_time"] for row in rows]),
        "shear": np.asarray([row["shear"] for row in rows]),
        "pdf_log10_sigma_edges": np.asarray(pdf_edges),
        "pdf_log10_sigma_centers": 0.5 * (
            np.asarray(pdf_edges[:-1]) + np.asarray(pdf_edges[1:])
        ),
        "pdf_log10_sigma_area": np.asarray([row["pdf_log10_sigma_area"] for row in rows]),
        "pdf_log10_sigma_mass": np.asarray([row["pdf_log10_sigma_mass"] for row in rows]),
        "k_edges": k_edges,
        "pdf_delta_edges": np.asarray(delta_pdf_edges),
        "pdf_delta_centers": 0.5 * (
            np.asarray(delta_pdf_edges[:-1]) + np.asarray(delta_pdf_edges[1:])
        ),
        "pdf_delta_area": np.asarray(
            [row["pdf_delta_area"] for row in rows]
        ),
        "pdf_delta_mass": np.asarray(
            [row["pdf_delta_mass"] for row in rows]
        ),
        "pdf_s_edges": np.asarray(s_pdf_edges),
        "pdf_s_centers": 0.5 * (
            np.asarray(s_pdf_edges[:-1]) + np.asarray(s_pdf_edges[1:])
        ),
        "pdf_s_area": np.asarray([row["pdf_s_area"] for row in rows]),
        "pdf_s_mass": np.asarray([row["pdf_s_mass"] for row in rows]),
        "k_centers": np.sqrt(k_edges[:-1] * k_edges[1:]),
        "power_delta": np.asarray([row["power_delta"] for row in rows]),
        "power_s": np.asarray([row["power_s"] for row in rows]),
        "mode_count": np.asarray([row["mode_count"] for row in rows]),
        "subregion_size_requested_pc": np.asarray(
            np.nan if subregion_size is None else subregion_size
        ),
        "local_size_pc": rows[0]["local_size"],
        "local_shape": rows[0]["local_shape"],
        "window": np.asarray(window),
        "tukey_alpha": np.asarray(tukey_alpha),
        "pad_factor": np.asarray(pad_factor),
    }
    output = output_dir / output_name
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(output)
    print(f"wrote {output}")
    return data


def _percentiles(series):
    return np.nanpercentile(series, [5.0, 50.0, 95.0], axis=0)


def plot_pdf_summary(results, output):
    """Plot median and 5--95 percentile PDFs of Sigma, delta, and s."""
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharey=False)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.82, len(results)))
    columns = (
        (
            "pdf_log10_sigma_centers", "pdf_log10_sigma_area", "pdf_log10_sigma_mass",
            r"$\log_{10}(\Sigma_{\rm gas}/[M_\odot\,{\rm pc}^{-2}])$",
        ),
        (
            "pdf_delta_centers", "pdf_delta_area", "pdf_delta_mass",
            r"$\delta=\Sigma/\langle\Sigma\rangle-1$",
        ),
        (
            "pdf_s_centers", "pdf_s_area", "pdf_s_mass",
            r"$s=\ln(\Sigma/\langle\Sigma\rangle)$",
        ),
    )
    for result, color in zip(results, colors):
        label = f"L={model_size(result['model'].item()):g} pc"
        for column, (xkey, area_key, mass_key, _) in enumerate(columns):
            x = result[xkey]
            for row, key in enumerate((area_key, mass_key)):
                axis = axes[row, column]
                low, median, high = _percentiles(result[key])
                positive = high > 0.0
                axis.fill_between(
                    x[positive], low[positive], high[positive],
                    color=color, alpha=0.2, linewidth=0,
                )
                axis.plot(
                    x[positive], median[positive], color=color, label=label
                )
    for column, (_, _, _, xlabel) in enumerate(columns):
        axes[1, column].set_xlabel(xlabel)
    for axis in axes.ravel():
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Area-weighted PDF")
    axes[1, 0].set_ylabel("Mass-weighted PDF")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Gas surface-density PDFs: median and 5--95% over time", y=0.995
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output}")


def plot_power_summary(results, output):
    """Plot median and 5--95 percentile delta/s spectra for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharex=False, sharey=False)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.82, len(results)))
    for result, color in zip(results, colors):
        k = result["k_centers"]
        label = f"L={model_size(result['model'].item()):g} pc"
        for axis, key in zip(axes, ("power_delta", "power_s")):
            low, median, high = _percentiles(result[key])
            positive = np.isfinite(median) & (median > 0.0)
            axis.fill_between(
                k[positive], low[positive], high[positive],
                color=color, alpha=0.2, linewidth=0,
            )
            axis.plot(k[positive], median[positive], color=color, label=label)
    for axis, title in zip(
        axes,
        (r"$\delta=\Sigma/\langle\Sigma\rangle-1$",
         r"$s=\ln(\Sigma/\langle\Sigma\rangle)$"),
    ):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$k\ [{\rm pc}^{-1}]$")
        axis.set_ylabel(r"$P(k)\ [{\rm pc}^{2}]$")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Shear-corrected spectra: median and 5--95% over time")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output}")


def analyze_suite(
    suite=DEFAULT_SUITE,
    pdf_range=DEFAULT_PDF_RANGE,
    delta_range=DEFAULT_DELTA_RANGE,
    s_range=DEFAULT_S_RANGE,
    pdf_bins=DEFAULT_PDF_BINS,
    k_bins=DEFAULT_K_BINS,
    subregion_size=None,
    window="none",
    tukey_alpha=0.25,
    pad_factor=1.0,
    time_tolerance=0.01,
    output_name=STATISTICS_NAME,
):
    """Analyze the late theta0 series and create combined suite figures."""
    suite = Path(suite)
    models = discover_late_models(suite, "theta0")
    pdf_edges = np.linspace(pdf_range[0], pdf_range[1], int(pdf_bins) + 1)
    delta_pdf_edges = np.linspace(
        delta_range[0], delta_range[1], int(pdf_bins) + 1
    )
    s_pdf_edges = np.linspace(
        s_range[0], s_range[1], int(pdf_bins) + 1
    )
    results = [
        analyze_model(
            model,
            pdf_edges,
            delta_pdf_edges,
            s_pdf_edges,
            k_bins=k_bins,
            subregion_size=subregion_size,
            window=window,
            tukey_alpha=tukey_alpha,
            pad_factor=pad_factor,
            time_tolerance=time_tolerance,
            output_name=output_name,
        )
        for model in models
    ]
    plot_pdf_summary(results, suite / "surface_density_pdf_summary.png")
    plot_power_summary(results, suite / "surface_density_power_summary.png")
    return results


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate theta0 surface-density PDFs and shear-corrected "
            "angle-averaged power spectra for all late runs."
        )
    )
    parser.add_argument("suite", nargs="?", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--pdf-min", type=float, default=DEFAULT_PDF_RANGE[0])
    parser.add_argument("--pdf-max", type=float, default=DEFAULT_PDF_RANGE[1])
    parser.add_argument("--pdf-bins", type=int, default=DEFAULT_PDF_BINS)
    parser.add_argument("--k-bins", type=int, default=DEFAULT_K_BINS)
    parser.add_argument("--delta-min", type=float, default=DEFAULT_DELTA_RANGE[0])
    parser.add_argument("--delta-max", type=float, default=DEFAULT_DELTA_RANGE[1])
    parser.add_argument("--s-min", type=float, default=DEFAULT_S_RANGE[0])
    parser.add_argument("--s-max", type=float, default=DEFAULT_S_RANGE[1])
    parser.add_argument(
        "--subregion-size", type=float, default=None, metavar="PC",
        help="centered square used for spectra; PDFs remain full-domain",
    )
    parser.add_argument(
        "--window", choices=("none", "hann", "tukey"), default="none",
        help="apodization applied before the spectrum (default: none)",
    )
    parser.add_argument("--tukey-alpha", type=float, default=0.25)
    parser.add_argument(
        "--pad-factor", type=float, default=1.0,
        help="zero-padded FFT size divided by selected-region size",
    )
    parser.add_argument("--time-tolerance", type=float, default=0.01)
    parser.add_argument("--output-name", default=STATISTICS_NAME)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.pdf_min < args.pdf_max:
        parser.error("--pdf-min must be smaller than --pdf-max")
    if not args.delta_min < args.delta_max:
        parser.error("--delta-min must be smaller than --delta-max")
    if not args.s_min < args.s_max:
        parser.error("--s-min must be smaller than --s-max")
    if args.pdf_bins <= 0 or args.k_bins <= 0:
        parser.error("--pdf-bins and --k-bins must be positive")
    if args.subregion_size is not None and args.subregion_size <= 0.0:
        parser.error("--subregion-size must be positive")
    if not 0.0 <= args.tukey_alpha <= 1.0:
        parser.error("--tukey-alpha must be between zero and one")
    if args.pad_factor < 1.0:
        parser.error("--pad-factor must be at least one")
    analyze_suite(
        args.suite,
        pdf_range=(args.pdf_min, args.pdf_max),
        pdf_bins=args.pdf_bins,
        k_bins=args.k_bins,
        delta_range=(args.delta_min, args.delta_max),
        s_range=(args.s_min, args.s_max),
        subregion_size=args.subregion_size,
        window=args.window,
        tukey_alpha=args.tukey_alpha,
        pad_factor=args.pad_factor,
        time_tolerance=args.time_tolerance,
        output_name=args.output_name,
    )


if __name__ == "__main__":
    main()
