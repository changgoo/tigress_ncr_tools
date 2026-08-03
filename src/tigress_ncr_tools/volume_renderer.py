"""Orthographic temperature-volume rendering for slice-story movies."""

from dataclasses import dataclass

import numpy as np

from pathena.plot_helpers import FIELD_META
from pathena.units import DEFAULT_MUH, tigress_units


@dataclass(frozen=True)
class VolumeImage:
    """A ray-composited image and its physical screen-plane bounds."""

    rgba: np.ndarray
    extent: tuple
    camera_fraction: float


def camera_basis(camera_fraction):
    """Return the registered view-normal and screen-up vectors.

    The path rotates about the x axis. At zero it looks down from +z with
    +y upward; at one it looks from -y with +z upward, matching an XZ slice.
    """
    fraction = float(camera_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("camera_fraction must lie in [0, 1]")
    angle = 0.5 * np.pi * fraction
    normal = np.array([0.0, np.sin(angle), -np.cos(angle)])
    north = np.array([0.0, np.cos(angle), np.sin(angle)])
    return normal, north


def volume_input_fields(field_names):
    """Select the smallest raw-field set needed for temperature and opacity."""
    names = set(field_names)
    if "density" not in names:
        raise KeyError("volume is missing required field 'density'")
    if "temperature" in names:
        return ("density", "temperature")
    required = ["pressure"]
    aliases = []
    for public, legacy in (("xH2", "specific_scalar_3"), ("xe", "specific_scalar_4")):
        if public in names:
            required.append(public)
        elif legacy in names:
            required.append(legacy)
        elif public == "xe":
            # The slice-field convention permits neutral-electron fallback.
            continue
        else:
            aliases.append(public)
    missing = [name for name in required if name not in names]
    missing.extend(aliases)
    if missing:
        raise KeyError(
            "volume cannot derive temperature; missing " + ", ".join(missing)
        )
    return tuple(["density", *required])


def volume_temperature(volume, muH=DEFAULT_MUH):
    """Return temperature in Kelvin from a reader result."""
    fields = volume["fields"]
    if "temperature" in fields:
        return np.asarray(fields["temperature"], dtype=float)
    density = np.asarray(fields["density"], dtype=float)
    pressure = np.asarray(fields["pressure"], dtype=float)
    xh2 = np.asarray(
        fields.get("xH2", fields.get("specific_scalar_3")), dtype=float
    )
    xe_raw = fields.get("xe", fields.get("specific_scalar_4"))
    xe = np.zeros_like(density) if xe_raw is None else np.asarray(xe_raw, dtype=float)
    denominator = density * (1.1 + xe - xh2)
    factor = tigress_units(muH)["temperature_per_p_over_rho"]
    return np.divide(
        pressure * factor,
        denominator,
        out=np.full_like(pressure, np.nan, dtype=float),
        where=denominator != 0.0,
    )


def _sample_plane(data, z_index, y_index):
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as error:
        raise RuntimeError(
            "intermediate volume camera angles require scipy; install the "
            "'movie3d' optional dependency"
        ) from error
    return map_coordinates(
        data,
        (z_index, y_index),
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def _camera_grid(volume, camera_fraction, stride):
    left = np.asarray(volume["left_edge"], dtype=float)
    right = np.asarray(volume["right_edge"], dtype=float)
    spacing = np.asarray(volume["spacing"], dtype=float) * stride
    center = 0.5 * (left + right)
    lengths = right - left
    angle = 0.5 * np.pi * camera_fraction
    cosine, sine = np.cos(angle), np.sin(angle)
    vertical_span = abs(cosine) * lengths[1] + abs(sine) * lengths[2]
    depth_span = abs(sine) * lengths[1] + abs(cosine) * lengths[2]
    sample_spacing = max(spacing[1], spacing[2])
    vertical_count = max(1, int(np.ceil(vertical_span / sample_spacing)))
    depth_count = max(1, int(np.ceil(depth_span / sample_spacing)))
    vertical = (
        np.arange(vertical_count, dtype=float) + 0.5
    ) * vertical_span / vertical_count - 0.5 * vertical_span
    depth = (
        np.arange(depth_count, dtype=float) + 0.5
    ) * depth_span / depth_count - 0.5 * depth_span
    w, v = np.meshgrid(depth, vertical, indexing="ij")
    y = center[1] + v * cosine - w * sine
    z = center[2] + v * sine + w * cosine

    original_spacing = np.asarray(volume["spacing"], dtype=float)
    y_first = left[1] + 0.5 * original_spacing[1]
    z_first = left[2] + 0.5 * original_spacing[2]
    y_index = (y - y_first) / (original_spacing[1] * stride)
    z_index = (z - z_first) / (original_spacing[2] * stride)
    vertical_center = center[1] * cosine + center[2] * sine
    extent = (
        float(left[0]), float(right[0]),
        float(vertical_center - 0.5 * vertical_span),
        float(vertical_center + 0.5 * vertical_span),
    )
    if camera_fraction == 1.0:
        extent = (
            float(left[0]), float(right[0]),
            float(left[2]), float(right[2]),
        )
    return z_index, y_index, extent


def _colormap(name):
    import matplotlib

    registry = getattr(matplotlib, "colormaps", None)
    if registry is not None:
        return registry.get_cmap(name)
    from matplotlib import cm
    return cm.get_cmap(name)


def render_temperature_volume(
    volume,
    camera_fraction,
    *,
    stride=1,
    opacity_scale=0.08,
    muH=DEFAULT_MUH,
):
    """Ray-composite temperature color with density-dependent opacity.

    The full reader arrays have ``(z, y, x)`` storage. ``stride`` controls
    resampling for rendering only; it never changes what the reader assembled.
    """
    fraction = float(camera_fraction)
    camera_basis(fraction)
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("volume stride must be a positive integer")
    if opacity_scale <= 0.0:
        raise ValueError("volume opacity scale must be positive")

    density = np.asarray(volume["fields"]["density"], dtype=float)
    temperature = volume_temperature(volume, muH=muH)
    if density.shape != temperature.shape or density.ndim != 3:
        raise ValueError("density and temperature must be matching 3D arrays")
    density = density[::stride, ::stride, ::stride]
    temperature = temperature[::stride, ::stride, ::stride]
    z_index, y_index, extent = _camera_grid(volume, fraction, stride)

    style_t = FIELD_META["T"]
    style_n = FIELD_META["nH"]
    log_t_min, log_t_max = np.log10([style_t["vmin"], style_t["vmax"]])
    log_n_min, log_n_max = np.log10([style_n["vmin"], style_n["vmax"]])
    cmap = _colormap(style_t["cmap"])
    image = np.zeros((z_index.shape[1], density.shape[2], 4), dtype=np.uint8)

    for x_index in range(density.shape[2]):
        sampled_t = _sample_plane(temperature[:, :, x_index], z_index, y_index)
        sampled_n = _sample_plane(density[:, :, x_index], z_index, y_index)
        valid = np.isfinite(sampled_t) & np.isfinite(sampled_n)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_level = (np.log10(sampled_t) - log_t_min) / (log_t_max - log_t_min)
            n_level = (np.log10(sampled_n) - log_n_min) / (log_n_max - log_n_min)
        t_level = np.nan_to_num(t_level, nan=0.0, posinf=1.0, neginf=0.0)
        n_level = np.nan_to_num(n_level, nan=0.0, posinf=1.0, neginf=0.0)
        colors = cmap(np.clip(t_level, 0.0, 1.0))[..., :3]
        optical_depth = opacity_scale * np.clip(n_level, 0.0, 1.0) ** 1.5
        alpha = np.where(valid, 1.0 - np.exp(-optical_depth), 0.0)

        rgb = np.zeros((z_index.shape[1], 3), dtype=float)
        transmission = np.ones(z_index.shape[1], dtype=float)
        for depth_index in range(z_index.shape[0] - 1, -1, -1):
            contribution = transmission * alpha[depth_index]
            rgb += contribution[:, None] * colors[depth_index]
            transmission *= 1.0 - alpha[depth_index]
        image[:, x_index, :3] = np.rint(rgb.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        image[:, x_index, 3] = 255

    return VolumeImage(image, extent, fraction)
