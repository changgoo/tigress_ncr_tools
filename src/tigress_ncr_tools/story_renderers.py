"""Fixed-layout renderers and pixel compositors for slice movie frames."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.image as mpl_image
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from pathena.plot_helpers import FIELD_META
from pathena.slice_fields import derive_plane_fields
from pathena.units import DEFAULT_MUH, star_particle_units


DENSITY_FIELDS = ("nH", "nH2", "nHI", "nHII")
PLANE_COORDINATES = {
    "x1": ("x2", "x3"),
    "x2": ("x1", "x3"),
    "x3": ("x1", "x2"),
}
PARTICLE_COORDINATES = {
    "x1": ("x2", "x3"),
    "x2": ("x1", "x3"),
    "x3": ("x1", "x2"),
}


@dataclass(frozen=True)
class CanvasSettings:
    """Pixel geometry and common visual settings for all movie frames."""

    width: int = 1920
    height: int = 1080
    dpi: int = 100
    background: str = "black"
    muH: float = DEFAULT_MUH
    particle_age_max_myr: float = 40.0
    particle_size_norm: float = 4.0

    def validate(self):
        if self.width <= 0 or self.height <= 0 or self.dpi <= 0:
            raise ValueError("canvas width, height, and dpi must be positive")
        if self.particle_age_max_myr <= 0.0:
            raise ValueError("particle_age_max_myr must be positive")
        if self.particle_size_norm <= 0.0:
            raise ValueError("particle_size_norm must be positive")


def field_style(field):
    """Return an independent plotting-style dictionary for a physical field."""
    if field not in FIELD_META:
        raise KeyError(f"no plotting metadata for field {field!r}")
    return dict(FIELD_META[field])


def field_norm(field):
    """Return the fixed normalization assigned to ``field``."""
    style = field_style(field)
    if style.get("log", False):
        return LogNorm(style["vmin"], style["vmax"])
    return Normalize(style["vmin"], style["vmax"])


def blend_rgba(first, second, fraction):
    """Linearly blend equal-sized RGBA canvases and return ``uint8`` pixels."""
    a = np.asarray(first)
    b = np.asarray(second)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 4:
        raise ValueError("RGBA canvases must have matching (height, width, 4) shapes")
    alpha = float(fraction)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend fraction must lie in [0, 1]")
    if alpha == 0.0:
        return a.astype(np.uint8, copy=True)
    if alpha == 1.0:
        return b.astype(np.uint8, copy=True)
    mixed = (1.0 - alpha) * a.astype(float) + alpha * b.astype(float)
    return np.rint(mixed).clip(0.0, 255.0).astype(np.uint8)


def _text_effects():
    return [path_effects.withStroke(linewidth=2.2, foreground="black")]


def _nice_scale(span):
    target = 0.2 * abs(float(span))
    if not np.isfinite(target) or target <= 0.0:
        return None
    exponent = np.floor(np.log10(target))
    scaled = target / 10.0**exponent
    leading = max(value for value in (1.0, 2.0, 5.0) if value <= scaled)
    return leading * 10.0**exponent


def _display_axis_name(name):
    return {"x1": "x", "x2": "y", "x3": "z"}.get(name, name)


def _decorate_coordinates(ax, plane_data):
    x_edges = plane_data["x_edges"]
    y_edges = plane_data["y_edges"]
    xmin, xmax = float(x_edges[0]), float(x_edges[-1])
    ymin, ymax = float(y_edges[0]), float(y_edges[-1])
    xspan, yspan = xmax - xmin, ymax - ymin
    effects = _text_effects()
    color = "white"

    x0 = xmin + 0.055 * xspan
    y0 = ymin + 0.065 * yspan
    arrow = 0.075 * min(abs(xspan), abs(yspan))
    ax.annotate(
        "", xy=(x0 + arrow, y0), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
    )
    ax.annotate(
        "", xy=(x0, y0 + arrow), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
    )
    axis_names = PLANE_COORDINATES[plane_data["normal"]]
    ax.text(
        x0 + 1.2 * arrow, y0, _display_axis_name(axis_names[0]),
        color=color, fontsize=11, ha="left", va="center", path_effects=effects,
    )
    ax.text(
        x0, y0 + 1.2 * arrow, _display_axis_name(axis_names[1]),
        color=color, fontsize=11, ha="center", va="bottom", path_effects=effects,
    )

    scale = _nice_scale(xspan)
    if scale is not None:
        x1 = xmax - 0.055 * xspan
        xs = x1 - scale
        ax.plot([xs, x1], [y0, y0], color=color, lw=2.2, solid_capstyle="butt")
        ax.text(
            0.5 * (xs + x1), y0 + 0.025 * yspan, f"{scale:g} pc",
            color=color, fontsize=10, ha="center", va="bottom",
            path_effects=effects,
        )


def _particle_arrays(particles):
    if particles is None:
        return None
    if isinstance(particles, dict) and "particles" in particles:
        particles = particles["particles"]
    if hasattr(particles, "columns"):
        names = set(particles.columns)
        getter = lambda name: np.asarray(particles[name], dtype=float)
    elif isinstance(particles, dict):
        names = set(particles)
        getter = lambda name: np.asarray(particles[name], dtype=float)
    else:
        raise TypeError("particles must be a DataFrame, mapping, or starpar frame")
    required = {"x1", "x2", "x3"}
    if not required.issubset(names):
        missing = ", ".join(sorted(required - names))
        raise KeyError(f"particle data is missing coordinates: {missing}")
    count = len(getter("x1"))
    result = {name: getter(name) for name in required}
    for name, default in (("mass", 0.0), ("age", 0.0), ("id", 0.0)):
        result[name] = getter(name) if name in names else np.full(count, default)
    if any(len(values) != count for values in result.values()):
        raise ValueError("particle columns must all have the same length")
    return result


def draw_particles(ax, particles, plane, alpha=1.0, *, muH=DEFAULT_MUH,
                   age_max_myr=40.0, size_norm=4.0):
    """Draw star particles directly in one coordinate-plane view."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("particle alpha must lie in [0, 1]")
    arrays = _particle_arrays(particles)
    if arrays is None or alpha == 0.0:
        return []
    if plane not in PARTICLE_COORDINATES:
        raise ValueError(f"unknown slice plane {plane!r}")

    xname, yname = PARTICLE_COORDINATES[plane]
    x, y = arrays[xname], arrays[yname]
    mass, age, particle_id = arrays["mass"], arrays["age"], arrays["id"]
    units = star_particle_units(muH)
    mass_msun = mass * units["mass_msun"]
    age_myr = age * units["time_myr"]
    cluster = (mass > 0.0) & (age_myr < age_max_myr)
    source = (mass <= 0.0) & (particle_id < 0.0)
    runaway = (mass <= 0.0) & ~source
    artists = []

    if np.any(cluster):
        artists.append(ax.scatter(
            x[cluster], y[cluster],
            s=np.sqrt(np.maximum(mass_msun[cluster], 0.0)) / size_norm,
            c=age_myr[cluster], marker="o", cmap="cool_r",
            vmin=0.0, vmax=age_max_myr, alpha=0.75 * alpha,
            linewidths=0.25, edgecolors="white", zorder=6,
        ))
    if np.any(runaway):
        artists.append(ax.scatter(
            x[runaway], y[runaway], s=10.0 / size_norm,
            marker="o", color="white", alpha=alpha, linewidths=0.0, zorder=6,
        ))
    if np.any(source):
        artists.append(ax.scatter(
            x[source], y[source], s=20.0 / size_norm,
            marker="*", color="#ff4040", alpha=alpha, linewidths=0.0, zorder=7,
        ))
    return artists


def _add_colorbar(fig, image, style):
    color_axis = fig.add_axes([0.32, 0.895, 0.36, 0.027])
    color_axis.set_facecolor((0.0, 0.0, 0.0, 0.65))
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(style["label"], color="white", fontsize=10, labelpad=2)
    colorbar.ax.tick_params(
        axis="x", colors="white", labelsize=8, length=3, pad=1
    )
    colorbar.outline.set_edgecolor("white")
    colorbar.outline.set_linewidth(0.8)
    return colorbar


def _new_canvas(settings):
    figure = Figure(
        figsize=(settings.width / settings.dpi, settings.height / settings.dpi),
        dpi=settings.dpi,
        facecolor=settings.background,
    )
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes([0.045, 0.055, 0.91, 0.875])
    axis.set_facecolor(settings.background)
    axis.set_axis_off()
    return figure, canvas, axis


def _finish_canvas(figure, canvas, slc, settings, title):
    effects = _text_effects()
    figure.text(
        0.025, 0.965, title, color="white", fontsize=18,
        ha="left", va="top", path_effects=effects,
    )
    time_myr = float(slc["time"]) * star_particle_units(settings.muH)["time_myr"]
    figure.text(
        0.975, 0.965, rf"$t={time_myr:.1f}\,\mathrm{{Myr}}$",
        color="white", fontsize=15, ha="right", va="top",
        path_effects=effects,
    )
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()
    figure.clear()
    return rgba


def render_slice_view(slc, plane, field, *, derived=None, particles=None,
                      particle_alpha=0.0, settings=None, title=None):
    """Render one fully annotated scalar slice as an RGBA pixel array."""
    settings = CanvasSettings() if settings is None else settings
    settings.validate()
    if plane not in slc["planes"]:
        raise KeyError(f"slice frame has no plane {plane!r}")
    plane_data = slc["planes"][plane]
    derived = (
        derive_plane_fields(slc, plane, muH=settings.muH)
        if derived is None
        else derived
    )
    if field not in derived:
        available = ", ".join(sorted(derived))
        raise KeyError(f"derived field {field!r} is unavailable; have: {available}")
    style = field_style(field)
    data = np.asarray(derived[field], dtype=float)
    if data.shape != (plane_data["Ny"], plane_data["Nx"]):
        raise ValueError(
            f"field {field!r} shape {data.shape} does not match plane "
            f"{(plane_data['Ny'], plane_data['Nx'])}"
        )
    if style.get("log", False):
        data = np.ma.masked_less_equal(data, 0.0)

    figure, canvas, axis = _new_canvas(settings)
    image = axis.imshow(
        data,
        origin="lower",
        extent=(
            plane_data["x_edges"][0], plane_data["x_edges"][-1],
            plane_data["y_edges"][0], plane_data["y_edges"][-1],
        ),
        cmap=style["cmap"],
        norm=field_norm(field),
        interpolation="nearest",
        aspect="equal",
    )
    draw_particles(
        axis,
        particles,
        plane,
        alpha=particle_alpha,
        muH=settings.muH,
        age_max_myr=settings.particle_age_max_myr,
        size_norm=settings.particle_size_norm,
    )
    _decorate_coordinates(axis, plane_data)
    _add_colorbar(figure, image, style)
    return _finish_canvas(
        figure, canvas, slc, settings, title or style["short"]
    )


def render_volume_view(volume_image, time, *, settings=None, title=None):
    """Place a ray-composited volume in the shared annotated movie canvas."""
    settings = CanvasSettings() if settings is None else settings
    settings.validate()
    pixels = np.asarray(volume_image.rgba)
    if pixels.ndim != 3 or pixels.shape[-1] != 4:
        raise ValueError("volume image must have shape (height, width, 4)")
    if len(volume_image.extent) != 4:
        raise ValueError("volume image extent must contain four bounds")
    fraction = float(volume_image.camera_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("volume camera fraction must lie in [0, 1]")

    figure, canvas, axis = _new_canvas(settings)
    axis.imshow(
        pixels,
        origin="lower",
        extent=volume_image.extent,
        interpolation="bilinear",
        aspect="equal",
    )
    xmin, xmax, ymin, ymax = volume_image.extent
    if fraction == 0.0 or fraction == 1.0:
        _decorate_coordinates(axis, {
            "normal": "x3" if fraction == 0.0 else "x2",
            "x_edges": np.asarray([xmin, xmax]),
            "y_edges": np.asarray([ymin, ymax]),
        })
    else:
        angle = 90.0 * fraction
        axis.text(
            0.025,
            0.025,
            rf"$x$ horizontal; camera turn ${angle:.0f}^\circ$",
            transform=axis.transAxes,
            color="white",
            fontsize=10,
            ha="left",
            va="bottom",
            path_effects=_text_effects(),
        )
    scalar = ScalarMappable(norm=field_norm("T"), cmap=field_style("T")["cmap"])
    _add_colorbar(figure, scalar, field_style("T"))
    if title is None:
        if fraction == 0.0:
            title = "Temperature volume — top-down (XY)"
        elif fraction == 1.0:
            title = "Temperature volume — side-on (XZ)"
        else:
            title = f"Temperature volume — camera turn {90.0 * fraction:.0f}°"
    return _finish_canvas(
        figure, canvas, {"time": float(time)}, settings, title
    )


def _plane_centers(plane_data):
    x = plane_data.get("x_centers")
    y = plane_data.get("y_centers")
    if x is None:
        edges = np.asarray(plane_data["x_edges"], dtype=float)
        x = 0.5 * (edges[:-1] + edges[1:])
    if y is None:
        edges = np.asarray(plane_data["y_edges"], dtype=float)
        y = 0.5 * (edges[:-1] + edges[1:])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _streamline_fields(kind, plane):
    suffixes = {
        "x1": ("2", "3"),
        "x2": ("1", "3"),
        "x3": ("1", "2"),
    }
    if plane not in suffixes:
        raise ValueError(f"unknown slice plane {plane!r}")
    first, second = suffixes[plane]
    if kind == "velocity":
        return "v" + first, "v" + second, "T"
    if kind == "magnetic":
        return "B" + first, "B" + second, "Bmag"
    raise ValueError("streamline kind must be 'velocity' or 'magnetic'")


def render_streamline_view(slc, plane, kind, *, derived=None, settings=None,
                           density=1.35, background_alpha=0.32, title=None):
    """Render velocity or magnetic streamlines on a dim temperature slice."""
    settings = CanvasSettings() if settings is None else settings
    settings.validate()
    if density <= 0.0:
        raise ValueError("streamline density must be positive")
    if not 0.0 <= background_alpha <= 1.0:
        raise ValueError("background_alpha must lie in [0, 1]")
    if plane not in slc["planes"]:
        raise KeyError(f"slice frame has no plane {plane!r}")
    plane_data = slc["planes"][plane]
    derived = (
        derive_plane_fields(slc, plane, muH=settings.muH)
        if derived is None
        else derived
    )
    u_name, v_name, color_name = _streamline_fields(kind, plane)
    needed = ("T", u_name, v_name, color_name)
    missing = [name for name in needed if name not in derived]
    if missing:
        raise KeyError(f"streamline fields are unavailable: {', '.join(missing)}")
    expected = (plane_data["Ny"], plane_data["Nx"])
    arrays = {
        name: np.asarray(derived[name], dtype=float)
        for name in needed
    }
    if any(array.shape != expected for array in arrays.values()):
        raise ValueError(f"streamline arrays must all match plane shape {expected}")

    figure, canvas, axis = _new_canvas(settings)
    temperature = np.ma.masked_less_equal(arrays["T"], 0.0)
    axis.imshow(
        temperature,
        origin="lower",
        extent=(
            plane_data["x_edges"][0], plane_data["x_edges"][-1],
            plane_data["y_edges"][0], plane_data["y_edges"][-1],
        ),
        cmap=field_style("T")["cmap"],
        norm=field_norm("T"),
        interpolation="nearest",
        aspect="equal",
        alpha=background_alpha,
    )
    x, y = _plane_centers(plane_data)
    color_values = np.ma.masked_less_equal(arrays[color_name], 0.0)
    stream = axis.streamplot(
        x,
        y,
        arrays[u_name],
        arrays[v_name],
        color=color_values,
        cmap=field_style(color_name)["cmap"],
        norm=field_norm(color_name),
        density=density,
        linewidth=1.15,
        arrowsize=0.8,
        minlength=0.08,
        zorder=5,
    )
    axis.set_xlim(plane_data["x_edges"][0], plane_data["x_edges"][-1])
    axis.set_ylim(plane_data["y_edges"][0], plane_data["y_edges"][-1])
    axis.set_aspect("equal", adjustable="box")
    _decorate_coordinates(axis, plane_data)
    _add_colorbar(figure, stream.lines, field_style(color_name))
    default_title = (
        "Velocity streamlines colored by temperature"
        if kind == "velocity"
        else "Magnetic streamlines colored by field strength"
    )
    return _finish_canvas(
        figure, canvas, slc, settings, title or default_title
    )


def _normalized_intensity(data, field):
    values = np.asarray(data, dtype=float)
    normalized = field_norm(field)(values)
    return np.ma.filled(normalized, 0.0).clip(0.0, 1.0)


def _get_cmap(name):
    registry = getattr(matplotlib, "colormaps", None)
    if registry is not None:
        return registry.get_cmap(name)
    from matplotlib import cm
    return cm.get_cmap(name)


def radiation_composite_rgba(fuv, lyc, *, fuv_cmap="magma",
                             lyc_cmap="winter"):
    """Map FUV and LyC independently and combine them with screen blending."""
    fuv = np.asarray(fuv, dtype=float)
    lyc = np.asarray(lyc, dtype=float)
    if fuv.shape != lyc.shape or fuv.ndim != 2:
        raise ValueError("FUV and LyC arrays must be matching two-dimensional maps")
    fuv_level = _normalized_intensity(fuv, "Erad_PE")
    lyc_level = _normalized_intensity(lyc, "Erad_PH")
    fuv_rgb = _get_cmap(fuv_cmap)(fuv_level)[..., :3] * fuv_level[..., None]
    lyc_rgb = _get_cmap(lyc_cmap)(lyc_level)[..., :3] * lyc_level[..., None]
    rgb = 1.0 - (1.0 - fuv_rgb) * (1.0 - lyc_rgb)
    alpha = np.ones(fuv.shape + (1,), dtype=float)
    return np.rint(np.concatenate([rgb, alpha], axis=-1) * 255.0).astype(np.uint8)


def _add_radiation_colorbar(fig, bounds, field, cmap, label):
    axis = fig.add_axes(bounds)
    axis.set_facecolor((0.0, 0.0, 0.0, 0.65))
    scalar = ScalarMappable(norm=field_norm(field), cmap=cmap)
    colorbar = fig.colorbar(scalar, cax=axis, orientation="horizontal")
    colorbar.set_label(label, color="white", fontsize=9, labelpad=1)
    colorbar.ax.tick_params(colors="white", labelsize=7, length=2, pad=1)
    colorbar.outline.set_edgecolor("white")
    colorbar.outline.set_linewidth(0.7)
    return colorbar


def render_radiation_view(slc, plane="x2", *, derived=None, settings=None,
                          fuv_cmap="magma", lyc_cmap="winter", title=None):
    """Render the dual-colormap FUV/LyC screen composite on one slice plane."""
    settings = CanvasSettings() if settings is None else settings
    settings.validate()
    if plane not in slc["planes"]:
        raise KeyError(f"slice frame has no plane {plane!r}")
    plane_data = slc["planes"][plane]
    derived = (
        derive_plane_fields(slc, plane, muH=settings.muH)
        if derived is None
        else derived
    )
    missing = [name for name in ("Erad_PE", "Erad_PH") if name not in derived]
    if missing:
        raise KeyError(f"radiation fields are unavailable: {', '.join(missing)}")
    composite = radiation_composite_rgba(
        derived["Erad_PE"],
        derived["Erad_PH"],
        fuv_cmap=fuv_cmap,
        lyc_cmap=lyc_cmap,
    )
    expected = (plane_data["Ny"], plane_data["Nx"])
    if composite.shape[:2] != expected:
        raise ValueError(f"radiation arrays must match plane shape {expected}")

    figure, canvas, axis = _new_canvas(settings)
    axis.imshow(
        composite,
        origin="lower",
        extent=(
            plane_data["x_edges"][0], plane_data["x_edges"][-1],
            plane_data["y_edges"][0], plane_data["y_edges"][-1],
        ),
        interpolation="nearest",
        aspect="equal",
    )
    _decorate_coordinates(axis, plane_data)
    _add_radiation_colorbar(
        figure, [0.18, 0.895, 0.27, 0.027], "Erad_PE", fuv_cmap,
        r"FUV $[\mathrm{erg\,cm^{-3}}]$",
    )
    _add_radiation_colorbar(
        figure, [0.55, 0.895, 0.27, 0.027], "Erad_PH", lyc_cmap,
        r"LyC $[\mathrm{erg\,cm^{-3}}]$",
    )
    return _finish_canvas(
        figure, canvas, slc, settings, title or "FUV + LyC radiation fields"
    )


def write_png_frame(path, rgba, overwrite=False):
    """Atomically write an RGBA movie frame as PNG."""
    pixels = np.asarray(rgba)
    if pixels.ndim != 3 or pixels.shape[-1] != 4:
        raise ValueError("frame pixels must have shape (height, width, 4)")
    path = Path(path)
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    mpl_image.imsave(temporary, pixels, format="png")
    temporary.replace(path)
    return True
