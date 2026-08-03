"""Fixed-layout renderers and pixel compositors for slice movie frames."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpl_image
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
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

    figure = Figure(
        figsize=(settings.width / settings.dpi, settings.height / settings.dpi),
        dpi=settings.dpi,
        facecolor=settings.background,
    )
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes([0.045, 0.055, 0.91, 0.875])
    axis.set_facecolor(settings.background)
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
    axis.set_axis_off()
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

    effects = _text_effects()
    figure.text(
        0.025, 0.965, title or style["short"], color="white", fontsize=18,
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
