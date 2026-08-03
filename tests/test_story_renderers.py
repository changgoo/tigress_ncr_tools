import matplotlib.image as mpl_image
import numpy as np
import pytest

from tigress_ncr_tools.story_renderers import (
    CanvasSettings,
    blend_rgba,
    field_norm,
    field_style,
    radiation_composite_rgba,
    render_radiation_view,
    render_slice_view,
    render_streamline_view,
    write_png_frame,
)


def synthetic_slice():
    x3_shape = (6, 8)
    x2_shape = (10, 8)
    return {
        "time": 20.0,
        "planes": {
            "x3": {
                "normal": "x3",
                "Nx": x3_shape[1],
                "Ny": x3_shape[0],
                "x_edges": np.linspace(-40.0, 40.0, x3_shape[1] + 1),
                "y_edges": np.linspace(-30.0, 30.0, x3_shape[0] + 1),
                "fields": {},
            },
            "x2": {
                "normal": "x2",
                "Nx": x2_shape[1],
                "Ny": x2_shape[0],
                "x_edges": np.linspace(-40.0, 40.0, x2_shape[1] + 1),
                "y_edges": np.linspace(-100.0, 100.0, x2_shape[0] + 1),
                "fields": {},
            },
        },
    }


def derived(shape):
    y, x = np.indices(shape)
    base = 10.0 ** (-2.0 + 5.0 * (x + y) / max(1, sum(shape) - 2))
    xc = x - 0.5 * (shape[1] - 1)
    yc = y - 0.5 * (shape[0] - 1)
    return {
        "nH": base,
        "nH2": 0.25 * base,
        "nHI": 0.5 * base,
        "nHII": 0.05 * base,
        "T": 10.0 ** (2.0 + 4.0 * y / max(1, shape[0] - 1)),
        "v1": -yc,
        "v3": xc,
        "B1": np.ones(shape),
        "B3": 0.25 * np.sin(x / 2.0),
        "Bmag": 1.0 + 10.0 * (x + 1.0) / shape[1],
        "Erad_PE": 10.0 ** (-15.0 + 5.0 * x / max(1, shape[1] - 1)),
        "Erad_PH": 10.0 ** (-15.0 + 5.0 * y / max(1, shape[0] - 1)),
    }


def test_density_fields_share_fixed_style_and_norm():
    density_fields = ("nH", "nH2", "nHI", "nHII")
    styles = [field_style(field) for field in density_fields]
    norms = [field_norm(field) for field in density_fields]
    assert {style["cmap"] for style in styles} == {"Spectral_r"}
    assert {(norm.vmin, norm.vmax) for norm in norms} == {(1.0e-4, 1.0e4)}
    styles[0]["vmin"] = 99.0
    assert field_style("nH")["vmin"] == 1.0e-4


def test_render_slice_canvas_dimensions_planes_and_particles():
    slc = synthetic_slice()
    settings = CanvasSettings(width=320, height=180, dpi=100)
    top_fields = derived((6, 8))
    plain = render_slice_view(
        slc, "x3", "nH", derived=top_fields, settings=settings
    )
    assert plain.shape == (180, 320, 4)
    assert plain.dtype == np.uint8
    assert np.all(plain[..., 3] == 255)

    particles = {
        "x1": np.array([0.0, 15.0, -15.0]),
        "x2": np.array([0.0, 10.0, -10.0]),
        "x3": np.array([0.0, 20.0, -20.0]),
        "mass": np.array([1.0e8, 0.0, 0.0]),
        "age": np.zeros(3),
        "id": np.array([1.0, 2.0, -3.0]),
    }
    with_particles = render_slice_view(
        slc,
        "x3",
        "nH",
        derived=top_fields,
        particles=particles,
        particle_alpha=1.0,
        settings=settings,
    )
    assert np.count_nonzero(with_particles != plain) > 0

    side = render_slice_view(
        slc, "x2", "T", derived=derived((10, 8)), settings=settings
    )
    assert side.shape == plain.shape
    assert np.count_nonzero(side != plain) > 0


def test_rgba_blending_has_exact_endpoints_and_validation():
    black = np.zeros((2, 3, 4), dtype=np.uint8)
    white = np.full((2, 3, 4), 255, dtype=np.uint8)
    assert np.array_equal(blend_rgba(black, white, 0.0), black)
    assert np.array_equal(blend_rgba(black, white, 1.0), white)
    assert np.all(blend_rgba(black, white, 0.5) == 128)
    with pytest.raises(ValueError, match="matching"):
        blend_rgba(black, white[:, :2], 0.5)
    with pytest.raises(ValueError, match="fraction"):
        blend_rgba(black, white, 1.1)


def test_streamline_views_and_radiation_canvas_are_fixed_size():
    slc = synthetic_slice()
    settings = CanvasSettings(width=320, height=180, dpi=100)
    fields = derived((10, 8))
    velocity = render_streamline_view(
        slc, "x2", "velocity", derived=fields, settings=settings, density=0.6
    )
    magnetic = render_streamline_view(
        slc, "x2", "magnetic", derived=fields, settings=settings, density=0.6
    )
    radiation = render_radiation_view(
        slc, "x2", derived=fields, settings=settings
    )
    assert velocity.shape == magnetic.shape == radiation.shape == (180, 320, 4)
    assert np.count_nonzero(velocity != magnetic) > 0
    assert np.count_nonzero(magnetic != radiation) > 0
    with pytest.raises(ValueError, match="kind"):
        render_streamline_view(
            slc, "x2", "invalid", derived=fields, settings=settings
        )


def test_radiation_screen_composite_preserves_two_channels():
    low = np.zeros((3, 4))
    high = np.full((3, 4), 1.0e-10)
    dark = radiation_composite_rgba(low, low)
    fuv = radiation_composite_rgba(high, low)
    lyc = radiation_composite_rgba(low, high)
    combined = radiation_composite_rgba(high, high)
    assert np.all(dark[..., :3] == 0)
    assert np.all(dark[..., 3] == 255)
    assert np.count_nonzero(fuv[..., :3] != lyc[..., :3]) > 0
    assert np.all(combined[..., :3] >= fuv[..., :3])
    assert np.all(combined[..., :3] >= lyc[..., :3])
    with pytest.raises(ValueError, match="matching"):
        radiation_composite_rgba(low, high[:, :2])


def test_atomic_png_write_and_overwrite(tmp_path):
    pixels = np.zeros((12, 20, 4), dtype=np.uint8)
    pixels[..., 0] = 100
    pixels[..., 3] = 255
    path = tmp_path / "frames" / "frame_000000.png"
    assert write_png_frame(path, pixels)
    assert not write_png_frame(path, np.full_like(pixels, 255))
    loaded = mpl_image.imread(path)
    assert loaded.shape == pixels.shape
    assert np.allclose(loaded[..., 0], 100.0 / 255.0)
    assert write_png_frame(path, np.full_like(pixels, 255), overwrite=True)
    assert np.allclose(mpl_image.imread(path), 1.0)
