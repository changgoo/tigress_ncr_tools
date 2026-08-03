import numpy as np
import pytest

from pathena.units import tigress_units
from tigress_ncr_tools.story_renderers import CanvasSettings, render_volume_view
from tigress_ncr_tools.volume_renderer import (
    VolumeImage,
    camera_basis,
    render_temperature_volume,
    volume_input_fields,
    volume_temperature,
)


def synthetic_volume():
    z, y, x = np.indices((8, 6, 4))
    temperature = 10.0 ** (2.0 + 4.0 * z / 7.0)
    density = 10.0 ** (-2.0 + 4.0 * (x + y) / 8.0)
    return {
        "time": 12.0,
        "left_edge": np.array([-2.0, -3.0, -4.0]),
        "right_edge": np.array([2.0, 3.0, 4.0]),
        "spacing": np.ones(3),
        "fields": {
            "density": np.broadcast_to(density, (8, 6, 4)).copy(),
            "temperature": np.broadcast_to(temperature, (8, 6, 4)).copy(),
        },
    }


def test_camera_basis_has_registered_orthogonal_endpoints():
    top_normal, top_north = camera_basis(0.0)
    side_normal, side_north = camera_basis(1.0)
    assert np.allclose(top_normal, [0.0, 0.0, -1.0])
    assert np.allclose(top_north, [0.0, 1.0, 0.0])
    assert np.allclose(side_normal, [0.0, 1.0, 0.0])
    assert np.allclose(side_north, [0.0, 0.0, 1.0])
    for fraction in np.linspace(0.0, 1.0, 7):
        normal, north = camera_basis(fraction)
        assert np.isclose(np.linalg.norm(normal), 1.0)
        assert np.isclose(np.linalg.norm(north), 1.0)
        assert np.isclose(np.dot(normal, north), 0.0)
    with pytest.raises(ValueError, match="camera_fraction"):
        camera_basis(1.1)


def test_volume_field_selection_and_temperature_fallback():
    assert volume_input_fields(["density", "temperature", "pressure"]) == (
        "density", "temperature"
    )
    selected = volume_input_fields(
        ["density", "pressure", "specific_scalar_3", "specific_scalar_4"]
    )
    assert selected == (
        "density", "pressure", "specific_scalar_3", "specific_scalar_4"
    )
    shape = (2, 3, 4)
    fallback = {
        "fields": {
            "density": np.full(shape, 2.0),
            "pressure": np.full(shape, 4.0),
            "specific_scalar_3": np.full(shape, 0.1),
            "specific_scalar_4": np.full(shape, 0.2),
        }
    }
    expected = (
        4.0 * tigress_units()["temperature_per_p_over_rho"]
        / (2.0 * (1.1 + 0.2 - 0.1))
    )
    assert np.allclose(volume_temperature(fallback), expected)
    with pytest.raises(KeyError, match="density"):
        volume_input_fields(["temperature"])
    with pytest.raises(KeyError, match="xH2"):
        volume_input_fields(["density", "pressure"])


def test_temperature_volume_registers_top_and_side_extents():
    volume = synthetic_volume()
    top = render_temperature_volume(volume, 0.0, opacity_scale=0.2)
    middle = render_temperature_volume(volume, 0.5, opacity_scale=0.2)
    side = render_temperature_volume(volume, 1.0, opacity_scale=0.2)
    assert top.rgba.shape == (6, 4, 4)
    assert side.rgba.shape == (8, 4, 4)
    assert middle.rgba.shape[0] > top.rgba.shape[0]
    assert top.extent == (-2.0, 2.0, -3.0, 3.0)
    assert side.extent == (-2.0, 2.0, -4.0, 4.0)
    assert np.count_nonzero(top.rgba[..., :3]) > 0
    assert np.count_nonzero(side.rgba[..., :3]) > 0
    assert np.all(top.rgba[..., 3] == 255)
    with pytest.raises(ValueError, match="stride"):
        render_temperature_volume(volume, 0.0, stride=0)


def test_volume_image_uses_shared_fixed_canvas():
    raw = VolumeImage(
        rgba=np.full((8, 4, 4), 128, dtype=np.uint8),
        extent=(-2.0, 2.0, -4.0, 4.0),
        camera_fraction=1.0,
    )
    raw.rgba[..., 3] = 255
    canvas = render_volume_view(
        raw,
        12.0,
        settings=CanvasSettings(width=320, height=180, dpi=100),
    )
    assert canvas.shape == (180, 320, 4)
    assert canvas.dtype == np.uint8
    assert np.all(canvas[..., 3] == 255)
