import matplotlib
import numpy as np

matplotlib.use("Agg")

from pathena.proj2d_reader import plot_physical_maps
from pathena.projection import project_shearing_periodic


def test_plot_physical_maps():
    shape = (2, 3)
    one = np.ones(shape)
    wnm = np.arange(1, 7, dtype=float).reshape(shape)
    proj = {
        "x_edges": np.arange(4), "y_edges": np.arange(3),
        "fields": {
            "nH": 4*one, "nHI": one, "nH2": one,
            "nH*Vlos": 8*one, "ne": one, "ne*Blos": -2*one,
            "nHI*Vlos": 3*one, "nHI*Vlos2": 13*one,
            "nHI_CNM": 0.25*one, "nHI_WNM": wnm,
        },
    }
    fig, axes = plot_physical_maps(proj)
    assert len([ax for ax in axes if ax.get_visible()]) == 8
    rgb = np.asarray(axes[0].images[0].get_array())
    assert np.allclose(rgb[..., 1], 0.65*rgb[..., 2])
    cnm_norm = axes[6].collections[0].norm
    wnm_norm = axes[7].collections[0].norm
    assert (cnm_norm.vmin, cnm_norm.vmax) == (wnm_norm.vmin, wnm_norm.vmax)
    fig.clear()

    proj["fields"] = {"nHI_WNM": one}
    fig, axes = plot_physical_maps(proj)
    assert len([ax for ax in axes if ax.get_visible()]) == 1
    fig.clear()

    try:
        plot_physical_maps(proj, hi_green_scale=-0.1)
    except ValueError as error:
        assert "hi_green_scale" in str(error)
    else:
        raise AssertionError("negative hi_green_scale should fail")


def test_vlos2_projection():
    one = np.ones((1, 1, 1))
    out = project_shearing_periodic(
        {"nHI": 3*one}, ((0, 1), (0, 1), (0, 1)),
        fields=["Vlos2", "nHI*Vlos2", "Blos2", "nHI*Blos2"],
        vectors={"V": (one, 2*one, 4*one), "B": (2*one, 3*one, 5*one)}, theta=0,
    )
    assert np.allclose(out["maps"]["Vlos2"], 16)
    assert np.allclose(out["maps"]["nHI*Vlos2"], 48)
    assert np.allclose(out["maps"]["Blos2"], 25)
    assert np.allclose(out["maps"]["nHI*Blos2"], 75)
