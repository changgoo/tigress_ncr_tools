import matplotlib
import numpy as np

matplotlib.use("Agg")

from pathena.proj2d_reader import plot_physical_maps
from pathena.projection import project_shearing_periodic


def test_plot_physical_maps():
    shape = (2, 3)
    one = np.ones(shape)
    proj = {
        "x_edges": np.arange(4), "y_edges": np.arange(3),
        "fields": {
            "nH": 4*one, "nHI": one, "nH2": one,
            "nH*Vlos": 8*one, "ne": one, "ne*Blos": -2*one,
            "nHI*Vlos": 3*one, "nHI*Vlos2": 13*one,
            "nHI_CNM": 0.25*one, "nHI_WNM": 0.5*one,
        },
    }
    fig, axes = plot_physical_maps(proj)
    assert len([ax for ax in axes if ax.get_visible()]) == 8
    fig.clear()

    proj["fields"] = {"nHI_WNM": one}
    fig, axes = plot_physical_maps(proj)
    assert len([ax for ax in axes if ax.get_visible()]) == 1
    fig.clear()


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
