from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_projections import (
    align_projection_series,
    discover_late_models,
    make_projection_movie,
    packed_layout,
    plot_combined_frame,
    projection_id,
)


def test_projection_id_and_late_discovery(tmp_path):
    assert projection_id(30) == "theta30"
    assert projection_id("theta45") == "theta45"
    assert projection_id("15.5") == "theta15.5"

    early = tmp_path / "R8_8pc_NCR_Lxy2048_early" / "proj2d" / "theta30"
    late_small = tmp_path / "R8_8pc_NCR_Lxy1024_late" / "proj2d" / "theta30"
    late_large = tmp_path / "R8_8pc_NCR_Lxy4096_late" / "proj2d" / "theta30"
    for path in (early, late_small, late_large):
        path.mkdir(parents=True)
    assert discover_late_models(tmp_path, "theta30") == [
        late_large.parents[1],
        late_small.parents[1],
    ]


def test_alignment_uses_time_not_snapshot_number():
    series = [
        [
            (201.0, Path("large.0001.proj2d")),
            (200.0, Path("large.9999.proj2d")),
        ],
        [
            (200.004, Path("medium.0400.proj2d")),
            (201.003, Path("medium.0002.proj2d")),
        ],
    ]
    aligned = align_projection_series(series, time_tolerance=0.01)
    assert aligned == [
        [Path("large.9999.proj2d"), Path("medium.0400.proj2d")],
        [Path("large.0001.proj2d"), Path("medium.0002.proj2d")],
    ]


def test_make_projection_movie_matches_summary_movie(tmp_path, monkeypatch):
    for time_tag in (200, 201):
        (tmp_path / f"projection.theta30.{time_tag:04d}.png").write_bytes(b"png")
    captured = {}
    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_projections.shutil.which",
        lambda name: "/usr/bin/ffmpeg",
    )

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "tigress_ncr_tools.plot_suite_projections.subprocess.run",
        fake_run,
    )
    movie = make_projection_movie(
        tmp_path, "theta30", fps_in=12, fps_out=24,
        codec="libx264", crf=20,
    )
    assert movie == tmp_path / "projection.theta30.mp4"
    assert captured["kwargs"] == {"check": True}
    assert captured["command"][:5] == ["ffmpeg", "-y", "-r", "12", "-f"]
    assert ["-r", "24"] == captured["command"][10:12]
    assert ["-crf", "20", "-preset", "slow"] == captured["command"][16:20]
    assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in captured["command"]


def test_packed_layout_matches_reference_notebook():
    extents, xlim, ylim = packed_layout([4096, 2048, 1024], gap=16)
    assert extents == [
        (-2048, 2048, -2048, 2048),
        (2064, 4112, -2048, 0),
        (2064, 3088, 16, 1040),
    ]
    assert xlim == (-2048, 4112)
    assert ylim == (-2048, 2048)


def test_plot_combined_frame():
    models = [
        Path("R8_8pc_NCR_Lxy4096_late"),
        Path("R8_8pc_NCR_Lxy2048_late"),
        Path("R8_8pc_NCR_Lxy1024_late"),
    ]
    frames = []
    for width, shape in zip((4096, 2048, 1024), (8, 4, 2)):
        frames.append({
            "path": f"theta30-{width}",
            "time": 300.0,
            "theta": 30.0,
            "phi": 270.0,
            "x_edges": np.linspace(-width / 2, width / 2, shape + 1),
            "y_edges": np.linspace(-width / 2, width / 2, shape + 1),
            "fields": {"nH": np.ones((shape, shape))},
        })
    fig = plot_combined_frame(models, frames)
    assert len(fig.axes[0].images) == 3
    assert tuple(fig.axes[0].get_xlim()) == (-2048, 4112)
    assert tuple(fig.axes[0].get_ylim()) == (-2048, 2048)
    fig.clear()
