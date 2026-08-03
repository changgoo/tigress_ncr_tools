from dataclasses import fields
from pathlib import Path

import matplotlib.image as mpl_image
import numpy as np
import pytest

import tigress_ncr_tools.slice_story_movie as movie
from tigress_ncr_tools.story_renderers import CanvasSettings
from tigress_ncr_tools.storyboard import FrameRequest, StoryDurations


RAW_BASE = ["density", "pressure", "velocity", "xHI", "xH2"]
RAW_SIDE = RAW_BASE + [
    "cell_centered_B",
    "rad_energy_density_PE",
    "rad_energy_density_PH",
]


def record(num="0001", time=1.0, x3=None, x2=None):
    return {
        "path": f"run.{num}.midplane.slice.vtk",
        "num": num,
        "time": time,
        "field_names": {
            "x3": list(RAW_BASE if x3 is None else x3),
            "x2": list(RAW_SIDE if x2 is None else x2),
        },
    }


def request(frame_number, *, field_a="nH", field_b=None, blend=0.0,
            particle_alpha=0.0):
    return FrameRequest(
        frame_number=frame_number,
        scene="test",
        source_index=0,
        source_path="source.slice.vtk",
        source_num="0001",
        simulation_time=1.0,
        view_a="slice",
        plane_a="x3",
        field_a=field_a,
        view_b="slice" if field_b else None,
        plane_b="x3" if field_b else None,
        field_b=field_b,
        blend=blend,
        particle_alpha=particle_alpha,
    )


def test_preflight_reports_capabilities_aliases_and_missing_fields():
    ready = movie.slice_preflight_report([record(), record("0002", 2.0)])
    assert ready["ready_for_slice_story"]
    assert ready["frame_count"] == 2
    assert ready["first_time"] == 1.0
    assert ready["last_time"] == 2.0

    legacy = record(
        x3=["density", "pressure", "velocity", "specific_scalar_2",
            "specific_scalar_3"]
    )
    assert movie.slice_preflight_report([legacy])["ready_for_slice_story"]

    broken = record(x2=["density"])
    report = movie.slice_preflight_report([broken])
    assert not report["ready_for_slice_story"]
    assert "cell_centered_B" in report["issues"][0]
    with pytest.raises(ValueError, match="preflight failed"):
        movie.require_slice_story_capabilities([broken])


def test_duration_scaling_and_series_pattern():
    scaled = movie.scaled_durations(0.25)
    defaults = StoryDurations()
    for item in fields(defaults):
        assert getattr(scaled, item.name) == 0.25 * getattr(defaults, item.name)
    with pytest.raises(ValueError, match="positive"):
        movie.scaled_durations(0.0)
    assert movie.slice_series_pattern("/run", "R8", "midplane") == Path(
        "/run/slice/midplane/R8.*.midplane.slice.vtk"
    )


def test_render_story_frames_blends_caches_and_resumes(tmp_path, monkeypatch):
    calls = {"read": 0, "derive": 0, "render": []}

    def fake_read(path):
        calls["read"] += 1
        return {"time": 1.0, "planes": {"x3": {}}}

    def fake_derive(slc, plane, muH):
        calls["derive"] += 1
        return {}

    def fake_render(slc, view, plane, field, **kwargs):
        calls["render"].append(field)
        value = {"nH": 0, "nH2": 200}[field]
        settings = kwargs["settings"]
        return np.full(
            (settings.height, settings.width, 4), value, dtype=np.uint8
        )

    monkeypatch.setattr(movie, "read_slicevtk", fake_read)
    monkeypatch.setattr(movie, "derive_plane_fields", fake_derive)
    monkeypatch.setattr(movie, "_render_single_view", fake_render)
    requests = [
        request(0),
        request(1, field_b="nH2", blend=0.5),
    ]
    settings = CanvasSettings(width=20, height=12, dpi=10)
    result = movie.render_story_frames(
        requests,
        tmp_path,
        "R8",
        tmp_path / "output",
        settings=settings,
        particles=False,
    )
    assert len(result["written"]) == 2
    assert calls == {"read": 1, "derive": 1, "render": ["nH", "nH2"]}
    blended = mpl_image.imread(result["written"][1])
    assert np.allclose(blended, 100.0 / 255.0)

    resumed = movie.render_story_frames(
        requests,
        tmp_path,
        "R8",
        tmp_path / "output",
        settings=settings,
        particles=False,
    )
    assert len(resumed["written"]) == 0
    assert len(resumed["skipped"]) == 2
    assert calls == {"read": 1, "derive": 1, "render": ["nH", "nH2"]}


def test_particle_time_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        movie, "read_slicevtk", lambda path: {"time": 1.0, "planes": {"x3": {}}}
    )
    monkeypatch.setattr(movie, "derive_plane_fields", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        movie,
        "_particle_frame",
        lambda *args: {"time": 2.0, "particles": {}},
    )
    monkeypatch.setattr(
        movie,
        "_render_single_view",
        lambda *args, **kwargs: np.zeros((8, 12, 4), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="particle time mismatch"):
        movie.render_story_frames(
            [request(0, particle_alpha=1.0)],
            tmp_path,
            "R8",
            tmp_path / "output",
            settings=CanvasSettings(width=12, height=8, dpi=10),
            time_tolerance=0.01,
        )


def test_encode_story_movie_uses_contiguous_numbered_frames(tmp_path, monkeypatch):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000000.png").write_bytes(b"png")
    (frames / "frame_000001.png").write_bytes(b"png")
    captured = {}
    monkeypatch.setattr(movie.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(movie, "resolve_video_codec", lambda codec: "libx264")

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(movie.subprocess, "run", fake_run)
    output = movie.encode_story_movie(
        frames, tmp_path / "story.mp4", fps=12, codec="auto", crf=20
    )
    assert output == tmp_path / "story.mp4"
    assert captured["kwargs"] == {"check": True}
    assert ["-framerate", "12"] == captured["command"][2:4]
    assert ["-crf", "20", "-preset", "slow"] == captured["command"][14:18]

    (frames / "frame_000001.png").rename(frames / "frame_000002.png")
    with pytest.raises(ValueError, match="contiguous"):
        movie.encode_story_movie(frames, tmp_path / "bad.mp4", codec="libx264")
