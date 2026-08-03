import csv
from dataclasses import fields

import pytest

from tigress_ncr_tools.storyboard import (
    FrameRequest,
    StoryDurations,
    build_slice_storyboard,
    duration_frame_count,
    smootherstep,
    transition_fractions,
    write_frame_manifest,
)


def records(times=(0.0, 5.0, 10.0)):
    return [
        {"path": f"slice.{index:04d}.vtk", "num": f"{index:04d}", "time": time}
        for index, time in enumerate(times)
    ]


def short_durations():
    return StoryDurations(**{
        item.name: 0.5
        for item in fields(StoryDurations)
    })


def scene_frames(story, scene):
    return [frame for frame in story if frame.scene == scene]


def test_easing_and_frame_counts_have_exact_endpoints():
    assert smootherstep(-1.0) == 0.0
    assert smootherstep(0.0) == 0.0
    assert smootherstep(1.0) == 1.0
    assert smootherstep(2.0) == 1.0
    fractions = transition_fractions(7)
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert duration_frame_count(0.0, 30) == 0
    assert duration_frame_count(0.01, 30) == 2
    with pytest.raises(ValueError, match="fps"):
        duration_frame_count(1.0, 0.0)


def test_default_storyboard_sequence_and_frozen_sources():
    story = build_slice_storyboard(
        records(),
        fps=4,
        freeze_index=1,
        durations=short_durations(),
    )
    assert [frame.frame_number for frame in story] == list(range(len(story)))
    assert len(story) == 26
    assert list(dict.fromkeys(frame.scene for frame in story)) == [
        "top_density_evolution",
        "particle_fade",
        "molecular_density",
        "atomic_density",
        "ionized_density",
        "temperature",
        "volume_reveal",
        "camera_turn",
        "side_slice_reveal",
        "side_temperature_evolution",
        "velocity_streamlines",
        "magnetic_streamlines",
        "radiation_composite",
    ]

    top = scene_frames(story, "top_density_evolution")
    assert [frame.source_index for frame in top] == [0, 1]
    assert all(frame.plane_a == "x3" and frame.field_a == "nH" for frame in top)
    assert all(frame.particle_alpha == 1.0 for frame in top)

    particle_fade = scene_frames(story, "particle_fade")
    assert [frame.particle_alpha for frame in particle_fade] == [1.0, 0.0]
    frozen_scenes = story[len(top):len(story) - 8]
    assert all(frame.source_index == 1 for frame in frozen_scenes)

    molecular = scene_frames(story, "molecular_density")
    assert (molecular[0].field_a, molecular[0].field_b) == ("nH", "nH2")
    assert [frame.blend for frame in molecular] == [0.0, 1.0]
    camera = scene_frames(story, "camera_turn")
    assert [frame.camera_fraction for frame in camera] == [0.0, 1.0]

    side = scene_frames(story, "side_temperature_evolution")
    assert [frame.source_index for frame in side] == [1, 2]
    assert all(frame.plane_a == "x2" and frame.field_a == "T" for frame in side)
    assert all(
        frame.source_index == 2
        for frame in scene_frames(story, "radiation_composite")
    )


def test_storyboard_can_skip_volume_and_samples_physical_time():
    durations = short_durations()
    durations = StoryDurations(
        **{
            **{item.name: getattr(durations, item.name) for item in fields(durations)},
            "top_evolution": 1.25,
        }
    )
    story = build_slice_storyboard(
        records((0.0, 1.0, 9.0, 10.0)),
        fps=4,
        freeze_index=3,
        durations=durations,
        include_volume=False,
    )
    names = {frame.scene for frame in story}
    assert "volume_reveal" not in names
    assert "camera_turn" not in names
    top = scene_frames(story, "top_density_evolution")
    assert [frame.source_index for frame in top] == [0, 1, 1, 2, 3]
    reveal = scene_frames(story, "side_slice_reveal")
    assert reveal[0].view_a == "slice"
    assert reveal[0].plane_a == "x3"


def test_storyboard_validation_and_atomic_manifest(tmp_path):
    with pytest.raises(ValueError, match="sorted"):
        build_slice_storyboard(records((1.0, 0.0)))
    with pytest.raises(ValueError, match="start_index"):
        build_slice_storyboard(records(), start_index=2, freeze_index=1)
    with pytest.raises(ValueError, match="non-negative"):
        build_slice_storyboard(
            records(),
            durations=StoryDurations(top_evolution=-1.0),
        )

    story = build_slice_storyboard(
        records(), fps=4, durations=short_durations(), include_volume=False
    )
    path = write_frame_manifest(tmp_path / "output" / "manifest.csv", story)
    assert path.exists()
    assert not path.with_suffix(".csv.tmp").exists()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(story)
    assert rows[0]["frame_number"] == "0"
    assert rows[-1]["scene"] == "radiation_composite"
