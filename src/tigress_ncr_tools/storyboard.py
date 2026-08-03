"""Deterministic storyboard expansion for the slice story movie."""

from bisect import bisect_left
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SourceFrame:
    """One physically ordered simulation output available to the movie."""

    source_index: int
    path: str
    num: Optional[str]
    time: float

    @classmethod
    def from_record(cls, source_index, record):
        return cls(
            source_index=source_index,
            path=str(record["path"]),
            num=record.get("num"),
            time=float(record["time"]),
        )


@dataclass(frozen=True)
class StoryDurations:
    """Default scene durations in seconds."""

    top_evolution: float = 10.0
    particle_fade: float = 1.0
    species_transition: float = 1.25
    temperature_transition: float = 1.5
    volume_reveal: float = 2.0
    camera_turn: float = 4.0
    side_reveal: float = 1.5
    side_evolution: float = 10.0
    velocity_reveal: float = 5.0
    magnetic_transition: float = 5.0
    radiation_transition: float = 5.0

    def validate(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if value < 0.0:
                raise ValueError(f"duration {item.name} must be non-negative")


@dataclass(frozen=True)
class FrameRequest:
    """Serializable instructions for rendering one output movie frame."""

    frame_number: int
    scene: str
    source_index: int
    source_path: str
    source_num: Optional[str]
    simulation_time: float
    view_a: str
    plane_a: Optional[str]
    field_a: Optional[str]
    view_b: Optional[str] = None
    plane_b: Optional[str] = None
    field_b: Optional[str] = None
    blend: float = 0.0
    particle_alpha: float = 0.0
    camera_fraction: float = 0.0

    def manifest_row(self):
        """Return a flat dictionary suitable for ``csv.DictWriter``."""
        return asdict(self)


def smootherstep(value):
    """Return a clamped fifth-order easing value with flat endpoints."""
    x = min(1.0, max(0.0, float(value)))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def duration_frame_count(seconds, fps):
    """Convert a positive duration to at least two endpoint-bearing frames."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if seconds < 0.0:
        raise ValueError("duration must be non-negative")
    if seconds == 0.0:
        return 0
    return max(2, int(round(seconds * fps)))


def transition_fractions(count):
    """Return ``count`` eased fractions containing exact zero and one."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    if count == 1:
        return [1.0]
    return [smootherstep(index / (count - 1)) for index in range(count)]


def _source_frames(records):
    sources = [
        SourceFrame.from_record(index, record)
        for index, record in enumerate(records)
    ]
    if not sources:
        raise ValueError("at least one source frame is required")
    if any(
        later.time < earlier.time
        for earlier, later in zip(sources, sources[1:])
    ):
        raise ValueError("source frames must be sorted by simulation time")
    return sources


def _nearest_source_indices(sources, start_index, stop_index, count):
    if count <= 0:
        return []
    selected = sources[start_index:stop_index + 1]
    times = [source.time for source in selected]
    start_time, stop_time = times[0], times[-1]
    if count == 1 or start_time == stop_time:
        return [start_index] * count

    result = []
    for fraction in (index / (count - 1) for index in range(count)):
        target = start_time + fraction * (stop_time - start_time)
        position = bisect_left(times, target)
        choices = [
            index for index in (position - 1, position)
            if 0 <= index < len(times)
        ]
        nearest = min(choices, key=lambda index: abs(times[index] - target))
        result.append(start_index + nearest)
    return result


class _StoryBuilder:
    def __init__(self, sources, fps):
        self.sources = sources
        self.fps = fps
        self.frames = []

    def append(self, scene, source, view_a, *, plane_a=None, field_a=None,
               view_b=None, plane_b=None, field_b=None, blend=0.0,
               particle_alpha=0.0, camera_fraction=0.0):
        self.frames.append(FrameRequest(
            frame_number=len(self.frames),
            scene=scene,
            source_index=source.source_index,
            source_path=source.path,
            source_num=source.num,
            simulation_time=source.time,
            view_a=view_a,
            plane_a=plane_a,
            field_a=field_a,
            view_b=view_b,
            plane_b=plane_b,
            field_b=field_b,
            blend=blend,
            particle_alpha=particle_alpha,
            camera_fraction=camera_fraction,
        ))

    def evolution(self, scene, start_index, stop_index, seconds, *,
                  plane, field, particles=False):
        count = duration_frame_count(seconds, self.fps)
        for source_index in _nearest_source_indices(
            self.sources, start_index, stop_index, count
        ):
            self.append(
                scene,
                self.sources[source_index],
                "slice",
                plane_a=plane,
                field_a=field,
                particle_alpha=1.0 if particles else 0.0,
            )

    def transition(self, scene, source_index, seconds, view_a, view_b, *,
                   plane_a=None, field_a=None, plane_b=None, field_b=None,
                   particles_a=0.0, particles_b=0.0,
                   camera_a=0.0, camera_b=0.0):
        source = self.sources[source_index]
        count = duration_frame_count(seconds, self.fps)
        for blend in transition_fractions(count):
            self.append(
                scene,
                source,
                view_a,
                plane_a=plane_a,
                field_a=field_a,
                view_b=view_b,
                plane_b=plane_b,
                field_b=field_b,
                blend=blend,
                particle_alpha=(
                    particles_a + blend * (particles_b - particles_a)
                ),
                camera_fraction=camera_a + blend * (camera_b - camera_a),
            )


def build_slice_storyboard(
    records,
    *,
    fps=30.0,
    start_index=0,
    freeze_index=None,
    stop_index=None,
    durations=None,
    include_volume=True,
):
    """Expand indexed slice records into the default movie frame requests.

    ``freeze_index`` is the snapshot used for gas-phase, temperature, and
    volume transitions. The XZ evolution resumes there and ends at
    ``stop_index``.
    """
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    durations = StoryDurations() if durations is None else durations
    durations.validate()
    sources = _source_frames(records)
    last_index = len(sources) - 1
    stop_index = last_index if stop_index is None else stop_index
    freeze_index = (
        (start_index + stop_index) // 2
        if freeze_index is None
        else freeze_index
    )
    if not 0 <= start_index <= freeze_index <= stop_index <= last_index:
        raise ValueError(
            "require 0 <= start_index <= freeze_index <= stop_index < "
            "number of sources"
        )

    story = _StoryBuilder(sources, fps)
    story.evolution(
        "top_density_evolution",
        start_index,
        freeze_index,
        durations.top_evolution,
        plane="x3",
        field="nH",
        particles=True,
    )
    story.transition(
        "particle_fade",
        freeze_index,
        durations.particle_fade,
        "slice",
        "slice",
        plane_a="x3",
        field_a="nH",
        plane_b="x3",
        field_b="nH",
        particles_a=1.0,
        particles_b=0.0,
    )

    previous = "nH"
    for field, name in (
        ("nH2", "molecular_density"),
        ("nHI", "atomic_density"),
        ("nHII", "ionized_density"),
    ):
        story.transition(
            name,
            freeze_index,
            durations.species_transition,
            "slice",
            "slice",
            plane_a="x3",
            field_a=previous,
            plane_b="x3",
            field_b=field,
        )
        previous = field

    story.transition(
        "temperature",
        freeze_index,
        durations.temperature_transition,
        "slice",
        "slice",
        plane_a="x3",
        field_a="nHII",
        plane_b="x3",
        field_b="T",
    )

    if include_volume:
        story.transition(
            "volume_reveal",
            freeze_index,
            durations.volume_reveal,
            "slice",
            "volume",
            plane_a="x3",
            field_a="T",
            field_b="T",
        )
        story.transition(
            "camera_turn",
            freeze_index,
            durations.camera_turn,
            "volume",
            "volume",
            field_a="T",
            field_b="T",
            camera_a=0.0,
            camera_b=1.0,
        )
        side_view_a = "volume"
        side_plane_a = None
    else:
        side_view_a = "slice"
        side_plane_a = "x3"

    story.transition(
        "side_slice_reveal",
        freeze_index,
        durations.side_reveal,
        side_view_a,
        "slice",
        plane_a=side_plane_a,
        field_a="T",
        plane_b="x2",
        field_b="T",
        camera_a=1.0 if include_volume else 0.0,
        camera_b=1.0 if include_volume else 0.0,
    )
    story.evolution(
        "side_temperature_evolution",
        freeze_index,
        stop_index,
        durations.side_evolution,
        plane="x2",
        field="T",
    )
    story.transition(
        "velocity_streamlines",
        stop_index,
        durations.velocity_reveal,
        "slice",
        "velocity_streamlines",
        plane_a="x2",
        field_a="T",
        plane_b="x2",
        field_b="T",
    )
    story.transition(
        "magnetic_streamlines",
        stop_index,
        durations.magnetic_transition,
        "velocity_streamlines",
        "magnetic_streamlines",
        plane_a="x2",
        field_a="T",
        plane_b="x2",
        field_b="Bmag",
    )
    story.transition(
        "radiation_composite",
        stop_index,
        durations.radiation_transition,
        "magnetic_streamlines",
        "radiation_composite",
        plane_a="x2",
        field_a="Bmag",
        plane_b="x2",
        field_b="Erad_PE+Erad_PH",
    )
    return story.frames


def write_frame_manifest(path, frames):
    """Write frame requests as a CSV manifest."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [item.name for item in fields(FrameRequest)]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame.manifest_row() for frame in frames)
    temporary.replace(path)
    return path
