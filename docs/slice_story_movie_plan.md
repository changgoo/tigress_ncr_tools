# Slice Story Movie Plan

## Goal

Build a reproducible command-line renderer that turns TIGRESS-NCR slice,
star-particle, and selected full-volume outputs into a seamless sequence of PNG
frames and, optionally, an H.264 movie for a scientific talk.

The renderer will keep the canvas, annotations, axes, and colorbar locations
fixed while the scientific view evolves in time or transitions between fields.
Every generated frame will be described by a manifest so interrupted renders
can be resumed and the final movie can be reproduced.

## Default Storyboard

The default cut targets about 50 seconds at 30 frames per second. All durations
will be configurable.

| Scene | Approximate duration | Content |
| --- | ---: | --- |
| Top-down evolution | 10 s | Evolving `x3` midplane total density with star particles |
| Particle fade | 1 s | Freeze the simulation time and fade particles out |
| Gas phases | 5 s | Crossfade through molecular, atomic, and ionized density |
| Temperature | 1.5 s | Crossfade from ionized density to temperature |
| Volume reveal | 2 s | Dissolve the temperature slice into a temperature-colored volume |
| Camera turn | 4 s | Rotate the frozen volume from top-down to side-on |
| Side-slice reveal | 1.5 s | Dissolve the side-on volume into the corresponding `x2`/XZ slice |
| Side evolution | 10 s | Evolve the XZ temperature slice forward in time |
| Velocity flow | 5 s | Freeze and reveal velocity streamlines colored by temperature |
| Magnetic flow | 5 s | Crossfade to magnetic streamlines colored by field strength |
| Radiation | 5 s | Crossfade to a two-color FUV and LyC radiation composite |

An optional closing scene may dissolve back to the opening frame for continuous
looping. It will be disabled by default because it does not preserve simulation
time continuity.

## Scientific Fields

Slice fields will use the definitions already established in
`pathena.plot_helpers`:

- Total hydrogen density: `nH = density`
- Molecular hydrogen density: `nH2 = nH*xH2`
- Atomic hydrogen density: `nHI = nH*xHI`
- Ionized hydrogen density: `nHII = nH - nHI - 2*nH2`
- Temperature:
  `T = pressure*temperature_unit / (nH*(1.1 + xe - xH2))`
- Magnetic strength: `Bmag = |cell_centered_B|` converted to microgauss
- FUV radiation: `rad_energy_density_PE`
- LyC radiation: `rad_energy_density_PH`

Legacy `specific_scalar_*` aliases will remain supported. Non-positive values
will be masked for logarithmic plots. Numerically negative derived ionized
density will be counted and reported rather than silently treated as physical.

The four density views will share exactly one fixed logarithmic normalization,
colormap, tick set, and colorbar position. The default range is
`1e-4`--`1e4 cm^-3`. The field title changes during phase transitions, but the
density scale does not.

## Input Discovery and Time Alignment

The renderer will build a metadata index before drawing frames:

1. Discover the selected `slice/<slice_id>/*.slice.vtk` series.
2. Read stored physical times and sort by time rather than filename.
3. Deduplicate restart-overlap outputs using stored time and file modification
   time, following the projection-series convention.
4. Match star-particle data by output number and verify its stored time.
5. Match the full-volume snapshot at the frozen volume-transition time.
6. Reject matches outside a configurable tolerance and write every match to the
   frame manifest.

Simulation fields will not be interpolated between snapshots. When simulation
and video cadence differ, source frames will be repeated in proportion to their
physical-time spacing.

## Visual Design

The initial target is a 1920 by 1080 canvas with a black background at 30 fps.
The layout will contain:

- A field or scene title in the upper left
- Simulation time in Myr in the upper right
- A coordinate glyph and physical scale bar
- A fixed colorbar region
- A borderless equal-aspect scientific viewport

Transitions will use cosine or smoothstep easing with exact endpoints. Scalar
fields will first be normalized and mapped to RGBA, then visually blended. Raw
physical fields will not be blended, because doing so would manufacture
intermediate gas states and distort logarithmic scales.

Star particles will use the existing mass-scaled and age-colored convention.
The top-down view maps positions to `(x1, x2)` and the optional side-view
overlay maps them to `(x1, x3)`.

## Streamlines and Radiation Composite

At the final XZ time:

- Velocity streamlines use `(v1, v3)` and are colored by temperature.
- Magnetic streamlines use `(B1, B3)` and are colored by `Bmag`.
- The underlying temperature slice is darkened during streamline scenes to
  improve line visibility.

FUV and LyC radiation will be normalized independently and mapped through
distinct colormaps. The default visual pairing will be a warm map for FUV and a
cyan/blue map for LyC. The normalized RGB layers will use a screen/additive
composite, accompanied by two compact color keys. A single scalar colorbar will
not be used for this two-field composite.

## Full-Volume Rendering

A new `pathena.vtk3d_reader` module will read the full-volume snapshot. It will:

- Inspect rank-piece metadata before allocating arrays.
- Validate common spacing, non-overlap, and complete domain coverage.
- Assemble selected fields as `(nz, ny, nx)` arrays, with a final component
  dimension for vector fields.
- Support live `vtk/NNNN/` directories and direct reads from archived snapshots.
- Load only density, pressure, `xH2`, and `xe` for the temperature volume.
- Report the native-resolution memory estimate before allocation.

The target environment did not provide PyVista/VTK, so the implemented backend
uses a deterministic orthographic ray compositor with SciPy interpolation as
the optional `movie3d` dependency. This avoids an EGL, OSMesa, or Xvfb runtime
requirement while retaining an explicitly registered camera path.

The volume will be colored by temperature and use density to control opacity.
This prevents diffuse hot material from making the full domain opaque. The
camera will use parallel projection and a smooth orientation interpolation that
ends exactly registered with the XZ slice.

Native-resolution loading will remain available with `volume_stride = 1`.
Automatic or configured rendering strides may be used when a native grid would
exceed the memory budget, but the resolved stride will always be reported.

## Proposed Code Layout

- `src/tigress_ncr_tools/slice_story_movie.py`
  - Command-line interface
  - Input indexing and time matching
  - Storyboard expansion into frame requests
  - Resume and overwrite behavior
  - Manifest writing and ffmpeg invocation
- `src/tigress_ncr_tools/story_renderers.py`
  - Derived-field preparation
  - Slice, particle, streamline, and radiation renderers
  - Persistent annotations and colorbars
  - RGBA transition functions
- `src/pathena/vtk3d_reader.py`
  - Full-volume metadata and piece assembly
- `examples/slice_story.toml`
  - Reproducible example configuration

Expected output layout:

```text
movie_output/
  frames/frame_000000.png
  frames/frame_000001.png
  cache/
  frame_manifest.csv
  resolved_config.toml
  slice_story.mp4
```

The manifest will record frame number, scene, simulation time, output number,
source paths, blend fraction, field names, and camera state.

## Command-Line Milestones

The first command will provide a slice-only path that works without optional 3D
packages:

```text
slice-story-movie RUN_DIR --problem-id ID --slice-id midplane --config CONFIG
```

Planned operational modes:

- `--preflight`: inventory inputs, matches, fields, dimensions, and memory only
- `--preview`: render a small, low-frame-rate draft
- `--start-frame` and `--stop-frame`: render part of a storyboard
- `--overwrite`: replace existing frames
- `--movie`: run ffmpeg after the frame sequence is complete
- `--no-volume`: omit the optional 3D segment

## Implementation Milestones

### 1. Plan and data model

- Commit this plan.
- Add public, tested derived-slice field helpers.
- Add metadata-only slice-series discovery, physical-time sorting, and restart
  deduplication.

### 2. Storyboard engine

- Define validated configuration and scene/frame data classes.
- Expand time-evolution, hold, and transition scenes deterministically.
- Add easing, source-frame selection, and manifest generation.

### 3. Slice rendering

- Implement fixed-layout top-down and side slice rendering.
- Add direct-plane particle overlays and synchronized labels.
- Implement density-phase and temperature transitions.

### 4. Flow and radiation scenes

- Add temperature-colored velocity streamlines.
- Add magnitude-colored magnetic streamlines.
- Add dual-colormap FUV/LyC compositing and keys.

### 5. Full-volume path

- Implement and test MPI-piece metadata and assembly using a small synthetic
  VTK fixture.
- Add memory preflight and rendering cache.
- Add density-opacity temperature rendering and registered camera rotation.

### 6. Production and polish

- Add resume-safe frame writing and the resolved configuration.
- Reuse the repository's codec detection and ffmpeg quality conventions.
- Render a low-resolution rough cut, adjust the visual timing, and then render
  the final talk-resolution sequence.

Each independently usable milestone will be committed separately. Changes will
remain on the local feature branch for the user to review and push.

## Verification

Tests will cover:

- Derived fields and unit conversions
- Legacy scalar aliases and missing optional fields
- Physical-time sorting, restart deduplication, and particle matching
- Storyboard frame counts and transition endpoint values
- Fixed density normalization across all gas phases
- RGBA blending and radiation screen compositing
- Velocity and magnetic component selection on XZ slices
- Synthetic multi-piece volume assembly
- One small end-to-end frame sequence with a valid manifest

The production acceptance check is a preview movie with no layout jumps,
correct labels and colorbars, exact top/side registration, continuous scene
joins, and a reproducible frame manifest.
