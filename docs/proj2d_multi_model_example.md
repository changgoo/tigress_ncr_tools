# Reading `proj2d` output from multiple TIGRESS-NCR models

This example uses `pathena.proj2d_reader` to read models under
`/projects/c/changgoo/nasa_athena/TIGRESS-NCR/`.

The suite has three physical box-size models. Each is split between an
`early` and a `late` run:

```text
R8_8pc_NCR_Lxy1024_early    R8_8pc_NCR_Lxy1024_late
R8_8pc_NCR_Lxy2048_early    R8_8pc_NCR_Lxy2048_late
R8_8pc_NCR_Lxy4096_early    R8_8pc_NCR_Lxy4096_late
```

Each run has `theta0`, `theta15`, `theta30`, `theta45`, and `theta60`
viewing-angle directories under `proj2d/`. These examples discover available
files rather than assuming a fixed last output number.

## Setup

Install this repository into the Python environment used for analysis:

```bash
python -m pip install -e /home/changgoo/tigress_ncr_tools
```

Then define the suite and model segments:

```python
from pathlib import Path

SUITE = Path("/projects/c/changgoo/nasa_athena/TIGRESS-NCR")
PROBLEM_ID = "R8_8pc_NCR"

MODELS = {
    "Lxy1024": [
        SUITE / "R8_8pc_NCR_Lxy1024_early",
        SUITE / "R8_8pc_NCR_Lxy1024_late",
    ],
    "Lxy2048": [
        SUITE / "R8_8pc_NCR_Lxy2048_early",
        SUITE / "R8_8pc_NCR_Lxy2048_late",
    ],
    "Lxy4096": [
        SUITE / "R8_8pc_NCR_Lxy4096_early",
        SUITE / "R8_8pc_NCR_Lxy4096_late",
    ],
}


def output_number(path):
    """Return 123 for R8_8pc_NCR.0123.theta30.proj2d."""
    return int(path.name.split(".")[1])


def projection_paths(segment_dirs, projection_id):
    """Join early/late files, ordered by Athena output number."""
    by_number = {}
    for run_dir in segment_dirs:
        directory = run_dir / "proj2d" / projection_id
        pattern = f"{PROBLEM_ID}.????.{projection_id}.proj2d"
        for path in directory.glob(pattern):
            by_number[output_number(path)] = path
    return [by_number[number] for number in sorted(by_number)]
```

The dictionary keyed by output number makes the join robust to a duplicate
file at an early/late boundary.

## Read one file

Read the newest available `theta30` output for `Lxy1024`:

```python
from pathena.proj2d_reader import print_metadata, read_proj2d

paths = projection_paths(MODELS["Lxy1024"], "theta30")
if not paths:
    raise FileNotFoundError("No theta30 proj2d files found for Lxy1024")

proj = read_proj2d(paths[-1])
print_metadata(proj)

print("source:", proj["path"])
print("output number:", proj["num"])
print("simulation time:", proj["time"])
print("view:", proj["theta"], proj["phi"])
print("available fields:", proj["field_names"])

nH_integral = proj["fields"]["nH"]
print("nH array shape:", nH_integral.shape)
print("x edges:", proj["x_edges"].shape)
print("y edges:", proj["y_edges"].shape)
```

Each entry in `proj["fields"]` is a NumPy array with shape
`(proj["Nbiny"], proj["Nbinx"])`. The edge arrays can be passed directly to
`matplotlib.pyplot.pcolormesh`.

For an `nH`-like field, the writer stores the line integral in `cm^-3 pc`.
Convert it to column density in `cm^-2` with `PC_CGS`:

```python
from pathena.units import PC_CGS

NH = proj["fields"]["nH"] * PC_CGS
```

## Compare the same output across models

Do not use each model's last file for a like-for-like comparison because the
models may have progressed to different output numbers. This finds the newest
output common to all three models, reads only those files, and reports their
stored simulation times:

```python
from pathena.proj2d_reader import read_proj2d

projection_id = "theta30"
paths_by_model = {
    model: projection_paths(segments, projection_id)
    for model, segments in MODELS.items()
}

missing = [model for model, paths in paths_by_model.items() if not paths]
if missing:
    raise FileNotFoundError(
        f"No {projection_id} proj2d files for: {', '.join(missing)}"
    )

numbers_by_model = {
    model: {output_number(path): path for path in paths}
    for model, paths in paths_by_model.items()
}
common_numbers = set.intersection(
    *(set(files) for files in numbers_by_model.values())
)
if not common_numbers:
    raise RuntimeError("The models have no common output number")

number = max(common_numbers)
frames = {
    model: read_proj2d(files[number])
    for model, files in numbers_by_model.items()
}

for model, frame in frames.items():
    print(
        f"{model}: num={frame['num']}, time={frame['time']:g}, "
        f"shape={frame['fields']['nH'].shape}"
    )
```

The `time` metadata is authoritative. Check the printed times before treating
an output-number comparison as a same-time comparison.

Plot column density with one color scale:

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pathena.units import PC_CGS

column_densities = {
    model: frame["fields"]["nH"] * PC_CGS
    for model, frame in frames.items()
}
positive = np.concatenate([
    data[np.isfinite(data) & (data > 0)]
    for data in column_densities.values()
])
norm = LogNorm(vmin=positive.min(), vmax=positive.max())

fig, axes = plt.subplots(
    1, len(frames), figsize=(14, 4), constrained_layout=True
)
image = None
for ax, (model, frame) in zip(axes, frames.items()):
    image = ax.pcolormesh(
        frame["x_edges"],
        frame["y_edges"],
        column_densities[model],
        cmap="magma",
        norm=norm,
    )
    ax.set(
        title=f"{model}: {projection_id}, output {number:04d}",
        xlabel="image x [pc]",
        ylabel="image y [pc]",
        aspect="equal",
    )

fig.colorbar(image, ax=axes, label=r"$N_{\rm H}\;[\mathrm{cm}^{-2}]$")
fig.savefig(
    f"proj2d_{projection_id}_{number:04d}_model_comparison.png",
    dpi=200,
)
```

## Compare all viewing angles for one model

`plot_field_all_projections` expects a mapping from projection ID to a list of
frames. Supply a one-element list for the latest output shared by all angles
so only five files remain in memory:

```python
from pathena.proj2d_reader import plot_field_all_projections, read_proj2d
from pathena.units import PC_CGS

segments = MODELS["Lxy2048"]
projection_ids = ["theta0", "theta15", "theta30", "theta45", "theta60"]
angle_files = {
    projection_id: {
        output_number(path): path
        for path in projection_paths(segments, projection_id)
    }
    for projection_id in projection_ids
}

common_angle_numbers = set.intersection(
    *(set(files) for files in angle_files.values())
)
if not common_angle_numbers:
    raise RuntimeError("The viewing angles have no common output number")

number = max(common_angle_numbers)
views = {
    projection_id: [read_proj2d(files[number])]
    for projection_id, files in angle_files.items()
}

fig, axes = plot_field_all_projections(
    views,
    "nH",
    frame=0,
    ncol=5,
    norm="log",
    cmap="magma",
    share_norm=True,
    data_scale=PC_CGS,
    field_label=r"$N_{\rm H}\;[\mathrm{cm}^{-2}]$",
)
fig.savefig(
    f"proj2d_Lxy2048_all_angles_{number:04d}.png",
    dpi=200,
)
```

For a quick view of all stored fields or the standard derived maps:

```python
from pathena.proj2d_reader import plot_all, plot_physical_maps

frame = views["theta30"][0]

all_fields_figure = plot_all(frame)
all_fields_figure.savefig("proj2d_all_fields.png", dpi=150)

physical_figure, axes = plot_physical_maps(frame)
physical_figure.savefig("proj2d_physical_maps.png", dpi=150)
```

## Process a time series without retaining every frame

`read_proj2d_series` and `read_all_proj2ds` eagerly load every matching field
array. Stream over paths when only a reduced time series is needed:

```python
import numpy as np
from pathena.proj2d_reader import read_proj2d

times = []
mean_nH_integrals = []

for path in projection_paths(MODELS["Lxy4096"], "theta30"):
    frame = read_proj2d(path)
    times.append(frame["time"])
    mean_nH_integrals.append(np.nanmean(frame["fields"]["nH"]))
    del frame

times = np.asarray(times)
mean_nH_integrals = np.asarray(mean_nH_integrals)
```

Use the eager series reader only when retaining every map is intended:

```python
from pathena.proj2d_reader import read_proj2d_series

run = MODELS["Lxy1024"][0]
pattern = str(
    run / "proj2d" / "theta30"
    / f"{PROBLEM_ID}.????.theta30.proj2d"
)
early_theta30_frames = read_proj2d_series(pattern)
```

## Common pitfalls

- Treat `early` and `late` as segments of one box-size model and order their
  files by four-digit output number.
- Use `frame["time"]` when physical time alignment matters.
- Access arrays as `frame["fields"][name]`; fields are not top-level entries.
- Use `x_edges` and `y_edges` to preserve physical image coordinates.
- Convert `nH`-like line integrals with `PC_CGS` before labeling them as
  column densities. Weighted ratios such as
  `fields["nH*Vlos"] / fields["nH"]` need no length conversion.
- Avoid `read_all_proj2ds` for the full suite unless sufficient memory is
  available; it loads every angle and output eagerly.
