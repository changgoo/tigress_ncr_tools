from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tigress_ncr_tools.plot_suite_phase_evolution as phase_module
from tigress_ncr_tools.plot_suite_phase_evolution import (
    FRACTION_FIELDS,
    PHASES,
    PHASE_CORRELATION_FAMILIES,
    PHASE_PARAMETER_SPECS,
    PLOTTED_PHASE_SUMMARY_FIELDS,
    PROFILE_FIELDS,
    aggregate_reduced_phase_time_series,
    phase_profile_moments,
    phase_parameter_correlations,
    reduced_phase_fraction_summary,
    reduce_phase_zprof_snapshot,
    reference_axis_limits,
    slab_overlap_weights,
)


def _write_zprof(path, time, fields, rows):
    header = f"# Athena vertical profile at t={time}\n"
    header += ",".join(fields) + "\n"
    body = "\n".join(",".join(str(value) for value in row) for row in rows)
    path.write_text(header + body + "\n")


def test_slab_overlap_weights_include_partial_boundary_cells():
    weights = slab_overlap_weights(np.asarray([-1.0, 0.0, 1.0]), 0.75)
    np.testing.assert_allclose(weights, [0.25, 1.0, 0.25])


def test_phase_profile_moments_subtract_bulk_flow_and_split_alfven_energy():
    mass = np.asarray([2.0, 2.0])
    values = {
        "d": mass,
        "A": np.asarray([3.0, 3.0]),
        "M1": np.asarray([2.0, -2.0]),
        "dM2": np.zeros(2),
        "M3": np.zeros(2),
        "Ek1": np.asarray([4.0, 4.0]),
        "dEk2": np.asarray([9.0, 9.0]),
        "Ek3": np.asarray([16.0, 16.0]),
        "P": np.asarray([6.0, 6.0]),
    }
    for component, mean_speed, perturbed_speed in (
        ("1", 1.0, 2.0),
        ("2", 2.0, 0.0),
        ("3", 2.0, 1.0),
    ):
        mean_cell_energy = 0.25 * 4.0 * mean_speed**2
        perturbed_cell_energy = 0.25 * 4.0 * perturbed_speed**2
        values[f"dPB{component}"] = np.full(2, perturbed_cell_energy)
        values[f"PB{component}"] = np.full(2, mean_cell_energy + perturbed_cell_energy)
    mean_field = {
        "B1": np.full(2, 1.0),
        "B2": np.full(2, 2.0),
        "B3": np.full(2, 2.0),
    }
    values["A"] = mass.copy()
    result = phase_profile_moments(values, np.ones(2), mean_field)
    assert result["sigma_x1"] == pytest.approx(2.0)
    assert result["sigma_x2"] == pytest.approx(3.0)
    assert result["sigma_x3"] == pytest.approx(4.0)
    assert result["sigma_3d"] == pytest.approx(np.sqrt(29.0))
    assert result["sigma_eff_z"] == pytest.approx(np.sqrt(21.0))
    assert result["alfven_mean_3d"] == pytest.approx(3.0)
    assert result["alfven_perturbed_3d"] == pytest.approx(np.sqrt(5.0))


def _phase_rows(z, fraction):
    rows = []
    for height in z:
        density = 100.0 * fraction
        values = {name: 0.0 for name in PROFILE_FIELDS}
        values.update(
            {
                "z": height,
                "A": 100.0 * fraction,
                "d": density,
                "Ek1": 0.5 * density,
                "dEk2": 2.0 * density,
                "Ek3": 2.0 * density,
            }
        )
        for component in ("1", "2", "3"):
            values[f"dPB{component}"] = 0.125 * density
            values[f"PB{component}"] = 0.625 * density
        rows.append(tuple(values[name] for name in PROFILE_FIELDS))
    return rows


def test_reduce_phase_snapshot_uses_six_phase_mapping_and_preserves_uim_residual(
    tmp_path,
):
    z = (-1.0, 0.0, 1.0)
    fractions = {
        7: 0.08,
        11: 0.12,
        12: 0.10,
        13: 0.30,
        9: 0.10,
        10: 0.10,
        5: 0.10,
        6: 0.05,
    }
    indexed = {}
    for index, fraction in fractions.items():
        path = tmp_path / f"model.0001.phase{index}.zprof"
        _write_zprof(path, 250.0, PROFILE_FIELDS, _phase_rows(z, fraction))
        indexed[index] = path
    grouped = {
        phase.key: [indexed[index] for index in phase.indices] for phase in PHASES
    }
    whole = tmp_path / "model.0001.whole.zprof"
    _write_zprof(
        whole,
        250.0,
        ("z", "A", "d", "B1", "B2", "B3"),
        [(height, 100.0, 100.0, 100.0, 100.0, 100.0) for height in z],
    )
    rows, closure = reduce_phase_zprof_snapshot(grouped, whole, 100.0)
    by_phase = {row["phase"]: row for row in rows}
    assert len(rows) == 6
    assert by_phase["cold"]["mass_fraction_box"] == pytest.approx(0.20)
    assert by_phase["wim"]["volume_fraction_hgas"] == pytest.approx(0.20)
    assert by_phase["unm"]["sigma_3d_box"] == pytest.approx(3.0)
    assert by_phase["unm"]["sigma_eff_z_box"] == pytest.approx(np.sqrt(4.625))
    assert by_phase["unm"]["alfven_mean_3d_box"] == pytest.approx(np.sqrt(3.0))
    assert by_phase["unm"]["alfven_perturbed_3d_box"] == pytest.approx(np.sqrt(0.75))
    assert closure["selected_mass_fraction_box"] == pytest.approx(0.95)
    assert closure["uim_residual_mass_fraction_box"] == pytest.approx(0.05)
    assert closure["uim_residual_volume_fraction_hgas"] == pytest.approx(0.05)


def test_phase_parameter_correlations_use_model_medians():
    rows = []
    for phase_index, phase in enumerate(PHASES, start=1):
        for model_index in range(1, 5):
            row = {
                "model": f"model{model_index}",
                "phase": phase.key,
                "phase_label": phase.label,
            }
            for parameter, _ in PHASE_PARAMETER_SPECS:
                row[parameter] = float(model_index)
            for field in PLOTTED_PHASE_SUMMARY_FIELDS:
                row[f"{field}_time_median"] = float(
                    phase_index * model_index
                )
            rows.append(row)
    correlations = phase_parameter_correlations(
        __import__("pandas").DataFrame(rows)
    )
    assert len(correlations) == (
        len(PHASES)
        * len(PLOTTED_PHASE_SUMMARY_FIELDS)
        * len(PHASE_PARAMETER_SPECS)
    )
    np.testing.assert_allclose(correlations["spearman_rho"], 1.0)
    assert np.all(correlations["model_count"] == 4)


def test_correlation_families_cover_every_plotted_quantity_once():
    fields = [
        field
        for _, family_fields, _ in PHASE_CORRELATION_FAMILIES
        for field in family_fields
    ]
    assert len(fields) == len(set(fields))
    assert set(fields) == set(PLOTTED_PHASE_SUMMARY_FIELDS)


def test_reference_axis_limits_exclude_early_anomaly():
    class Axis:
        limits = None

        def set_ylim(self, limits):
            self.limits = limits

    axis = Axis()
    data = pd.DataFrame(
        {"time": [0.0, 200.0, 400.0, 600.0], "value": [1.0e9, 1.0, 2.0, 4.0]}
    )
    phase_module._set_reference_ylim(
        axis, data, ("value",), (200.0, 600.0), log=True
    )
    limits = axis.limits
    assert limits[0] < 1.0
    assert limits[1] > 4.0
    assert limits[1] < 10.0
    assert reference_axis_limits([np.nan, -1.0, 0.0], log=True) is None


def test_parameter_relation_limits_ignore_errorbar_artists():
    models = [Path(f"model{index}") for index in range(1, 5)]
    summary = pd.DataFrame(
        {
            "model": [model.name for model in models],
            "phase": "cold",
            "stellar_scale_height": [200.0, 300.0, 500.0, 800.0],
            "mean_sfr10": [1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3],
            "sigma_3d_box_time_median": [10.0, 20.0, 30.0, 40.0],
            "sigma_3d_box_time_percentile16": [8.0, 18.0, 28.0, 38.0],
            "sigma_3d_box_time_percentile84": [12.0, 22.0, 32.0, 42.0],
        }
    )
    ranked = [(model, float(index) * 1.0e-3) for index, model in enumerate(models, 1)]
    cmap, norm = phase_module.sfr_colormap(
        summary["mean_sfr10"], phase_module.DEFAULT_CMAP, "log"
    )
    figure, axis = phase_module.plt.subplots()
    phase_module._plot_parameter_relation_series(
        axis,
        summary,
        ranked,
        "cold",
        "sigma_3d_box",
        "stellar_scale_height",
        "log",
        cmap,
        norm,
    )
    axis.set_xscale("log")
    np.testing.assert_allclose(
        axis.get_xlim(),
        phase_module.parameter_axis_limits(summary["stellar_scale_height"], "log"),
    )
    phase_module.plt.close(figure)


def test_reduced_phase_fraction_summary_sums_without_renormalizing(monkeypatch):
    fractions = {
        "cold": 0.10,
        "unm": 0.10,
        "wnm": 0.20,
        "wim": 0.10,
        "whim": 0.10,
        "him": 0.10,
    }
    rows = []
    for time, factor in ((400.0, 1.0), (500.0, 1.1)):
        for phase, fraction in fractions.items():
            row = {
                "model": "model1",
                "phase": phase,
                "time": time,
            }
            for field in FRACTION_FIELDS:
                row[field] = fraction * factor
            rows.append(row)
    monkeypatch.setattr(
        phase_module,
        "model_history_parameters",
        lambda model: {
            "stellar_surface_density": 40.0,
            "stellar_scale_height": 200.0,
            "stellar_midplane_density": 0.1,
            "omega": 0.03,
            "qshear": 1.0,
        },
    )
    summary = reduced_phase_fraction_summary(
        pd.DataFrame(rows), [(Path("model1"), 1.0e-3)], bounds=(400.0, 600.0)
    ).set_index("phase")

    assert summary.loc["neutral", "component_phases"] == "cold+unm+wnm"
    assert summary.loc["ionized", "component_phases"] == "wim+whim+him"
    assert summary.loc[
        "neutral", "mass_fraction_box_time_median"
    ] == pytest.approx(0.42)
    assert summary.loc[
        "ionized", "mass_fraction_box_time_median"
    ] == pytest.approx(0.315)
    assert (
        summary.loc["neutral", "mass_fraction_box_time_median"]
        + summary.loc["ionized", "mass_fraction_box_time_median"]
        < 1.0
    )


def test_aggregate_reduced_phase_time_series_recombines_velocity_moments():
    rows = []
    for index, phase in enumerate(PHASES):
        row = {
            "model": "model1",
            "dump": 1,
            "time": 400.0,
            "phase": phase.key,
            "mean_sfr10": 1.0e-3,
            "gas_scale_height_pc": 100.0,
            "mass_scale_height_pc": 10.0 + index,
            "volume_scale_height_pc": 20.0 + index,
        }
        for field in FRACTION_FIELDS:
            row[field] = 0.1
        for region in ("box", "hgas"):
            row[f"mass_code_{region}"] = 1.0
            row[f"volume_pc3_{region}"] = 2.0
            for component in ("x1", "x2", "x3"):
                row[f"mean_velocity_{component}_{region}"] = (
                    float(index) if component == "x1" else 0.0
                )
                row[f"sigma_{component}_{region}"] = 1.0
                row[f"alfven_mean_{component}_{region}"] = 2.0
                row[f"alfven_perturbed_{component}_{region}"] = 3.0
            row[f"sigma_eff_z_{region}"] = 4.0 + index
        rows.append(row)

    combined = aggregate_reduced_phase_time_series(pd.DataFrame(rows))
    neutral = combined.set_index("phase").loc["neutral"]
    assert len(combined) == 2
    assert neutral["mass_fraction_box"] == pytest.approx(0.3)
    assert neutral["mass_code_box"] == pytest.approx(3.0)
    assert neutral["mean_velocity_x1_box"] == pytest.approx(1.0)
    assert neutral["sigma_x1_box"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert neutral["sigma_3d_box"] == pytest.approx(np.sqrt(11.0 / 3.0))
    assert neutral["sigma_eff_z_box"] == pytest.approx(np.sqrt(77.0 / 3.0))
    assert neutral["alfven_mean_3d_box"] == pytest.approx(np.sqrt(12.0))
