import numpy as np
import pytest

from tigress_ncr_tools.plot_suite_phase_evolution import (
    PHASES,
    PHASE_PARAMETER_SPECS,
    PLOTTED_PHASE_SUMMARY_FIELDS,
    PROFILE_FIELDS,
    phase_profile_moments,
    phase_parameter_correlations,
    reduce_phase_zprof_snapshot,
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
