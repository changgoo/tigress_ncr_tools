from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from tigress_ncr_tools.plot_suite_hst_evolution import sfr_colormap
from tigress_ncr_tools.plot_suite_prfm import (
    PARAMETER_COLOR_SPECS,
    PRESSURE_COMPONENTS,
    YIELD_KMS_PER_POK_SFR,
    ZPROF_PRESSURE_OVER_KB,
    plot_prfm_balance,
    plot_prfm_components,
    plot_prfm_vertical_profiles,
    reduce_zprof_snapshot,
    summarize_prfm,
)


def _write_zprof(path, time, fields, rows):
    header = "# Athena vertical profile at t={}\n".format(time)
    header += ",".join(fields) + "\n"
    body = "\n".join(",".join(str(value) for value in row) for row in rows)
    path.write_text(header + body + "\n")


def test_reduce_zprof_snapshot_matches_reference_stress_and_weight_definitions(
    tmp_path,
):
    z = (-25.0, -15.0, -5.0, 5.0, 15.0, 25.0)
    pressure_fields = (
        "z",
        "A",
        "Ek3",
        "d",
        "P",
        "PB1",
        "PB2",
        "PB3",
        "dPB1",
        "dPB2",
        "dPB3",
    )
    phase_paths = []
    for phase in (7, 11, 12, 13):
        path = tmp_path / f"model.0001.phase{phase}.zprof"
        rows = [
            (
                height,
                1.0,
                3.0,
                4.0,
                1.0 if abs(height) == 15.0 else 2.0,
                10.0,
                8.0,
                2.0,
                4.0,
                3.0,
                1.0,
            )
            for height in z
        ]
        _write_zprof(path, 250.0, pressure_fields, rows)
        phase_paths.append(path)

    whole = tmp_path / "model.0001.whole.zprof"
    whole_rows = [
        (
            height,
            20.0,
            5.0,
            6.0,
            10.0,
            8.0,
            2.0,
            4.0,
            3.0,
            1.0,
            -1.0 if height < 0.0 else 1.0,
            -0.5 if height < 0.0 else 0.5,
        )
        for height in z
    ]
    whole_fields = (
        "z",
        "d",
        "Ek3",
        "P",
        "PB1",
        "PB2",
        "PB3",
        "dPB1",
        "dPB2",
        "dPB3",
        "dWext",
        "dWsg",
    )
    _write_zprof(whole, 250.0, whole_fields, whole_rows)

    result, profile = reduce_zprof_snapshot(
        phase_paths,
        whole,
        horizontal_area=100.0,
        midplane_half_width=10.0,
        top_half_width=1.0,
        return_profile=True,
    )
    unit = ZPROF_PRESSURE_OVER_KB
    assert result["time"] == 250.0
    assert result["area_two_phase_fraction"] == 0.04
    assert result["pressure_turbulent"] == 6.0 * unit
    assert result["pressure_thermal"] == 2.0 * unit
    assert result["pressure_magnetic_turbulent"] == 6.0 * unit
    assert result["pressure_magnetic_mean"] == 10.0 * unit
    assert result["pressure_total"] == 24.0 * unit
    assert result["weight_external"] == 0.3 * unit
    assert result["weight_self_gravity"] == 0.15 * unit
    np.testing.assert_allclose(result["weight_total"], 0.45 * unit)
    np.testing.assert_allclose(result["pressure_delta_total"], 1.0 * unit)
    np.testing.assert_allclose(profile["density_two_phase"], 0.16)
    np.testing.assert_allclose(profile["pressure_total"].max(), 0.96 * unit)
    np.testing.assert_allclose(profile["total_gas_density"], 0.2)
    np.testing.assert_allclose(profile["total_gas_pressure_total"], 0.32 * unit)


def _time_series():
    rows = []
    for model, scale in (("high", 2.0), ("low", 1.0)):
        for time, factor in ((200.0, 0.8), (400.0, 1.0), (600.0, 1.2)):
            row = {
                "model": model,
                "time": time,
                "sfr10": scale * factor * 1.0e-3,
                "sfr40": scale * factor * 1.2e-3,
                "pressure_turbulent": scale * factor * 1000.0,
                "pressure_thermal": scale * factor * 800.0,
                "pressure_magnetic_turbulent": scale * factor * 500.0,
                "pressure_magnetic_mean": scale * factor * 700.0,
                "weight_external": scale * factor * 2200.0,
                "weight_self_gravity": scale * factor * 800.0,
            }
            row["pressure_total"] = sum(
                row[field] for field, _, _ in PRESSURE_COMPONENTS
            )
            for pressure, _, _ in PRESSURE_COMPONENTS:
                suffix = pressure.removeprefix("pressure_")
                row[f"pressure_top_{suffix}"] = 0.25 * row[pressure]
                row[f"pressure_delta_{suffix}"] = 0.75 * row[pressure]
            row["pressure_top_total"] = 0.25 * row["pressure_total"]
            row["pressure_delta_total"] = 0.75 * row["pressure_total"]
            row["weight_total"] = row["weight_external"] + row["weight_self_gravity"]
            row["pressure_weight_ratio"] = row["pressure_total"] / row["weight_total"]
            row["pressure_delta_weight_ratio"] = (
                row["pressure_delta_total"] / row["weight_total"]
            )
            for pressure, _, _ in PRESSURE_COMPONENTS:
                suffix = pressure.removeprefix("pressure_")
                row[f"yield_{suffix}"] = (
                    row[pressure] / row["sfr40"] * YIELD_KMS_PER_POK_SFR
                )
                row[f"yield_delta_{suffix}"] = (
                    row[f"pressure_delta_{suffix}"]
                    / row["sfr40"]
                    * YIELD_KMS_PER_POK_SFR
                )
            row["yield_total"] = (
                row["pressure_total"] / row["sfr40"] * YIELD_KMS_PER_POK_SFR
            )
            row["yield_delta_total"] = (
                row["pressure_delta_total"] / row["sfr40"] * YIELD_KMS_PER_POK_SFR
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _profile_summary():
    rows = []
    fields = (
        "density_two_phase",
        "pressure_turbulent",
        "pressure_thermal",
        "pressure_magnetic_turbulent",
        "pressure_magnetic_mean",
        "pressure_total",
        "total_gas_density",
        "total_gas_pressure_turbulent",
        "total_gas_pressure_thermal",
        "total_gas_pressure_magnetic_turbulent",
        "total_gas_pressure_magnetic_mean",
        "total_gas_pressure_total",
    )
    for model, scale in (("high", 2.0), ("low", 1.0)):
        for z in (-100.0, 0.0, 100.0):
            row = {"model": model, "z": z}
            for field in fields:
                value = (
                    scale
                    * np.exp(-abs(z) / 100.0)
                    * (1.0 if "density" in field else 1000.0)
                )
                for statistic, factor in (
                    ("mean", 1.0),
                    ("p16", 0.8),
                    ("p50", 1.0),
                    ("p84", 1.2),
                ):
                    row[f"{field}_{statistic}"] = value * factor
            rows.append(row)
    return pd.DataFrame(rows)


def test_summary_and_prfm_figures_include_all_relations(tmp_path):
    ranked = [(Path("high"), 2.0e-3), (Path("low"), 1.0e-3)]
    parameters = {
        "high": {"omega": 0.05, "stellar_midplane_density": 0.1, "qshear": 1.0},
        "low": {"omega": 0.025, "stellar_midplane_density": 0.05, "qshear": 0.5},
    }
    summary = summarize_prfm(_time_series(), ranked, parameters)
    assert summary["model"].tolist() == ["high", "low"]
    assert np.all(summary["samples"] == 3)
    assert summary.loc[0, "pressure_total_mean"] == 6000.0
    assert summary.loc[0, "omega"] == 0.05

    cmap, norm = sfr_colormap([2.0e-3, 1.0e-3], "plasma", "log")
    balance = tmp_path / "balance.png"
    components = tmp_path / "components.png"
    delta = tmp_path / "delta.png"
    vertical = tmp_path / "vertical.png"
    vertical_total_gas = tmp_path / "vertical_total_gas.png"
    plot_prfm_balance(summary, balance, cmap=cmap, norm=norm, dpi=50)
    plot_prfm_components(summary, components, cmap=cmap, norm=norm, dpi=50)
    plot_prfm_balance(
        summary,
        delta,
        cmap=cmap,
        norm=norm,
        pressure_field="pressure_delta_total",
        pressure_symbol=r"\Delta P_{\rm tot,2p}",
        dpi=50,
    )
    plot_prfm_vertical_profiles(
        _profile_summary(), summary, vertical, cmap=cmap, norm=norm, dpi=50
    )
    plot_prfm_vertical_profiles(
        _profile_summary(),
        summary,
        vertical_total_gas,
        cmap=cmap,
        norm=norm,
        gas_selection="total_gas",
        colorbar_label=r"$q$",
        dpi=50,
    )
    assert delta.stat().st_size > 0
    assert vertical.stat().st_size > 0
    assert vertical_total_gas.stat().st_size > 0
    assert balance.stat().st_size > 0
    assert components.stat().st_size > 0


def test_parameter_colormaps_are_distinct():
    assert [field for field, _, _, _ in PARAMETER_COLOR_SPECS] == [
        "omega",
        "stellar_midplane_density",
        "qshear",
    ]
    assert len({cmap for _, _, cmap, _ in PARAMETER_COLOR_SPECS}) == 3
