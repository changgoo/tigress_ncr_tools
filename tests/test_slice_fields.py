import numpy as np
import pytest

from pathena.slice_fields import derive_slice_fields
from pathena.units import tigress_units


def raw_fields(legacy=False, radiation=True):
    shape = (2, 3)
    density = np.full(shape, 10.0)
    fields = {
        "density": density,
        "pressure": np.full(shape, 2.0),
        "velocity": np.broadcast_to([1.0, 2.0, 3.0], shape + (3,)).copy(),
        "cell_centered_B": np.broadcast_to(
            [1.0, 2.0, 2.0], shape + (3,)
        ).copy(),
    }
    species = {
        "xHI": np.full(shape, 0.4),
        "xH2": np.full(shape, 0.2),
        "xe": np.full(shape, 0.1),
    }
    if legacy:
        fields.update({
            "specific_scalar_2": species["xHI"],
            "specific_scalar_3": species["xH2"],
            "specific_scalar_4": species["xe"],
        })
    else:
        fields.update(species)
    if radiation:
        fields["rad_energy_density_PE"] = np.full(shape, 4.0)
        fields["rad_energy_density_PH"] = np.full(shape, 5.0)
    return fields


def test_derive_slice_fields_and_vector_components():
    fields = raw_fields()
    result = derive_slice_fields(fields, plane="x3")
    units = tigress_units()

    assert np.allclose(result["nH"], 10.0)
    assert np.allclose(result["nH2"], 2.0)
    assert np.allclose(result["nHI"], 4.0)
    assert np.allclose(result["nHII"], 2.0)
    assert np.allclose(result["P"], 2.0 * units["pressure_over_kB"])
    assert np.allclose(
        result["T"],
        2.0 * units["temperature_per_p_over_rho"] / 10.0,
    )
    assert np.allclose(result["v1"], 1.0)
    assert np.allclose(result["v2"], 2.0)
    assert np.allclose(result["v3"], 3.0)
    assert np.allclose(result["vz"], result["v3"])
    assert np.allclose(
        result["Bmag"], 3.0 * units["magnetic_field_microgauss"]
    )
    assert np.allclose(
        result["Erad_PE"], 4.0 * units["energy_density"]
    )
    assert np.allclose(
        result["Erad_PH"], 5.0 * units["energy_density"]
    )


def test_derive_slice_fields_accepts_legacy_species_and_missing_radiation():
    with pytest.warns(RuntimeWarning, match="rad_energy_density") as warnings:
        result = derive_slice_fields(raw_fields(legacy=True, radiation=False))
    assert len(warnings) == 2
    assert np.allclose(result["nHII"], 2.0)
    assert np.isnan(result["Erad_PE"]).all()
    assert np.isnan(result["Erad_PH"]).all()


def test_derive_slice_fields_reports_missing_required_field():
    fields = raw_fields()
    del fields["pressure"]
    with pytest.raises(KeyError, match="pressure"):
        derive_slice_fields(fields, plane="x2")
