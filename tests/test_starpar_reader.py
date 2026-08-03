from pathlib import Path

import pytest

from pathena.starpar_reader import index_starpar_series, match_starpar_time


def write_metadata_only(path, time):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"# vtk DataFile Version 2.0\n"
        + f"PRIMITIVE vars at time= {time:.8e}\n".encode()
    )


def test_particle_series_matches_half_cadence_by_physical_time(tmp_path):
    directory = tmp_path / "starpar"
    write_metadata_only(directory / "R8.0201.starpar.vtk", 201.0)
    write_metadata_only(directory / "R8.0202.starpar.vtk", 202.0)
    series = index_starpar_series(tmp_path, "R8")
    assert [(record["num"], record["time"]) for record in series] == [
        ("0201", 201.0), ("0202", 202.0)
    ]
    assert match_starpar_time(series, 201.5)["num"] == "0201"
    with pytest.raises(FileNotFoundError, match="closest is 0201"):
        match_starpar_time(series, 200.0)


def test_particle_series_can_be_absent_and_tolerance_is_validated():
    assert match_starpar_time([], 1.0) is None
    records = [{"path": str(Path("particle.vtk")), "num": "0001", "time": 1.0}]
    with pytest.raises(ValueError, match="non-negative"):
        match_starpar_time(records, 1.0, time_tolerance=-1.0)
