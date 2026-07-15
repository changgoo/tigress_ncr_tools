"""Small, dependency-light reader for Athena history files."""

import re

import numpy as np


_COLUMN = re.compile(r"\[(\d+)]\s*=\s*(.*?)(?=\s+\[\d+]\s*=|$)")


def restart_survivor_indices(time):
    """Return rows on the latest branch of an append-only history file.

    Athena appends to an existing history after a restart.  When the restart
    time precedes the last dump, the old rows at and beyond that time describe
    an abandoned branch.  A stack removes that branch while preserving file
    order and the later value at a duplicate timestamp.
    """
    time = np.asarray(time, dtype=float)
    survivors = []
    for index, value in enumerate(time):
        if not np.isfinite(value):
            continue
        while survivors and time[survivors[-1]] >= value:
            survivors.pop()
        survivors.append(index)
    return np.asarray(survivors, dtype=int)


def strip_restart_branches(history, time_key="time"):
    """Return a history dictionary with abandoned restart branches removed."""
    if time_key not in history:
        return history
    time = np.asarray(history[time_key])
    keep = restart_survivor_indices(time)
    if len(keep) == len(time):
        return history
    return {
        name: value[keep]
        if isinstance(value, np.ndarray) and value.ndim and len(value) == len(time)
        else value
        for name, value in history.items()
    }


def read_hst(filename, max_rows=None, prune_restarts=True):
    """Return history columns, skipping partial rows and abandoned branches."""
    first_line = ""
    header = None
    raw = np.array([])
    with open(filename, "rb") as stream:
        while True:
            position = stream.tell()
            line = stream.readline()
            if not line:
                break
            text = line.decode(errors="replace")
            if not first_line:
                first_line = text
            if text.startswith("#") and "[1]" in text:
                header = text
            if not text.lstrip().startswith("#"):
                if max_rows:
                    stream.seek(0, 2)
                    end = stream.tell()
                    ncols = len(list(_COLUMN.finditer(header or "")))
                    rows = []
                    for offset in np.linspace(position, max(position, end - 1), max_rows, dtype=int):
                        stream.seek(offset)
                        if offset != position:
                            stream.readline()
                        row = np.fromstring(stream.readline().decode(errors="replace"), sep=" ")
                        if len(row) == ncols:
                            rows.append(row)
                    stream.seek(max(position, end - 65536))
                    for tail in reversed(stream.read().splitlines()):
                        row = np.fromstring(tail.decode(errors="replace"), sep=" ")
                        if len(row) == ncols:
                            rows.append(row)
                            break
                    raw = np.asarray(rows).ravel()
                else:
                    stream.seek(position)
                    raw = np.fromfile(stream, sep=" ")
                break
    if header is None:
        raise ValueError(f"no indexed history header in {filename}")

    names = [re.sub(r"\W", "", match.group(2)) for match in _COLUMN.finditer(header)]
    if not names:
        raise ValueError(f"no history columns in {filename}")

    # Athena only leaves an incomplete row at the end while actively writing.
    values = raw[: len(raw) - len(raw) % len(names)].reshape(-1, len(names))
    result = {name: values[:, i] if len(values) else np.array([]) for i, name in enumerate(names)}
    volume = re.search(r"volume=([+\-\d.eE]+)", first_line)
    if volume:
        result["vol"] = float(volume.group(1))
    if prune_restarts:
        result = strip_restart_branches(result)
    return result
