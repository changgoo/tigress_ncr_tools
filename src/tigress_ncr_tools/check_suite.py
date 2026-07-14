#!/usr/bin/env python3
"""Print the current state of an Athena model suite."""

import argparse
import re
import subprocess
import time as time_module
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}
PROBLEM = {"FAILED", "CRASHED", "STALLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
FAILURE = re.compile(
    r"### Fatal error:.*|Fatal error in .*|.*MPI_Abort.*|.*OFI poll failed.*|"
    r".*segmentation fault.*|.*out of memory.*",
    re.IGNORECASE,
)


def failure_reason(model_dir, since=None):
    """Return the most useful strong failure marker from application logs."""
    matches = []
    for path in sorted(Path(model_dir).glob("err*.txt"), key=lambda p: p.stat().st_mtime, reverse=True):
        if since is not None and path.stat().st_mtime < since:
            continue
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            lines = stream.read().decode(errors="replace").splitlines()
        for line in lines:
            if FAILURE.search(line):
                matches.append(line.strip())
        if matches:
            break
    if not matches:
        return ""
    return next((line for line in matches if "Fatal error" in line), matches[-1])[:140]


def parse_start(value):
    """Convert Slurm's local ISO timestamp to epoch seconds when available."""
    if not value or value in {"None", "Unknown", "N/A"}:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def parse_sacct(text):
    """Keep the newest top-level accounting record for each job name."""
    jobs = {}
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) >= 5 and re.fullmatch(r"\d+", fields[0]):
            jobs[fields[1]] = dict(
                jobid=fields[0], state=fields[2].split()[0], elapsed=fields[4],
                start=parse_start(fields[5]) if len(fields) > 5 else None,
            )
    return jobs


def slurm_jobs():
    try:
        result = subprocess.run(
            ["sacct", "--starttime", "now-14days", "-X", "-n", "-P",
             "--format=JobIDRaw,JobName%40,State,ExitCode,Elapsed,Start"],
            check=True, text=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    return parse_sacct(result.stdout)


def history_time(model_dir):
    files = sorted((Path(model_dir) / "hst").glob("*.hst"))
    files = [path for path in files if ".phase" not in path.name and ".whole" not in path.name]
    if not files:
        return None
    try:
        with files[0].open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            lines = stream.read().decode(errors="replace").splitlines()
        for line in reversed(lines):
            if line.strip() and not line.lstrip().startswith("#"):
                return float(line.split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return None


def progress_mtime(model_dir):
    """Return the newest timestamp from files written during time stepping."""
    model_dir = Path(model_dir)
    paths = list(model_dir.glob("out*.txt"))
    paths.extend(model_dir.glob("timeit*.txt"))
    paths.extend(
        path for path in (model_dir / "hst").glob("*.hst")
        if ".phase" not in path.name and ".whole" not in path.name
    )
    return max((path.stat().st_mtime for path in paths), default=None)


def application_exit_code(model_dir, since=None):
    """Return Athena's exit code from the newest current-attempt Slurm log."""
    paths = sorted(Path(model_dir).glob("ncr-*.out"),
                   key=lambda path: path.stat().st_mtime, reverse=True)
    for path in paths:
        if since is not None and path.stat().st_mtime < since:
            continue
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - 8192))
                text = stream.read().decode(errors="replace")
        except OSError:
            continue
        matches = re.findall(r"^EXITCODE\s*=\s*(\d+)\s*$", text, re.MULTILINE)
        if matches:
            return int(matches[-1])
    return None


def model_status(model_dir, job, reason=None, progress=None, now=None,
                 stale_seconds=7200.0, app_exit=None):
    since = job.get("start") if job else None
    reason = failure_reason(model_dir, since=since) if reason is None else reason
    if job and job["state"] in {"PENDING", "CONFIGURING", "REQUEUED"}:
        return job["state"], ""
    if job and job["state"] in ACTIVE:
        if reason:
            return "CRASHED", reason
        progress = progress_mtime(model_dir) if progress is None else progress
        now = time_module.time() if now is None else now
        references = [value for value in (progress, since) if value is not None]
        if not references:
            return job["state"], "no progress file yet"
        reference = max(references)
        idle = now - reference
        if stale_seconds > 0.0 and idle > stale_seconds:
            return "STALLED", f"no progress output for {idle / 3600.0:.1f}h"
        return job["state"], ""
    # Athena can finish successfully while summary plotting or movie creation
    # makes the enclosing batch job fail. Treat the simulation as complete.
    if app_exit == 0:
        return "COMPLETE", ""
    if reason:
        return "FAILED", reason
    if job:
        state = job["state"]
        return ("COMPLETE" if state == "COMPLETED" else state), ""
    return "UNKNOWN", "no Slurm record; file-only fallback"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", default="/anvil/scratch/x-ckim5/TIGRESS-NCR")
    parser.add_argument(
        "--stale-hours", type=float, default=2.0,
        help="mark an active job STALLED after this many hours without output (default: 2)",
    )
    args = parser.parse_args(argv)
    models = sorted(Path(args.suite).glob("R8_8pc_NCR_row????"))
    jobs = slurm_jobs()
    counts = Counter()
    def inspect(model):
        job = jobs.get(model.name)
        since = job.get("start") if job else None
        return (history_time(model), failure_reason(model, since),
                progress_mtime(model), application_exit_code(model, since))

    with ThreadPoolExecutor(max_workers=8) as pool:
        checks = dict(zip(models, pool.map(inspect, models)))

    print(f"{'MODEL':28} {'JOBID':>9} {'STATUS':>10} {'ELAPSED':>11} {'T':>8}  REASON")
    for model in models:
        job = jobs.get(model.name)
        time, log_failure, progress, app_exit = checks[model]
        status, reason = model_status(
            model, job, log_failure, progress=progress,
            stale_seconds=args.stale_hours * 3600.0,
            app_exit=app_exit,
        )
        counts[status] += 1
        print(f"{model.name:28} {(job or {}).get('jobid', '-'):>9} {status:>10} "
              f"{(job or {}).get('elapsed', '-'):>11} {time if time is not None else '-':>8}  {reason}")

    print("\n" + "  ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    return 1 if any(counts[state] for state in PROBLEM) else 0


if __name__ == "__main__":
    raise SystemExit(main())
