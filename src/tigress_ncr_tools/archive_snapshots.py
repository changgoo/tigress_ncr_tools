#!/usr/bin/env python3
"""Atomically archive completed TIGRESS VTK/restart snapshot directories."""

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import os
import re
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


SNAPSHOT_DIR = re.compile(r"^\d{4,}$")


@dataclass
class Snapshot:
    path: Path
    kind: str
    number: int
    problem_id: str
    files: list
    state: dict

    @property
    def tar_path(self):
        return self.path.parent / f"{self.problem_id}.{self.number:04d}.tar"


def file_state(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def inspect_snapshot(path, kind, expected_ranks=None):
    """Validate names and return a stable rank-indexed snapshot description."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"not a real snapshot directory: {path}")
    if not SNAPSHOT_DIR.fullmatch(path.name):
        raise ValueError(f"snapshot directory must be numeric: {path}")
    number = int(path.name)
    pattern = re.compile(
        rf"^(?P<problem>.+?)(?:-id(?P<rank>\d+))?\."
        rf"(?P<number>{number:04d})\.{re.escape(kind)}$"
    )
    ranks = {}
    problem_id = None
    state = {}
    for entry in sorted(path.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"snapshot contains a non-regular file: {entry}")
        match = pattern.fullmatch(entry.name)
        if match is None:
            raise ValueError(f"unexpected file in snapshot: {entry}")
        problem = match.group("problem")
        rank = int(match.group("rank") or 0)
        if problem_id is None:
            problem_id = problem
        elif problem != problem_id:
            raise ValueError(f"mixed problem IDs in {path}: {problem_id}, {problem}")
        if rank in ranks:
            raise ValueError(f"duplicate MPI rank {rank} in {path}")
        ranks[rank] = entry
        state[entry.name] = file_state(entry)
    if not ranks:
        raise ValueError(f"empty snapshot directory: {path}")
    contiguous_ranks = set(range(max(ranks) + 1))
    if set(ranks) != contiguous_ranks:
        missing = sorted(contiguous_ranks - set(ranks))
        raise ValueError(f"non-contiguous MPI ranks in {path}; missing {missing[:10]}")
    if expected_ranks is not None and len(ranks) != expected_ranks:
        raise ValueError(
            f"incomplete snapshot {path}: found {len(ranks)} of "
            f"{expected_ranks} expected MPI ranks"
        )
    files = [ranks[rank] for rank in sorted(ranks)]
    return Snapshot(path, kind, number, problem_id, files, state)


def unchanged(snapshot):
    try:
        names = {entry.name for entry in snapshot.path.iterdir()}
        if names != set(snapshot.state):
            return False
        return all(file_state(snapshot.path / name) == value
                   for name, value in snapshot.state.items())
    except OSError:
        return False


def expected_tar_members(snapshot):
    prefix = snapshot.path.name
    return {
        f"{prefix}/{entry.name}": snapshot.state[entry.name][0]
        for entry in snapshot.files
    }


def verify_tar(tar_path, snapshot):
    """Verify archive member names and uncompressed sizes against originals."""
    expected = expected_tar_members(snapshot)
    try:
        with tarfile.open(tar_path, "r") as archive:
            members = {
                member.name: member.size
                for member in archive.getmembers()
                if member.isfile()
            }
    except (OSError, tarfile.TarError):
        return False
    return members == expected


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_verified_tar(snapshot, overwrite=False):
    """Create and atomically install a verified tar, returning (path, created)."""
    tar_path = snapshot.tar_path
    if tar_path.exists() and not overwrite:
        if not verify_tar(tar_path, snapshot):
            raise ValueError(f"existing archive does not match snapshot: {tar_path}")
        return tar_path, False

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{tar_path.name}.", suffix=".tmp", dir=tar_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(temporary, "w") as archive:
            archive.add(snapshot.path, arcname=snapshot.path.name, recursive=False)
            for entry in snapshot.files:
                archive.add(
                    entry,
                    arcname=f"{snapshot.path.name}/{entry.name}",
                    recursive=False,
                )
        if not unchanged(snapshot):
            raise RuntimeError(f"snapshot changed while archiving: {snapshot.path}")
        if not verify_tar(temporary, snapshot):
            raise RuntimeError(f"archive verification failed: {temporary}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, tar_path)
        fsync_directory(tar_path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return tar_path, True


def remove_verified_snapshot(snapshot, tar_path):
    """Remove only the files captured in a matching verified archive."""
    if not unchanged(snapshot):
        raise RuntimeError(f"snapshot changed before removal: {snapshot.path}")
    if not verify_tar(tar_path, snapshot):
        raise RuntimeError(f"refusing removal; archive verification failed: {tar_path}")
    for entry in snapshot.files:
        entry.unlink()
    snapshot.path.rmdir()
    fsync_directory(snapshot.path.parent)


def numeric_snapshot_dirs(kind_dir):
    if not kind_dir.is_dir():
        return []
    return sorted(
        (entry for entry in kind_dir.iterdir()
         if entry.is_dir() and not entry.is_symlink()
         and SNAPSHOT_DIR.fullmatch(entry.name)),
        key=lambda entry: int(entry.name),
    )


def infer_expected_ranks(run_dir):
    """Infer MPI ranks from a copied generated PBS script."""
    values = set()
    for script in Path(run_dir).glob("*.pbs"):
        try:
            with script.open() as stream:
                for line in stream:
                    match = re.match(r"^NPROCS=(\d+)\s*$", line)
                    if match:
                        values.add(int(match.group(1)))
                        break
        except OSError:
            continue
    return values.pop() if len(values) == 1 else None


@contextmanager
def archive_lock(run_dir):
    lock_path = run_dir / ".archive-snapshots.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another archiver holds {lock_path}")
        yield


def newest_mtime(snapshot):
    return max(
        [snapshot.path.stat().st_mtime]
        + [entry.stat().st_mtime for entry in snapshot.files]
    )


def process_run(run_dir, kinds=("vtk", "rst"), keep_latest=1,
                min_age_seconds=1800.0, remove_originals=False,
                overwrite=False, dry_run=False, expected_ranks=None,
                now=None, output=print):
    """Archive eligible snapshots in one simulation directory."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    if keep_latest < 0:
        raise ValueError("keep_latest must be non-negative")
    now = time.time() if now is None else now
    stats = dict(archived=0, reused=0, removed=0, skipped=0, errors=0,
                 objects_freed=0)
    expected_ranks = expected_ranks or infer_expected_ranks(run_dir)
    if remove_originals and expected_ranks is None:
        raise ValueError(
            "cannot establish the expected MPI rank count from a copied PBS "
            "script; pass --expected-ranks before allowing removal"
        )

    lock_context = nullcontext() if dry_run else archive_lock(run_dir)
    with lock_context:
        for kind in kinds:
            directories = numeric_snapshot_dirs(run_dir / kind)
            protected = set(directories[-keep_latest:]) if keep_latest else set()
            for path in directories:
                label = f"{run_dir.name}/{kind}/{path.name}"
                if path in protected:
                    output(f"SKIP newest: {label}")
                    stats["skipped"] += 1
                    continue
                try:
                    snapshot = inspect_snapshot(
                        path, kind, expected_ranks=expected_ranks
                    )
                    age = now - newest_mtime(snapshot)
                    if age < min_age_seconds:
                        output(f"SKIP young ({age / 60.0:.1f} min): {label}")
                        stats["skipped"] += 1
                        continue
                    if dry_run:
                        verb = "archive and remove" if remove_originals else "archive"
                        output(f"DRY-RUN {verb}: {label} ({len(snapshot.files)} files)")
                        stats["skipped"] += 1
                        continue
                    tar_path, created = create_verified_tar(snapshot, overwrite=overwrite)
                    output(
                        f"{'CREATED' if created else 'VERIFIED'}: {tar_path} "
                        f"({len(snapshot.files)} files)"
                    )
                    stats["archived" if created else "reused"] += 1
                    if remove_originals:
                        remove_verified_snapshot(snapshot, tar_path)
                        # N files plus one directory become one tar file.
                        stats["objects_freed"] += len(snapshot.files)
                        stats["removed"] += 1
                        output(f"REMOVED: {path}")
                except (OSError, ValueError, RuntimeError, tarfile.TarError) as error:
                    output(f"ERROR {label}: {error}")
                    stats["errors"] += 1
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="simulation run directories")
    parser.add_argument(
        "--kind", action="append", choices=("vtk", "rst"), dest="kinds",
        help="snapshot kind; repeat for both (default: vtk and rst)",
    )
    parser.add_argument(
        "--keep-latest", type=int, default=1,
        help="never archive the newest N directories per kind (default: 1)",
    )
    parser.add_argument(
        "--min-age-minutes", type=float, default=30.0,
        help="skip snapshots modified more recently than this (default: 30)",
    )
    parser.add_argument(
        "--remove-originals", action="store_true",
        help="remove a directory only after its tar archive verifies",
    )
    parser.add_argument(
        "--expected-ranks", type=int, default=None,
        help="expected files per snapshot; normally inferred from NPROCS in the PBS script",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing tar files")
    parser.add_argument("--dry-run", action="store_true", help="show eligible snapshots only")
    args = parser.parse_args(argv)

    if args.keep_latest < 0:
        parser.error("--keep-latest must be non-negative")
    if args.min_age_minutes < 0:
        parser.error("--min-age-minutes must be non-negative")
    if args.expected_ranks is not None and args.expected_ranks <= 0:
        parser.error("--expected-ranks must be positive")

    total = dict(archived=0, reused=0, removed=0, skipped=0, errors=0,
                 objects_freed=0)
    for run in args.runs:
        try:
            stats = process_run(
                run,
                kinds=tuple(args.kinds or ("vtk", "rst")),
                keep_latest=args.keep_latest,
                min_age_seconds=args.min_age_minutes * 60.0,
                remove_originals=args.remove_originals,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                expected_ranks=args.expected_ranks,
            )
        except (OSError, ValueError, RuntimeError) as error:
            print(f"ERROR {run}: {error}")
            total["errors"] += 1
            continue
        for key, value in stats.items():
            total[key] += value

    print(
        "SUMMARY archived={archived} reused={reused} removed={removed} "
        "skipped={skipped} errors={errors} quota_objects_freed={objects_freed}".format(
            **total
        )
    )
    return 1 if total["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
