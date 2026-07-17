import os
import tarfile
import time
import unittest

from tigress_ncr_tools.archive_snapshots import inspect_snapshot, process_run


def make_snapshot(run, kind, number, ranks=3, age_seconds=3600):
    path = run / kind / f"{number:04d}"
    path.mkdir(parents=True)
    for rank in range(ranks):
        rank_part = "" if rank == 0 else f"-id{rank}"
        output = path / f"R8_8pc_NCR{rank_part}.{number:04d}.{kind}"
        output.write_bytes(f"rank {rank}\n".encode())
        old = time.time() - age_seconds
        os.utime(output, (old, old))
    old = time.time() - age_seconds
    os.utime(path, (old, old))
    return path


class ArchiveSnapshotsTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temporary = tempfile.TemporaryDirectory()
        from pathlib import Path
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_archives_old_snapshot_and_protects_latest(self):
        old = make_snapshot(self.tmp_path, "vtk", 0)
        latest = make_snapshot(self.tmp_path, "vtk", 1)

        stats = process_run(
            self.tmp_path, kinds=("vtk",), keep_latest=1,
            min_age_seconds=60, remove_originals=True, expected_ranks=3,
        )

        archive = self.tmp_path / "vtk" / "R8_8pc_NCR.0000.tar"
        self.assertTrue(archive.is_file())
        self.assertFalse(old.exists())
        self.assertTrue(latest.is_dir())
        self.assertEqual(stats["removed"], 1)
        self.assertEqual(stats["objects_freed"], 3)
        with tarfile.open(archive) as stream:
            self.assertEqual(stream.getnames(), [
                "0000",
                "0000/R8_8pc_NCR.0000.vtk",
                "0000/R8_8pc_NCR-id1.0000.vtk",
                "0000/R8_8pc_NCR-id2.0000.vtk",
            ])


    def test_existing_verified_tar_can_remove_original(self):
        snapshot = make_snapshot(self.tmp_path, "rst", 3)
        first = process_run(
            self.tmp_path, kinds=("rst",), keep_latest=0,
            min_age_seconds=0, remove_originals=False, expected_ranks=3,
        )
        self.assertEqual(first["archived"], 1)
        self.assertTrue(snapshot.exists())

        second = process_run(
            self.tmp_path, kinds=("rst",), keep_latest=0,
            min_age_seconds=0, remove_originals=True, expected_ranks=3,
        )
        self.assertEqual(second["reused"], 1)
        self.assertEqual(second["removed"], 1)
        self.assertFalse(snapshot.exists())


    def test_noncontiguous_ranks_are_never_archived(self):
        snapshot = make_snapshot(self.tmp_path, "vtk", 2, ranks=3)
        (snapshot / "R8_8pc_NCR-id1.0002.vtk").unlink()

        with self.assertRaisesRegex(ValueError, "non-contiguous MPI ranks"):
            inspect_snapshot(snapshot, "vtk", expected_ranks=3)

        stats = process_run(
            self.tmp_path, kinds=("vtk",), keep_latest=0,
            min_age_seconds=0, remove_originals=True, expected_ranks=3,
        )
        self.assertEqual(stats["errors"], 1)
        self.assertTrue(snapshot.exists())
        self.assertFalse(
            (self.tmp_path / "vtk" / "R8_8pc_NCR.0002.tar").exists()
        )


    def test_missing_trailing_ranks_are_never_archived(self):
        snapshot = make_snapshot(self.tmp_path, "vtk", 5, ranks=2)
        with self.assertRaisesRegex(ValueError, "found 2 of 3"):
            inspect_snapshot(snapshot, "vtk", expected_ranks=3)

    def test_dry_run_does_not_modify_snapshot(self):
        snapshot = make_snapshot(self.tmp_path, "vtk", 4)
        stats = process_run(
            self.tmp_path, kinds=("vtk",), keep_latest=0,
            min_age_seconds=0, remove_originals=True, dry_run=True,
            expected_ranks=3,
        )
        self.assertTrue(snapshot.exists())
        self.assertEqual(stats["skipped"], 1)
        self.assertFalse(
            (self.tmp_path / "vtk" / "R8_8pc_NCR.0004.tar").exists()
        )
        self.assertFalse((self.tmp_path / ".archive-snapshots.lock").exists())


if __name__ == "__main__":
    unittest.main()
