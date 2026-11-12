"""Verify reviewed figures can ship without allowing arbitrary binary assets."""

import importlib.util
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("release_checker", ROOT / "EasyR1/scripts/check_release.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class ReleaseFiguresTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "release"
        self.root.mkdir()

    def copy_figure(self, target="assets/figures/overview.png"):
        path = self.root / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / "assets/figures/overview.png").read_bytes())
        return path

    def test_reviewed_figure_is_in_export(self):
        self.copy_figure()
        report = CHECKER.check_tree(self.root)
        self.assertEqual(report["status"], "pass")
        output = self.root.parent / "release.tar.gz"
        CHECKER.export_archive(self.root, output, report)
        with tarfile.open(output) as archive:
            payload = archive.extractfile("EventLens/assets/figures/overview.png").read()
        self.assertEqual(payload, (ROOT / "assets/figures/overview.png").read_bytes())

    def test_modified_figure_requires_review(self):
        path = self.copy_figure()
        path.write_bytes(path.read_bytes() + b"unreviewed metadata")
        report = CHECKER.check_tree(self.root)
        self.assertEqual(report["issues"], [{"file": "assets/figures/overview.png", "rule": "unreviewed_image"}])

    def test_arbitrary_binary_remains_blocked(self):
        self.copy_figure("assets/figures/other.png")
        report = CHECKER.check_tree(self.root)
        self.assertEqual(report["issues"], [{"file": "assets/figures/other.png", "rule": "binary_artifact"}])

    def test_reviewed_figure_does_not_allow_private_files(self):
        self.copy_figure()
        (self.root / ".env").write_text("PRIVATE=value")
        report = CHECKER.check_tree(self.root)
        self.assertEqual(report["status"], "fail")
        with self.assertRaises(ValueError):
            CHECKER.export_archive(self.root, self.root.parent / "blocked.tar.gz", report)


if __name__ == "__main__":
    unittest.main()
