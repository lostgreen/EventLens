"""Check the public tree and optionally export only that tree as a source archive."""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist"}
FORBIDDEN_DIRS = {".codex", ".aws", "outputs", "checkpoints", "wandb", "Evaluation", "evaluation", "VLMEvalKit"}
RULES = {
    "internal_path": re.compile(r"/(?:m2v_intern\d*|home/xuboshen|Users/lostgreen)(?:/|\b)"),
    "internal_endpoint": re.compile(r"ko\.txyun|corp\.kuaishou\.com|pa/" r"gmn-"),
    "credential": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}"),
}
# Only visually reviewed README figures may bypass text scanning.
REVIEWED_IMAGES = {
    "assets/figures/overview.png": "cba4786490a3bd7bd766c608d50e82fc6cb173250031eeae4beeaf54ace54b52",
    "assets/figures/training-pipeline.png": "1ba9541a79adb9c8d182f7bc2872e638f8cf84453026401f5da2503277361817",
    "assets/figures/event-structure.png": "af3fad7bb35616c9a1dff521aa19bb5fc8fb4be42f36b015fba1efda8060ace0",
    "assets/figures/data-domains.png": "6ea84ec38c7da0e5f102fa29954c62f6deaab70312080bf5ad9c58978cb1ed4a",
}


def check_tree(root):
    root = Path(root).resolve()
    files, issues = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if any(part in FORBIDDEN_DIRS for part in relative.parts) or path.name == ".env":
            issues.append({"file": str(relative), "rule": "private_or_generated_path"})
            continue
        if path.is_symlink():
            issues.append({"file": str(relative), "rule": "symlink"})
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            issues.append({"file": str(relative), "rule": "oversized_code_artifact"})
            continue
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected_image = REVIEWED_IMAGES.get(relative.as_posix())
        if expected_image is not None:
            if digest != expected_image or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                issues.append({"file": str(relative), "rule": "unreviewed_image"})
                continue
            files.append({"path": str(relative), "sha256": digest})
            continue
        try:
            contents = payload.decode("utf-8")
        except UnicodeDecodeError:
            issues.append({"file": str(relative), "rule": "binary_artifact"})
            continue
        for rule, pattern in RULES.items():
            if pattern.search(contents):
                issues.append({"file": str(relative), "rule": rule})
        files.append({"path": str(relative), "sha256": digest})
    return {"schema_version": 1, "status": "pass" if not issues else "fail", "file_count": len(files), "issues": issues, "files": files}


def export_archive(root, output, report):
    output = Path(output).resolve()
    if output.is_relative_to(Path(root).resolve()):
        raise ValueError("Archive must be outside the release tree")
    if output.exists():
        raise ValueError("Archive exists; select a new output path")
    if report["issues"]:
        raise ValueError("Release checks must pass before export")
    with tarfile.open(output, "w:gz") as archive:
        for entry in report["files"]:
            archive.add(Path(root) / entry["path"], arcname="EventLens/" + entry["path"], recursive=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    report = check_tree(args.root)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.archive:
        export_archive(args.root, args.archive, report)
    print(json.dumps({key: report[key] for key in ("status", "file_count", "issues")}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
