"""Check training JSONL without printing sample prompts or answers."""

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_proxy.training.common.data_check import inspect_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--prompt-mode", choices=("no_cot", "cot"), required=True)
    parser.add_argument("--reasoning-tag", choices=("thought", "think"))
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--check-frames", action="store_true")
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        reports = [inspect_jsonl(path, args.prompt_mode, args.media_root, args.check_frames, args.reasoning_tag, args.max_frames) for path in args.files]
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Data check failed: {exc}\n")
    if args.report:
        args.report.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    for report in reports:
        print(json.dumps(report))


if __name__ == "__main__":
    main()
