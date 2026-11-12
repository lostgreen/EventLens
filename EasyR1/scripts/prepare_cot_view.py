"""Create a CoT prompting view and its parent mapping without modifying the source."""

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from video_proxy.data.scripts.convert_jsonl_to_cot import convert_record
from video_proxy.training.common.data_check import inspect_jsonl


def content_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def prepare_view(input_dir, output_dir, parent_version, reasoning_tag="thought"):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory already exists; select a new view directory")
    source = {split: inspect_jsonl(input_dir / f"{split}.jsonl", "no_cot") for split in ("train", "val")}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cot-view-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "view"
        staging.mkdir()
        reasons = Counter()
        with (staging / "parent_map.jsonl").open("w", encoding="utf-8") as mapping:
            for split in ("train", "val"):
                with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as src, (staging / f"{split}.jsonl").open("w", encoding="utf-8") as dst:
                    for line_no, line in enumerate(src, 1):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        converted, changed, reason = convert_record(row, reasoning_tag=reasoning_tag)
                        if not changed or f"<{reasoning_tag}>" not in converted["prompt"]:
                            raise ValueError(f"Unsupported CoT conversion at {split}:{line_no} ({reason})")
                        if converted["answer"] != row["answer"] or converted["videos"] != row["videos"]:
                            raise ValueError(f"Answer or media changed at {split}:{line_no}")
                        parent_id = content_hash(row)
                        sample_id = content_hash([parent_version, parent_id, split, line_no, reasoning_tag])
                        metadata = converted.get("metadata") or {}
                        if not isinstance(metadata, dict):
                            raise ValueError(f"metadata must be an object at {split}:{line_no}")
                        converted["metadata"] = metadata
                        metadata["eventlens_view"] = {
                            "sample_id": sample_id, "parent_sample_id": parent_id,
                            "parent_dataset_version": parent_version, "prompt_mode": "cot",
                            "reasoning_tag": reasoning_tag,
                        }
                        dst.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        mapping.write(json.dumps({
                            "sample_id": sample_id, "parent_sample_id": parent_id,
                            "split": split, "parent_line": line_no,
                            "answer_sha256": content_hash(row["answer"]), "media_sha256": content_hash(row["videos"]),
                        }) + "\n")
                        reasons[reason] += 1
        derived = {split: inspect_jsonl(staging / f"{split}.jsonl", "cot", reasoning_tag=reasoning_tag) for split in source}
        for split in source:
            if source[split]["rows"] != derived[split]["rows"]:
                raise ValueError("Parent/child count mismatch")
            derived[split].pop("path")
            source[split].pop("path")
        converter = Path(__file__).resolve().parents[1] / "video_proxy/data/scripts/convert_jsonl_to_cot.py"
        manifest = {
            "schema_version": 1, "dataset_version": "cot_v1", "parent_dataset_version": parent_version,
            "prompt_mode": "cot", "reasoning_tag": reasoning_tag,
            "contains_gold_reasoning": False, "validation_role": "monitor_val",
            "source": source, "derived": derived, "conversion_reasons": dict(reasons),
            "converter_sha256": hashlib.sha256(converter.read_bytes()).hexdigest(),
        }
        (staging / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="No-CoT mix with train.jsonl and val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-version", default="vpmixed_20260516_full_comp")
    parser.add_argument("--reasoning-tag", choices=("think", "thought"), default="thought")
    args = parser.parse_args()
    try:
        manifest = prepare_view(args.input_dir, args.output_dir, args.parent_version, args.reasoning_tag)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Conversion failed: {exc}\n")
    print(f"CoT view: train={manifest['derived']['train']['rows']}, monitor_val={manifest['derived']['val']['rows']}")
    print(f"Manifest: {Path(args.output_dir).resolve() / 'conversion_manifest.json'}")


if __name__ == "__main__":
    main()
