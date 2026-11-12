"""Planned Parquet schema and a synthetic local Hugging Face compatibility check."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset, Features, List, Value


def features():
    text, number = Value("string"), Value("float64")
    media = {
        "media_id": text, "source_dataset": text, "source_video_id": text,
        "kind": text, "video_path": text, "frame_keys": List(text),
        "source_frame_indices": List(Value("int64")),
        "source_timestamps_sec": List(number), "presentation_timestamps_sec": List(number),
        "clip_start_sec": number, "clip_end_sec": number, "duration_sec": number,
        "source_fps": number, "effective_fps": number, "ordering": text,
    }
    return Features({
        "row_id": text, "sample_id": text, "source_sample_id": text,
        "dataset_version": text, "source_dataset": text, "source_video_ids": List(text),
        "task_family": text, "problem_type": text, "level": Value("int8"),
        "prompt_mode": text, "reasoning_tag": text, "parent_sample_id": text,
        "parent_dataset_revision": text, "prompt": text,
        "messages": List({"role": text, "content": text}),
        "answer": text, "answer_format": text, "answer_time_coordinate": text,
        "media": List(media), "split_role": text, "mix_id": text,
        "mix_row_index": Value("int64"), "annotation_id": text, "build_id": text,
        "license_ids": List(text), "prompt_sha256": text, "answer_sha256": text,
        "training_metadata_json": text,
    })


def fixture_row(row_number=0):
    prompt = "<video>\nChoose A or B. Provide your answer inside <answer></answer> tags."
    row = dict.fromkeys(features())
    row.update({
        "row_id": f"fixture_row_{row_number}", "sample_id": f"fixture_sample_{row_number}",
        "dataset_version": "synthetic_schema_fixture", "source_dataset": "synthetic",
        "source_video_ids": ["fixture_video"], "task_family": "mcq", "problem_type": "llava_mcq",
        "prompt_mode": "no_cot", "prompt": prompt,
        "messages": [{"role": "user", "content": prompt}], "answer": "A", "answer_format": "choice_letter",
        "media": [{
            "media_id": "fixture_sequence", "source_dataset": "synthetic", "source_video_id": "fixture_video",
            "kind": "frames", "video_path": None,
            "frame_keys": ["fixture/video/000001.jpg", "fixture/video/000002.jpg"],
            "source_frame_indices": [1, 2], "source_timestamps_sec": [0.0, 0.5],
            "presentation_timestamps_sec": [0.0, 0.5], "clip_start_sec": 0.0, "clip_end_sec": 1.0,
            "duration_sec": 1.0, "source_fps": 2.0, "effective_fps": 2.0, "ordering": "forward",
        }],
        "split_role": "train", "mix_id": "fixture", "mix_row_index": row_number,
        "build_id": "synthetic", "license_ids": ["synthetic_fixture"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "answer_sha256": hashlib.sha256(b"A").hexdigest(), "training_metadata_json": "{}",
    })
    return row


def write_fixture(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Fixture directory exists; choose a fresh path")
    directory = destination / "data/paper_mt_opd"
    directory.mkdir(parents=True)
    train = Dataset.from_list([fixture_row(0), fixture_row(1)], features=features())
    monitor = fixture_row(2)
    monitor["split_role"] = "monitor_val"
    Dataset.from_list([monitor], features=features()).to_parquet(str(directory / "monitor_val-00000-of-00001.parquet"))
    train.to_parquet(str(directory / "train-00000-of-00001.parquet"))
    card = """---
pretty_name: EventLens Synthetic Schema Fixture
configs:
- config_name: paper_mt_opd
  default: true
  data_files:
  - split: train
    path: data/paper_mt_opd/train-*.parquet
  - split: monitor_val
    path: data/paper_mt_opd/monitor_val-*.parquet
---
# Synthetic schema fixture

Three synthetic rows; no paper data or real media is included.
Frame references intentionally have no JPEG payload. This checks Parquet
schema, nested fields, and local dataset-card config/split loading only.
"""
    (destination / "README.md").write_text(card)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-output", type=Path)
    parser.add_argument("--fixture-dir", type=Path)
    args = parser.parse_args()
    if args.schema_output:
        args.schema_output.write_text(json.dumps(features().to_dict(), indent=2) + "\n")
    if args.fixture_dir:
        print(write_fixture(args.fixture_dir))
    if not args.schema_output and not args.fixture_dir:
        print(json.dumps({"columns": len(features()), "schema": "eventlens_parquet_v1"}))


if __name__ == "__main__":
    main()
