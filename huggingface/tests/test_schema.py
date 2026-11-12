"""Check local Hub layout loading and timing semantics without real media."""

import importlib.util
from pathlib import Path

from datasets import Dataset, load_dataset


SPEC = importlib.util.spec_from_file_location("eventlens_hf_schema", Path(__file__).parents[1] / "schema.py")
SCHEMA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCHEMA)


def test_local_card_loads_named_and_default_config(tmp_path):
    root = SCHEMA.write_fixture(tmp_path / "fixture")
    named = load_dataset(str(root), "paper_mt_opd", cache_dir=str(tmp_path / "cache"))
    default = load_dataset(str(root), cache_dir=str(tmp_path / "cache"))
    assert {key: len(value) for key, value in named.items()} == {"train": 2, "monitor_val": 1}
    assert default["train"].features == SCHEMA.features()
    assert named["monitor_val"][0]["split_role"] == "monitor_val"
    assert named["train"][0]["media"][0]["frame_keys"] == [
        "fixture/video/000001.jpg", "fixture/video/000002.jpg",
    ]


def test_parquet_preserves_multi_video_order_and_unknown_source_timing(tmp_path):
    row = SCHEMA.fixture_row()
    reversed_media = dict(row["media"][0])
    reversed_media.update({
        "media_id": "fixture_reversed", "ordering": "reverse",
        "frame_keys": list(reversed(reversed_media["frame_keys"])),
        "source_frame_indices": [2, 1], "source_timestamps_sec": [None, None],
    })
    row["media"].append(reversed_media)
    path = tmp_path / "multi.parquet"
    Dataset.from_list([row], features=SCHEMA.features()).to_parquet(path)
    restored = Dataset.from_parquet(str(path), cache_dir=str(tmp_path / "cache"))[0]
    assert restored["media"] == row["media"]
    assert restored["parent_sample_id"] is None
    assert restored["media"][1]["source_timestamps_sec"] == [None, None]
    assert restored["media"][1]["presentation_timestamps_sec"] == [0.0, 0.5]
