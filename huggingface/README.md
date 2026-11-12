# Hugging Face Schema Tools

`schema.py` defines typed Parquet rows.
`templates/DATASET_CARD.md` and `templates/LICENSES.md` provide Hub card
templates. The current training launcher uses JSONL; these tools demonstrate
the planned dataset format.

From the repository root:

```bash
python -m pip install -r huggingface/requirements.txt
python huggingface/schema.py --fixture-dir /tmp/eventlens-hf-fixture
python -m pytest huggingface/tests -q
```

The fixture contains three synthetic rows and no media payloads. Tests cover
local dataset loading, multiple videos, reversed frames, and nullable
timestamps.

For dataset loading and training usage, follow the
[project README](../README.md#training).
