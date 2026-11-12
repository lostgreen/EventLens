---
pretty_name: EventLens Training Data
language:
- en
tags:
- video
- datasets
- multimodal
- post-training
- event-understanding
task_categories:
- visual-question-answering
license: other
license_name: source-specific-terms
license_link: LICENSES.md
configs:
- config_name: paper_mt_opd
  default: true
  data_files:
  - split: train
    path: data/paper_mt_opd/train-*.parquet
  - split: monitor_val
    path: data/paper_mt_opd/monitor_val-*.parquet
- config_name: paper_seg_teacher
  data_files:
  - split: train
    path: data/paper_seg_teacher/train-*.parquet
  - split: monitor_val
    path: data/paper_seg_teacher/monitor_val-*.parquet
- config_name: paper_prog_teacher
  data_files:
  - split: train
    path: data/paper_prog_teacher/train-*.parquet
  - split: monitor_val
    path: data/paper_prog_teacher/monitor_val-*.parquet
- config_name: paper_relation_teacher
  data_files:
  - split: train
    path: data/paper_relation_teacher/train-*.parquet
  - split: monitor_val
    path: data/paper_relation_teacher/monitor_val-*.parquet
- config_name: tg
  data_files:
  - split: train
    path: data/tasks/tg/train-*.parquet
  - split: monitor_val
    path: data/tasks/tg/monitor_val-*.parquet
- config_name: mcq
  data_files:
  - split: train
    path: data/tasks/mcq/train-*.parquet
  - split: monitor_val
    path: data/tasks/mcq/monitor_val-*.parquet
- config_name: seg
  data_files:
  - split: train
    path: data/tasks/seg/train-*.parquet
  - split: monitor_val
    path: data/tasks/seg/monitor_val-*.parquet
- config_name: prog
  data_files:
  - split: train
    path: data/tasks/prog/train-*.parquet
  - split: monitor_val
    path: data/tasks/prog/monitor_val-*.parquet
- config_name: relation
  data_files:
  - split: train
    path: data/tasks/relation/train-*.parquet
  - split: monitor_val
    path: data/tasks/relation/monitor_val-*.parquet
---

# EventLens Training Data

**Draft template.** This file is not a published dataset. The Parquet paths
refer to planned export artifacts. Publish only populated configs and verified
source terms; the placeholder license summary is not a redistribution grant.

## Contents

The paper release uses no-CoT prompting. The full mix has 37,428 training rows
and 1,950 monitor-validation rows, based on audited existing release statistics.
Specialist and per-task counts, source revisions, membership manifests, and
export checksums must be completed before upload.

`monitor_val` preserves the historical training monitor. Historical progression
monitor rows include 300 exact train overlaps; relation includes 112. It is not
an independent validation/test split. New independent splits will receive a
separate dataset version and must not overwrite the paper data.

## Load

```python
from datasets import load_dataset

data = load_dataset("<org>/EventLens-Train", "paper_mt_opd", revision="<commit_sha>")
train = data["train"]
monitor = data["monitor_val"]
```

This loads annotations and media references. It does not fetch or decode the
underlying source videos or frame archives automatically. Media availability
and source-specific rights are documented separately.

## Schema And Provenance

Rows identify the source, task/subtype, prompt mode, original answer, ordered
media references, source/presentation timing, and build/hash provenance.
`training_metadata_json` preserves the sanitized legacy timing/reward metadata
without introducing inconsistent nested Arrow types. Stable `sample_id`
identifies content; `row_id` and mix manifests preserve repeated occurrences
and frozen mix order. Unknown provenance remains null and is tracked in the
release manifest rather than guessed.

## Sources And Licenses

Fill `LICENSES.md`, source revisions, attribution, and media access instructions
before publication. ET-Instruct annotations and TimeLens annotations have
different source terms. Other base data and media permissions remain under
review. Code Apache-2.0 does not license these data.

## CoT Version

CoT prompting views will be published separately with an immutable parent
dataset revision, `parent_sample_id`, conversion provenance, and identical
ground-truth/media semantics. They do not contain gold reasoning traces.

## Citation

Insert the confirmed paper citation and source attributions before publication.
