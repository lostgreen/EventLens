# Data Preparation Modules

`base_sources/` prepares TG and MCQ source views. `pipelines/data_curation/`
performs duration-first filtering. `pipelines/proxy_construction/` contains
annotation, boundary, progression, and relation builders. `mixing/` composes
task views and preserves the frame sampling contract. `scripts/` retains the
prompt converter and video checks.

Run module CLIs from `EasyR1/`, for example:

```bash
python -m video_proxy.data.pipelines.data_curation.curation.duration_filter \
  --dataset et_instruct_164k --input /path/to/annotations.json \
  --video-root /path/to/videos --output-dir /path/to/curated
python -m video_proxy.data.pipelines.proxy_construction.annotation.build_hierarchical_annotations --help
python -m video_proxy.data.pipelines.proxy_construction.event_boundary.build_boundary_frame_list_data --help
python -m video_proxy.data.pipelines.proxy_construction.event_progression.build_progression_frame_list_data --help
python -m video_proxy.data.pipelines.proxy_construction.event_relation.build_relation_frame_list_data --help
```

VLM annotation/relation generation requires an explicit `--model`. Configure
`--api-base` or `OPENAI_BASE_URL` and supply the appropriate key through the
provider environment. Optional processing dependencies are listed separately
in `requirements/data.txt`; `ffmpeg` must be installed for frame extraction.
Do not regenerate the paper mix with today's annotation model and label it as
the original frozen data.
