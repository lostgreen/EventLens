# EventLens Training Module

This directory retains the `verl` and `video_proxy` import namespaces of the
current research engine. The distribution remains named `verl`; use an
isolated environment when working with other verl versions.

## Install

Training requires Linux, NVIDIA GPUs, Python 3.10 or newer, and a compatible
CUDA/PyTorch/vLLM/FlashAttention stack, with
`transformers>=4.54.0,<=4.57.0`. Use an isolated Python 3.11 environment:

```bash
conda create -n eventlens python=3.11 -y
conda activate eventlens
python -m pip install -e .
```

For configuration/data preflight and CPU tests without installing the GPU stack:

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q
```

## Train

Run commands from this directory. The public launcher is:

```bash
python -m video_proxy.training.launchers.train \
  --recipe configs/paper/seg_teacher.yaml \
  --train-file /path/to/train.jsonl --val-file /path/to/val.jsonl \
  --media-root /path/to/media --output-dir /path/to/run --dry-run
```

`--data-root` resolves prepared release mixes. Explicit files override the
resolved mix. Relative media references resolve against `--media-root` or the
data root.
`--check-frames` additionally checks media existence. Data is consumed as-is;
the launcher does not resample frames or rebuild frozen mixes.

`--set KEY=VALUE` changes a structured trainer setting. Unknown fields fail.
Use a fresh output directory for each run; automatic checkpoint discovery is
disabled by default. Thin wrappers in `video_proxy/experiments/` select recipes
without adding cluster-specific behavior.

## Export A Checkpoint

```bash
python scripts/model_merger.py \
  --local_dir /path/to/checkpoint/actor \
  --base_model /path/to/original-base-model
```

The merger produces a local Hugging Face checkpoint. `--hf_upload_path`
optionally uploads the result to a Hub repository.

For tasks, Hugging Face data loading, teacher training, and MT-OPD, follow
the [project README](../README.md#training).
