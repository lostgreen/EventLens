"""Resolve a public recipe, check its inputs, and run the existing verl trainer."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from verl.trainer.config import PPOConfig
from verl.utils.task_grouping import resolve_task_homogeneous_bucket
from verl.workers.teacher_routing import resolve_opd_teacher_name
from video_proxy.training.common.data_check import inspect_jsonl


ENGINE_ROOT = Path(__file__).resolve().parents[3]


def load_recipe(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError(f"Recipe inheritance cycle at {path.name}")
    current = OmegaConf.load(path)
    if not OmegaConf.is_dict(current):
        raise ValueError("Recipe must be a YAML mapping")
    parent = current.pop("extends", None)
    base = load_recipe(path.parent / parent, (*ancestors, path)) if parent else OmegaConf.create({})
    return OmegaConf.merge(base, current)


def resolve_recipe(args):
    recipe = load_recipe(args.recipe)
    release = OmegaConf.to_container(recipe.pop("release"), resolve=True)
    config = OmegaConf.merge(OmegaConf.structured(PPOConfig()), recipe)
    if args.data_root:
        root = Path(args.data_root).expanduser().resolve()
        if release["prompt_mode"] == "no_cot":
            mix = root / "compat" / "multi_task" / "experiments" / release["mix"]
        else:
            mix = root / "mixes" / "cot_v1" / release["mix"]
        config.data.train_files = str(mix / "train.jsonl")
        config.data.val_files = str(mix / "val.jsonl")
        config.data.image_dir = str(root)
    for arg, key in (("train_file", "train_files"), ("val_file", "val_files")):
        if getattr(args, arg):
            config.data[key] = str(Path(getattr(args, arg)).expanduser().resolve())
    if args.media_root:
        config.data.image_dir = str(Path(args.media_root).expanduser().resolve())
    if args.model_path:
        config.worker.actor.model.model_path = args.model_path
    for value in args.teacher:
        name, sep, path = value.partition("=")
        if not sep or not path or name not in config.worker.ref.teacher_models:
            raise ValueError("Use --teacher aot=PATH, seg=PATH, or eventlogic=PATH as required by the recipe")
        config.worker.ref.teacher_models[name].model_path = path
    if args.output_dir:
        config.trainer.save_checkpoint_path = str(Path(args.output_dir).expanduser().resolve())
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.set))
    for key in ("train_files", "val_files", "image_dir"):
        if config.data[key]:
            config.data[key] = str(Path(config.data[key]).expanduser().resolve())
    config.trainer.save_checkpoint_path = str(Path(config.trainer.save_checkpoint_path).expanduser().resolve())
    reward, function = config.worker.reward.reward_function.rsplit(":", 1)
    config.worker.reward.reward_function = f"{(ENGINE_ROOT / reward).resolve()}:{function}"
    validate_config(config, release)
    return config, release


def validate_config(config, release):
    mode = release["prompt_mode"]
    rollout, reward = config.worker.rollout, config.worker.reward
    if mode not in {"no_cot", "cot"}:
        raise ValueError("prompt_mode must be no_cot or cot")
    if mode == "no_cot" and (
        rollout.cot_budget_enabled or rollout.cot_budget_max_tokens != 0
        or reward.reward_function_kwargs.get("cot_format_reward_enabled", False)
        or reward.enable_response_loss_weight_mask
    ):
        raise ValueError("no_cot recipe cannot enable CoT budget, format reward, or segment loss weights")
    if mode == "cot":
        tag = release["reasoning_tag"]
        if not rollout.cot_budget_enabled or not 0 < rollout.cot_budget_max_tokens < config.data.max_response_length:
            raise ValueError("CoT budget must be enabled and leave room for the answer")
        if (rollout.cot_budget_start_token, rollout.cot_budget_end_token) != (f"<{tag}>", f"</{tag}>"):
            raise ValueError("CoT budget tags must match the data reasoning_tag")
    if reward.enable_response_loss_weight_mask and config.worker.actor.loss_avg_mode not in {"token_weighted", "seq_weighted"}:
        raise ValueError("Segment loss weights require token_weighted or seq_weighted loss averaging")
    if rollout.max_num_batched_tokens < config.data.max_prompt_length + config.data.max_response_length:
        raise ValueError("max_num_batched_tokens must cover prompt + response lengths")
    gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node
    if gpus <= 0 or config.worker.actor.global_batch_size <= 0 or config.worker.actor.global_batch_size % gpus:
        raise ValueError("global_batch_size must be a positive multiple of the total GPU count")
    if config.algorithm.training_mode == "opd":
        if rollout.n != 1 or config.algorithm.online_filtering or config.algorithm.use_kl_loss or config.algorithm.disable_kl:
            raise ValueError("MT-OPD requires n=1, reference teachers, and no RL filtering or reference KL loss")
        for name, model in config.worker.ref.teacher_models.items():
            if not model.model_path:
                raise ValueError(f"Missing teacher path: {name}")


def preflight(config, release, check_frames=False):
    summaries = {}
    for split, path in (("train", config.data.train_files), ("monitor_val", config.data.val_files)):
        if not path:
            raise ValueError(f"Missing {split} file; set --data-root or explicit files")
        summaries[split] = inspect_jsonl(
            path, release["prompt_mode"], config.data.image_dir, check_frames,
            release.get("reasoning_tag"), config.data.max_frames,
        )
    counts = Counter()
    for task, count in summaries["train"]["tasks"].items():
        counts[resolve_task_homogeneous_bucket(task, config.data.task_homogeneous_grouping)] += count
    if config.data.task_weights is None:
        config.data.task_weights = {task: count / sum(counts.values()) for task, count in sorted(counts.items())}
    if config.algorithm.training_mode == "opd":
        ref = config.worker.ref
        for summary in summaries.values():
            for task in summary["tasks"]:
                resolve_opd_teacher_name(task, tuple(ref.teacher_models), ref.teacher_task_map, ref.default_teacher)
        for name, model in ref.teacher_models.items():
            path = Path(model.model_path).expanduser()
            if not (path / "config.json").is_file():
                raise ValueError(f"Teacher {name} must be a merged local HF checkpoint containing config.json")
            model.model_path = str(path.resolve())
            model.tokenizer_path = model.model_path
    return summaries


def write_run(config, release, summaries, dry_run):
    destination = Path(config.trainer.save_checkpoint_path)
    destination.mkdir(parents=True, exist_ok=True)
    config_name = "preflight_config.yaml" if dry_run else "resolved_config.yaml"
    manifest_name = "preflight_manifest.json" if dry_run else "run_manifest.json"
    config_path, manifest_path = destination / config_name, destination / manifest_name
    if (config_path.exists() or manifest_path.exists()) and not config.trainer.find_last_checkpoint:
        raise ValueError("Run directory already has provenance; use a new --output-dir or explicit trainer.find_last_checkpoint=true")
    config_yaml = OmegaConf.to_yaml(config, resolve=True)
    resolved_path = destination / "resume_config.yaml" if config_path.exists() else config_path
    manifest_target = destination / "resume_manifest.json" if manifest_path.exists() else manifest_path
    if resolved_path.exists() or manifest_target.exists():
        raise ValueError("Run directory already has provenance; use a fresh output directory")
    resolved_path.write_text(config_yaml, encoding="utf-8")
    versions = {}
    for name in ("torch", "transformers", "vllm", "ray", "omegaconf"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    snapshot = ENGINE_ROOT.parent / "manifests" / "source_snapshot.json"
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "preflight_only" if dry_run else "ready_to_launch",
        "recipe": release, "data": summaries, "dataset_identity_status": "declared_version_membership_unverified",
        "config_sha256": hashlib.sha256(config_yaml.encode()).hexdigest(),
        "config_file": resolved_path.name, "python_version": sys.version.split()[0], "packages": versions,
        "source_snapshot": json.loads(snapshot.read_text()) if snapshot.exists() else None,
        "training_code_sha256": code_checksum(),
    }
    manifest_target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return resolved_path, manifest_target


def code_checksum():
    digest = hashlib.sha256()
    for root in ("verl", "video_proxy", "configs"):
        for path in sorted((ENGINE_ROOT / root).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml"}:
                digest.update(str(path.relative_to(ENGINE_ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--data-root", help="Root of a prepared EventLens data release")
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    parser.add_argument("--media-root")
    parser.add_argument("--model-path")
    parser.add_argument("--teacher", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--check-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check files and write provenance without importing GPU workers")
    args = parser.parse_args()
    try:
        config, release = resolve_recipe(args)
        summaries = preflight(config, release, args.check_frames)
        path, manifest = write_run(config, release, summaries, args.dry_run)
    except (ValueError, OSError, OmegaConfBaseException) as exc:
        parser.exit(2, f"Preflight failed: {exc}\n")
    print(f"Recipe: {release['recipe_id']}; train={summaries['train']['rows']}; monitor_val={summaries['monitor_val']['rows']}")
    print(f"Config: {path}\nManifest: {manifest}")
    if args.dry_run:
        return 0
    os.environ.setdefault("TENSORBOARD_DIR", str(Path(config.trainer.save_checkpoint_path) / "tensorboard"))
    os.environ.setdefault("TENSORBOARD_FLAT", "1")
    if config.worker.rollout.cot_budget_enabled:
        os.environ.setdefault("VLLM_USE_V1", "1")
    return subprocess.call([sys.executable, "-m", "verl.trainer.main", f"config={path}"], cwd=ENGINE_ROOT)


if __name__ == "__main__":
    sys.exit(main())
