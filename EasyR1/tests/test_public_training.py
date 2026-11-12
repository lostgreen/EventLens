import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.reward_function.mixed_proxy_reward import compute_score
from video_proxy.training.common.data_check import inspect_jsonl
from video_proxy.training.launchers.train import ENGINE_ROOT, load_recipe, preflight, resolve_recipe, write_run


def args_for(recipe, tmp_path, **kwargs):
    values = dict(
        recipe=ENGINE_ROOT / "configs" / recipe, data_root=None, train_file=None, val_file=None,
        media_root=None, model_path=None, teacher=[], output_dir=str(tmp_path / "run"), set=[],
    )
    values.update(kwargs)
    return argparse.Namespace(**values)


def row(task="llava_mcq"):
    return {
        "prompt": "<video>\nChoose the correct option.\nA. First\nB. Second\n\nProvide your answer (a single letter from A, B) inside <answer> </answer> tags.",
        "answer": "A", "problem_type": task, "videos": [["frames/000001.jpg", "frames/000002.jpg"]],
        "metadata": {"video_fps_override": 2.0, "frame_indices": [1, 2]},
    }


def write_mix(path, rows):
    path.mkdir(parents=True)
    for split in ("train", "val"):
        (path / f"{split}.jsonl").write_text("".join(json.dumps(value) + "\n" for value in rows))


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ENGINE_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("recipe", [
    "paper/seg_teacher.yaml", "paper/prog_teacher.yaml", "paper/relation_teacher.yaml",
    "paper/student_4b.yaml", "paper/student_8b.yaml", "paper/mixed_ema_grpo.yaml", "cot/seg_teacher.yaml",
    "cot/prog_teacher.yaml", "cot/relation_teacher.yaml", "smoke/teacher.yaml",
])
def test_all_public_recipes_merge_with_actual_trainer_schema(recipe, tmp_path):
    options = args_for(recipe, tmp_path)
    if "student" in recipe:
        options.teacher = [f"{name}={tmp_path / name}" for name in ("aot", "seg", "eventlogic")]
    config, release = resolve_recipe(options)
    assert config.data.max_frames == 256
    assert "pending" in release["reproduction_status"] or recipe.startswith("smoke/")
    assert Path(config.worker.reward.reward_function.rsplit(":", 1)[0]).is_file()
    if "student" in recipe:
        assert config.algorithm.opd_topk == 10
        assert config.algorithm.opd_kl_coef == 1.0
        assert config.worker.rollout.n == 1
        assert config.data.task_homogeneous_grouping == "opd_task_group"
        assert config.data.task_homogeneous_batching
        assert config.data.max_response_length == 256
    else:
        assert config.algorithm.online_filtering is False
        assert config.data.max_response_length == 512


def test_inheritance_cycle_fails(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_recipe(tmp_path / "a.yaml")


@pytest.mark.parametrize("override", [
    "worker.rollout.cot_budget_enabled=true",
    "worker.rollout.cot_budget_max_tokens=128",
    "worker.reward.reward_function_kwargs.cot_format_reward_enabled=true",
    "worker.reward.enable_response_loss_weight_mask=true",
])
def test_direct_recipe_rejects_cot_flags(tmp_path, override):
    with pytest.raises(ValueError, match="no_cot"):
        resolve_recipe(args_for("paper/seg_teacher.yaml", tmp_path, set=[override]))


def test_cot_weighting_requires_weighted_average(tmp_path):
    options = args_for("cot/seg_teacher.yaml", tmp_path, set=["worker.reward.enable_response_loss_weight_mask=true"])
    with pytest.raises(ValueError, match="loss averaging"):
        resolve_recipe(options)
    options.set.append("worker.actor.loss_avg_mode=token_weighted")
    config, _ = resolve_recipe(options)
    assert config.worker.reward.enable_response_loss_weight_mask


def test_data_root_resolution_and_run_provenance(tmp_path):
    mix = tmp_path / "compat/multi_task/experiments/composition_base_seg_hier10k_mf256_ema"
    write_mix(mix, [row()])
    config, release = resolve_recipe(args_for("paper/seg_teacher.yaml", tmp_path, data_root=str(tmp_path)))
    summaries = preflight(config, release)
    assert summaries["train"]["rows"] == 1
    assert config.data.task_weights == {"llava_mcq": 1.0}
    path, manifest = write_run(config, release, summaries, True)
    assert OmegaConf.load(path).worker.rollout.cot_budget_enabled is False
    saved = json.loads(manifest.read_text())
    assert saved["status"] == "preflight_only"
    assert saved["data"]["train"]["sha256"] == summaries["train"]["sha256"]
    assert len(saved["training_code_sha256"]) == 64
    with pytest.raises(ValueError, match="provenance"):
        write_run(config, release, summaries, True)
    actual_path, _ = write_run(config, release, summaries, False)
    assert actual_path.name == "resolved_config.yaml"


def test_opd_preflight_uses_real_routing_and_group_counts(tmp_path):
    mix = tmp_path / "mix"
    write_mix(mix, [row("temporal_grounding"), row("llava_mcq"), row("seg_aot_action_v2t_3way"), row("temporal_seg_hier_L2"), row("event_logic_sort")])
    teachers = []
    for name in ("aot", "seg", "eventlogic"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text("{}")
        teachers.append(f"{name}={path}")
    config, release = resolve_recipe(args_for("paper/student_8b.yaml", tmp_path,
        teacher=teachers, train_file=str(mix / "train.jsonl"), val_file=str(mix / "val.jsonl")))
    preflight(config, release)
    assert dict(config.data.task_weights) == {"aot": 0.2, "base": 0.4, "logic": 0.2, "seg": 0.2}
    write_mix(tmp_path / "unknown", [row("unknown_task")])
    config.data.val_files = str(tmp_path / "unknown/val.jsonl")
    with pytest.raises(ValueError, match="No OPD teacher"):
        preflight(config, release)


def test_checker_rejects_mode_message_and_media_mismatch(tmp_path):
    sample = row()
    sample["prompt"] += " Think step by step inside <thought></thought>."
    write_mix(tmp_path / "bad", [sample])
    with pytest.raises(ValueError, match="no_cot"):
        inspect_jsonl(tmp_path / "bad/train.jsonl", "no_cot")
    with pytest.raises(ValueError, match="missing media"):
        inspect_jsonl(tmp_path / "bad/train.jsonl", "cot", tmp_path, True, "thought")
    sample["messages"] = [{"role": "user", "content": "different"}]
    write_mix(tmp_path / "mismatch", [sample])
    with pytest.raises(ValueError, match="disagree"):
        inspect_jsonl(tmp_path / "mismatch/train.jsonl", "cot")


def test_cot_view_preserves_answers_media_and_parent_identity(tmp_path):
    tool = load_tool("prepare_cot_view")
    parent = tmp_path / "parent"
    write_mix(parent, [row()])
    original = (parent / "train.jsonl").read_bytes()
    output = tmp_path / "cot"
    manifest = tool.prepare_view(parent, output, "paper_fixture")
    converted = json.loads((output / "train.jsonl").read_text())
    assert converted["answer"] == row()["answer"]
    assert converted["videos"] == row()["videos"]
    assert converted["metadata"]["frame_indices"] == [1, 2]
    assert converted["metadata"]["eventlens_view"]["parent_sample_id"] == tool.content_hash(row())
    assert manifest["derived"]["train"]["rows"] == 1
    assert manifest["contains_gold_reasoning"] is False
    assert (parent / "train.jsonl").read_bytes() == original
    config, release = resolve_recipe(args_for("cot/seg_teacher.yaml", tmp_path,
        train_file=str(output / "train.jsonl"), val_file=str(output / "val.jsonl")))
    assert preflight(config, release)["train"]["rows"] == 1
    with pytest.raises(ValueError, match="exists"):
        tool.prepare_view(parent, output, "paper_fixture")


def test_unsupported_conversion_does_not_leave_partial_view(tmp_path):
    sample = row()
    sample["prompt"] = "Unrecognized task instructions"
    write_mix(tmp_path / "parent", [sample])
    with pytest.raises(ValueError, match="Unsupported"):
        load_tool("prepare_cot_view").prepare_view(tmp_path / "parent", tmp_path / "cot", "fixture")
    assert not (tmp_path / "cot").exists()


def test_choice_reward_and_cot_format_use_actual_reward_functions():
    sample = {"response": "<answer>F</answer>", "ground_truth": "F", "problem_type": "event_logic_predict_next"}
    assert compute_score([sample])[0]["accuracy"] == 1.0
    sample["cot_budget_debug"] = {"cot_budget_enabled": True, "cot_repaired": True}
    scores = compute_score([sample], cot_format_reward_enabled=True, cot_format_truncated=0.5)
    assert scores[0]["accuracy"] == 1.0
    assert scores[0]["overall"] == 0.5


def test_release_scan_and_archive_refuse_private_material(tmp_path):
    checker = load_tool("check_release")
    (tmp_path / "README.md").write_text("Public source")
    assert checker.check_tree(tmp_path)["status"] == "pass"
    (tmp_path / ".env").write_text("PRIVATE=value")
    report = checker.check_tree(tmp_path)
    assert report["status"] == "fail"
    with pytest.raises(ValueError, match="checks must pass"):
        checker.export_archive(tmp_path, tmp_path.parent / "blocked.tar.gz", report)


@pytest.mark.parametrize("mode", ["paper", "cot"])
def test_shell_entrypoints_run_preflight_from_another_directory(tmp_path, mode):
    write_mix(tmp_path / "source", [row()])
    mix = tmp_path / "source"
    if mode == "cot":
        mix = tmp_path / "cot"
        load_tool("prepare_cot_view").prepare_view(tmp_path / "source", mix, "fixture")
    command = [
        "bash", str(ENGINE_ROOT / "video_proxy/experiments/teacher_train/run.sh"), "seg", mode,
        "--train-file", str(mix / "train.jsonl"), "--val-file", str(mix / "val.jsonl"),
        "--output-dir", str(tmp_path / "run"), "--dry-run",
    ]
    completed = subprocess.run(command, cwd=tmp_path, env={**os.environ, "PYTHON": sys.executable}, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "train=1; monitor_val=1" in completed.stdout
    manifest = json.loads((tmp_path / "run/preflight_manifest.json").read_text())
    assert manifest["recipe"]["prompt_mode"] == ("no_cot" if mode == "paper" else "cot")
    assert "verl.trainer.main" not in sys.modules
