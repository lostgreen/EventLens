"""Stream training data into compact counts and checksums without logging prompts."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


_COT = re.compile(r"<(?:think|thought)>|think\s+step\s+by\s+step|first,?\s+think", re.IGNORECASE)


def inspect_jsonl(path, prompt_mode, media_root=None, check_frames=False, reasoning_tag=None, max_frames=256):
    path = Path(path)
    counts = Counter()
    digest = hashlib.sha256()
    rows = 0
    errors = []
    with path.open("rb") as stream:
        for line_no, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                prompt = row.get("prompt")
                task = row.get("problem_type")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("missing prompt")
                if not isinstance(task, str) or not task:
                    raise ValueError("missing problem_type")
                if "answer" not in row:
                    raise ValueError("missing answer")
                is_cot = bool(_COT.search(prompt))
                if is_cot != (prompt_mode == "cot"):
                    raise ValueError(f"prompt does not match {prompt_mode}")
                if prompt_mode == "cot" and reasoning_tag and f"<{reasoning_tag}>" not in prompt:
                    raise ValueError(f"missing <{reasoning_tag}> instruction")
                messages = row.get("messages")
                if messages is not None:
                    users = [m for m in messages if m.get("role") == "user"]
                    if not users or users[-1].get("content") != prompt:
                        raise ValueError("messages and prompt disagree")
                videos = row.get("videos")
                if not isinstance(videos, list) or not videos:
                    raise ValueError("missing videos")
                for video in videos:
                    refs = video if isinstance(video, list) else [video]
                    if not refs or any(not isinstance(ref, str) or not ref for ref in refs):
                        raise ValueError("invalid video/frame references")
                    if isinstance(video, list) and len(video) > max_frames:
                        raise ValueError(f"frame list exceeds max_frames={max_frames}")
                    if check_frames:
                        for ref in refs:
                            target = Path(media_root or ".") / ref
                            if not target.is_file():
                                raise ValueError("missing media reference")
                rows += 1
                counts[task] += 1
            except (ValueError, TypeError, AttributeError) as exc:
                errors.append(f"line {line_no}: {exc}")
                if len(errors) == 5:
                    break
    if errors:
        raise ValueError(f"Invalid training data {path}: {'; '.join(errors)}")
    if not rows:
        raise ValueError(f"Empty training data: {path}")
    return {"path": str(path.resolve()), "rows": rows, "sha256": digest.hexdigest(), "tasks": dict(sorted(counts.items()))}
