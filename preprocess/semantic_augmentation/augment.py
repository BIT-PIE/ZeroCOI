"""Augment normalized captions using Qwen, build interaction GloVe, and validate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Keep the source folder limited to the four release files when scripts run.
sys.dont_write_bytecode = True
if __package__:
    from .normalize import (Caption, PROJECT_ROOT, add_glove_arguments, atomic_write,
                            build_glove, field, glove_options, load_nlp, parse_caption,
                            text_files, validate, write_report)
else:
    from normalize import (Caption, PROJECT_ROOT, add_glove_arguments, atomic_write,
                           build_glove, field, glove_options, load_nlp, parse_caption,
                           text_files, validate, write_report)

DEFAULT_MODEL = "qwen3.7-plus"
SYSTEM_PROMPT = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
DIRECTIONS = {"on", "under", "near", "around", "in front of", "behind"}


@dataclass(frozen=True)
class ObjectDescription:
    objid: str
    description: str


def load_objects(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Object directory does not exist: {root}")
    objects = {}
    # <objid>.txt, <objid>/description.txt, or <objid>/caption.txt.
    # Multiple lines in a description describe the SAME object.
    manifest = root / "descriptions.json"
    for path in ([] if manifest.is_file() else sorted(root.rglob("*.txt"))):
        objid = path.parent.name if path.stem in {"description", "caption"} else path.stem
        description = " ".join(path.read_text(encoding="utf-8-sig").split())
        obj = ObjectDescription(field(objid), field(description))
        if obj.objid in objects:
            raise ValueError(f"Duplicate object description for {obj.objid}: {path}")
        objects[obj.objid] = obj
    # Optional explicit manifest for other layouts; values are paths, never default descriptions.
    if manifest.is_file():
        mapping = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise ValueError("descriptions.json must map objid to description file path")
        for objid, relative in mapping.items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Description path must stay inside the object directory")
            description = " ".join(path.read_text(encoding="utf-8-sig").split())
            objects[field(objid)] = ObjectDescription(field(objid), field(description))
    if not objects:
        raise ValueError(f"No object descriptions in {root}; supply real .txt descriptions or descriptions.json")
    return list(sorted(objects.values(), key=lambda obj: obj.objid))


def validate_response(result, obj, source=None):
    if not isinstance(result, dict) or result.get("objid") != obj.objid:
        raise ValueError("Qwen returned a different object ID")
    required = {"original_text", "object", "objid", "initial_relation", "direction", "augmented_text"}
    if set(result) != required:
        raise ValueError("Qwen response must contain exactly the six required fields")
    for value in result.values():
        field(value)
    if result["object"] != obj.description:
        raise ValueError("Qwen returned a different object description")
    if source is not None and result["original_text"] != source.text:
        raise ValueError("Qwen changed the original_text echo")
    caption = field(result["augmented_text"])
    relation = result.get("initial_relation")
    if relation not in {"far", "close"}:
        raise ValueError("Invalid initial_relation")
    direction = result["direction"]
    if direction not in DIRECTIONS:
        raise ValueError("Invalid direction")
    if not re.search(r"\b" + re.escape(direction) + r"\b", caption, re.I):
        raise ValueError("Caption does not contain its direction label")
    if relation == "far" and not re.search(r"\b(?:towards?|approach(?:es|ing|ed)?|closer to)\b", caption, re.I):
        raise ValueError("far requires an explicit approach phrase; local motion near an object is close")
    return caption, relation


class QwenRequestError(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__("Qwen generation failed (" + details["reason"] + "); no incomplete file was saved")


class QwenClient:
    def __init__(self, model=DEFAULT_MODEL, base_url=None, retries=3, timeout=90):
        self.key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.key or self.key.startswith("<") or self.key == "YOUR_DASHSCOPE_API_KEY":
            raise ValueError("Set DASHSCOPE_API_KEY to your own Qwen API key")
        self.model = model
        self.base_url = base_url or os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.retries = retries
        self.timeout = timeout

    def enhance(self, caption, obj):
        result = self.generate(caption, obj)
        return result["augmented_text"], result["initial_relation"]

    def generate(self, caption, obj):
        """Return validated interaction fields plus API metadata, without credentials."""
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"species": caption.species, "original_text": caption.text, "objid": obj.objid, "object_description": obj.description}, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
            "temperature": 0.3,
            "max_tokens": 512,
        }
        request = Request(self.base_url.rstrip("/") + "/chat/completions", data=json.dumps(data).encode(), headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        details = {"reason": "unknown failure"}
        for attempt in range(self.retries):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                choice = body["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Qwen output incomplete")
                result = json.loads(choice["message"]["content"])
                validate_response(result, obj, caption)
                return {**result, "api_metadata": {
                    "response_id": body.get("id"), "model": body.get("model"),
                    "usage": body.get("usage"), "finish_reason": choice.get("finish_reason"),
                }}
            except HTTPError as exc:
                # Never echo credentials or remote request/response bodies in errors.
                details = {"reason": f"HTTP {exc.code}", "http_status": exc.code}
                try:
                    error = json.load(exc).get("error", {})
                    if isinstance(error, dict):
                        details["provider_code"] = str(error.get("code", ""))[:100]
                        details["provider_message"] = str(error.get("message", "")).replace(self.key, "<REDACTED>")[:300]
                except (ValueError, TypeError, AttributeError):
                    pass
                if exc.code not in {408, 429} and exc.code < 500:
                    break
            except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
                details = {"reason": type(exc).__name__}
                if isinstance(exc, URLError):
                    details["network_reason"] = str(exc.reason).replace(self.key, "<REDACTED>")[:200]
                elif isinstance(exc, ValueError):
                    details["validation_error"] = str(exc)[:200]
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 8))
        raise QwenRequestError(details)


def augment(input_dir, output_dir, objects, nlp, client, seed=42, min_objects=1,
            max_objects=3, workers=4):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve().is_relative_to(output_dir.resolve()) or output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("Normalized and interaction directories must not overlap")
    if not 1 <= min_objects <= max_objects <= 3:
        raise ValueError("Require 1 <= min_objects <= max_objects <= 3")
    if min_objects > len(objects):
        raise ValueError("Object directory has fewer objects than --min-objects")
    if workers < 1:
        raise ValueError("--workers must be positive")
    rng = random.Random(seed)
    plans = []
    signature = sha256(json.dumps({
        "seed": seed, "min": min_objects, "max": max_objects,
        "relation": True, "model": getattr(client, "model", "test"),
        "endpoint": getattr(client, "base_url", "test"), "nlp": nlp.meta,
        "objects": [(obj.objid, obj.description) for obj in objects], "version": 2,
        "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest(),
    }, sort_keys=True).encode())
    for path in text_files(input_dir):
        name = path.relative_to(input_dir)
        content = path.read_text(encoding="utf-8")
        signature.update(str(name).encode() + b"\0" + content.encode())
        tasks = []
        for number, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                caption = parse_caption(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            count = rng.randint(min_objects, min(max_objects, len(objects)))
            tasks.extend((caption, obj) for obj in rng.sample(objects, count))
        if not tasks:
            raise ValueError(f"Empty normalized file: {path}")
        plans.append((name, tasks))
    output = Path(output_dir)
    state_path = output / ".augmentation_state.json"
    digest = signature.hexdigest()
    if state_path.exists():
        if json.loads(state_path.read_text())["signature"] != digest:
            raise ValueError("Inputs/options changed; use a new interaction output directory")
    elif any(output.rglob("*.txt")):
        raise ValueError("Existing interaction files have no matching state; use an empty output directory")
    atomic_write(state_path, json.dumps({"signature": digest}))
    expected = {str(name) for name, _ in plans}
    if {str(p.relative_to(output)) for p in output.rglob("*.txt")} - expected:
        raise ValueError("Unexpected interaction files in output directory")
    total = 0
    resumed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for name, tasks in plans:
            destination = output / name
            if destination.exists():
                lines = destination.read_text().splitlines()
                if len(lines) != len(tasks):
                    raise ValueError(f"Incomplete saved interaction file: {destination}")
                for line, (caption, obj) in zip(lines, tasks):
                    parts = line.split("#")
                    if parts[0] not in {"far", "close"}:
                        raise ValueError(f"Invalid relation in {destination}")
                    parts = parts[1:]
                    if len(parts) != 6 or parts[0] != obj.objid or parts[1] != caption.species:
                        raise ValueError(f"Saved object/species mismatch: {destination}")
                    parse_caption("#".join(parts[1:]))
                resumed += 1
            else:
                try:
                    # executor.map retains task order even when requests finish out of order.
                    results = list(executor.map(lambda task: client.enhance(*task), tasks))
                    captions = [text for text, relation in results]
                    docs = nlp.pipe(captions)
                    lines = []
                    for (source, obj), (text, relation), doc in zip(tasks, results, docs):
                        tokens = " ".join(f"{t.text}/{t.pos_}" for t in doc if not t.is_space)
                        record = Caption(source.species, text, tokens, source.start, source.end)
                        lines.append("#".join([relation, obj.objid] + record.fields()))
                    atomic_write(destination, "\n".join(lines) + "\n")
                except Exception as exc:
                    raise RuntimeError(f"Failed to augment {name}: {exc}") from exc
            total += len(tasks)
    return {"files": len(plans), "captions": total, "resumed_files": resumed, "objects": len(objects), "seed": seed}


def remove_augmentation_state(output_dir):
    (Path(output_dir) / ".augmentation_state.json").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "dataset" / "DogML3D")
    parser.add_argument("--texts-dir", type=Path, help="Override normalized text input directory")
    parser.add_argument("--obj-dir", type=Path, help="Default: <output-root>/obj")
    parser.add_argument("--interaction-dir", type=Path, help="Override interaction text output directory")
    parser.add_argument("--interaction-glove-dir", type=Path, help="Override interaction GloVe output directory")
    parser.add_argument("--nlp-model", default="en_core_web_sm")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", help="Qwen OpenAI-compatible endpoint")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--max-objects", type=int, default=3)
    add_glove_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.retries < 1 or args.timeout <= 0:
            raise ValueError("Retries and timeout must be positive")
        texts = args.texts_dir or args.output_root / "texts"
        interactions = args.interaction_dir or args.output_root / "texts_interaction"
        glove = args.interaction_glove_dir or args.output_root / "interaction_glove"
        objects = load_objects(args.obj_dir or args.output_root / "obj")
        client = QwenClient(args.model, args.base_url, args.retries, args.timeout)
        nlp = load_nlp(args.nlp_model)
        print("Generating object interactions with Qwen...", flush=True)
        report = {"augmentation": augment(texts, interactions, objects, nlp, client,
                                          args.seed, args.min_objects, args.max_objects,
                                          args.workers)}
        print("Building interaction GloVe...", flush=True)
        report["interaction_glove"] = build_glove(interactions, glove, True, True, **glove_options(args))
        report["interaction_validation"] = validate(interactions, glove, objects, True, True, nlp)
        write_report(args.output_root / "augment_report.json", report)
        remove_augmentation_state(interactions)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
