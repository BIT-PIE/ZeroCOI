"""Normalize dataset captions, build GloVe, and validate the generated files."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import pickle
import re

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MIRRORED_DATASETS = {"DigiDogs", "AniMo4D", "AnimalML3D"}


@dataclass(frozen=True)
class Caption:
    species: str
    text: str
    tokens: str
    start: str = "0.0"
    end: str = "0.0"

    def fields(self):
        return [self.species, self.text, self.tokens, self.start, self.end]

    def line(self):
        return "#".join(self.fields())


def field(value):
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "#\r\n"):
        raise ValueError("Fields must be nonempty strings without # or newlines")
    return value.strip()


def parse_caption(line):
    parts = line.strip().split("#")
    if len(parts) != 5:
        raise ValueError("Expected species#caption#tokens#start#end")
    parts = [field(p) for p in parts]
    for timing in parts[-2:]:
        if not math.isfinite(float(timing)):
            raise ValueError("Nonfinite caption timing")
    for token in parts[2].split():
        word, pos = token.rsplit("/", 1)
        if not word or not pos:
            raise ValueError("Expected word/POS tokens")
    return Caption(*parts)


def load_nlp(model):
    import spacy
    try:
        return spacy.load(model, disable=["ner"])
    except OSError as exc:
        raise ValueError(f"Missing spaCy model {model}; run python -m spacy download {model}") from exc


def tokenize(nlp, text):
    return " ".join(f"{t.text}/{t.pos_}" for t in nlp(text) if not t.is_space)


def text_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise ValueError(f"No .txt input files in {root}")
    return files


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def add_glove_arguments(parser):
    parser.add_argument("--dimension", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-glove", type=Path, help="Use a pretrained GloVe text file instead of training")


def glove_options(args):
    return {"dimension": args.dimension, "epochs": args.epochs, "window": args.window,
            "seed": args.seed, "pretrained": args.pretrained_glove}


def write_report(path, report):
    atomic_write(path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary_keys = {"files", "captions", "output_files", "vocabulary", "embedding_shape",
                    "objects", "resumed_files", "method"}
    for stage, result in report.items():
        print(stage + ": " + json.dumps({key: value for key, value in result.items() if key in summary_keys}, ensure_ascii=False))
    print(f"Report: {path}")


def load_corpus(root, interaction=False, include_relation=False):
    corpus = []
    for path in text_files(root):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("#")
            offset = (2 if include_relation else 1) if interaction else 0
            try:
                record = parse_caption("#".join(parts[offset:]))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            words = [token.rsplit("/", 1)[0].lower() for token in record.tokens.split()]
            words = [word for word in words if re.fullmatch(r"\w+", word)]
            if words:
                corpus.append(words)
    if not corpus:
        raise ValueError("Empty GloVe corpus")
    return corpus


def train_glove(corpus, words, dimension, epochs, window, seed):
    index = {word: i for i, word in enumerate(words)}
    counts = defaultdict(float)
    for sentence in corpus:
        ids = [index[word] for word in sentence]
        for center, i in enumerate(ids):
            for previous in range(max(0, center - window), center):
                j = ids[previous]
                weight = 1.0 / (center - previous)
                counts[i, j] += weight
                counts[j, i] += weight
    if not counts:
        raise ValueError("GloVe needs at least one within-sentence word pair")
    pairs = np.array(sorted(counts), dtype=np.int64)
    values = np.array([counts[tuple(pair)] for pair in pairs], dtype=np.float32)
    target = np.log(values)
    weights = np.minimum((values / 100.0) ** 0.75, 1.0)
    rng = np.random.default_rng(seed)
    size = len(words)
    w = rng.uniform(-0.5 / dimension, 0.5 / dimension, (size, dimension)).astype(np.float32)
    context = rng.uniform(-0.5 / dimension, 0.5 / dimension, (size, dimension)).astype(np.float32)
    bias = np.zeros(size, dtype=np.float32)
    context_bias = bias.copy()
    parameters = [w, context, bias, context_bias]
    accumulators = [np.ones_like(p) for p in parameters]

    def loss():
        i, j = pairs.T
        error = np.einsum("ij,ij->i", w[i], context[j]) + bias[i] + context_bias[j] - target
        return float(np.mean(weights * error ** 2))

    history = [loss()]
    # Mini-batch AdaGrad, summing repeated-row gradients before each update.
    for epoch in range(epochs):
        order = rng.permutation(len(pairs))
        for start in range(0, len(order), 256):
            selected = order[start:start + 256]
            i, j = pairs[selected].T
            wi, cj = w[i].copy(), context[j].copy()
            error = np.einsum("ij,ij->i", wi, cj) + bias[i] + context_bias[j] - target[selected]
            delta = 2 * weights[selected] * error
            gradients = [delta[:, None] * cj, delta[:, None] * wi, delta, delta]
            for parameter, accumulator, rows, gradient in zip(parameters, accumulators, [i, j, i, j], gradients):
                unique, inverse = np.unique(rows, return_inverse=True)
                combined = np.zeros((len(unique), *parameter.shape[1:]), dtype=np.float32)
                np.add.at(combined, inverse, gradient)
                combined = np.clip(combined, -10, 10)
                accumulator[unique] += combined ** 2
                parameter[unique] -= 0.05 * combined / np.sqrt(accumulator[unique])
        history.append(loss())
    return w + context, {"cooccurrence_pairs": len(pairs), "loss": history}


def build_glove(input_dir, output_dir, interaction=False, include_relation=False,
                dimension=300, epochs=50, window=5, seed=42, pretrained=None):
    if min(dimension, epochs, window) < 1:
        raise ValueError("GloVe dimension, epochs and window must be positive")
    corpus = load_corpus(input_dir, interaction, include_relation)
    frequencies = Counter(word for sentence in corpus for word in sentence)
    words = ["unk", "sos", "eos"] + sorted(set(frequencies) - {"unk", "sos", "eos"})
    rng = np.random.default_rng(seed)
    report = {"sentences": len(corpus), "vocabulary": len(words), "dimension": dimension}
    if pretrained:
        index = {word: i for i, word in enumerate(words)}
        vectors = rng.normal(0, 0.01, (len(words), dimension)).astype(np.float32)
        found = set()
        with Path(pretrained).open(encoding="utf-8") as stream:
            for line in stream:
                parts = line.split()
                if parts and parts[0] in index:
                    if len(parts) != dimension + 1:
                        raise ValueError(f"Pretrained vector dimension mismatch: {parts[0]}")
                    vectors[index[parts[0]]] = np.asarray(parts[1:], dtype=np.float32)
                    found.add(parts[0])
        if not found:
            raise ValueError("No vocabulary words found in pretrained GloVe file")
        # Corpus OOV words use the matched-vector mean; no silent random OOV vectors.
        unknown = vectors[[index[word] for word in sorted(found)]].mean(axis=0)
        for word in set(words) - found:
            vectors[index[word]] = unknown
        report.update({"method": "pretrained_glove", "matched_words": len(found), "oov_words": len(words) - len(found)})
    else:
        vectors, training = train_glove(corpus, words, dimension, epochs, window, seed)
        # Special tokens have no observed context; initialize them explicitly.
        vectors[:3] = rng.normal(0, 0.01, (3, dimension))
        report.update({"method": "trained_glove", "epochs": epochs, **training})
    if not np.isfinite(vectors).all():
        raise ValueError("GloVe training produced nonfinite values")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "our_vab_data.npy", vectors)
    with (output / "our_vab_words.pkl").open("wb") as stream:
        pickle.dump(words, stream, protocol=4)
    with (output / "our_vab_idx.pkl").open("wb") as stream:
        pickle.dump({word: i for i, word in enumerate(words)}, stream, protocol=4)
    return report


def validate(text_dir, glove_dir=None, objects=None, interaction=False,
             include_relation=False, nlp=None):
    vocabulary = {"unk", "sos", "eos"}
    valid_ids = {obj.objid for obj in objects} if objects is not None else None
    count = 0
    files = text_files(text_dir)
    species = set()
    for path in files:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            raise ValueError(f"Empty output file: {path}")
        for number, line in enumerate(lines, 1):
            parts = line.split("#")
            try:
                offset = (2 if include_relation else 1) if interaction else 0
                if len(parts) != 5 + offset:
                    raise ValueError("Unexpected number of # fields")
                if include_relation and interaction and parts[0] not in {"far", "close"}:
                    raise ValueError("Invalid initial relation")
                if interaction and valid_ids is not None and parts[offset - 1] not in valid_ids:
                    raise ValueError("Object ID missing from object directory")
                caption = parse_caption("#".join(parts[offset:]))
                if (caption.start, caption.end) != ("0.0", "0.0"):
                    raise ValueError("Expected 0.0#0.0 timing fields")
                words = [t.rsplit("/", 1)[0] for t in caption.tokens.split()]
                if nlp is not None:
                    expected = [t.text for t in nlp.make_doc(caption.text) if not t.is_space]
                    if words != expected:
                        raise ValueError("Caption words and POS token words differ")
                vocabulary.update(w.lower() for w in words if re.fullmatch(r"\w+", w))
                species.add(caption.species)
                count += 1
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    result = {"files": len(files), "captions": count, "species": sorted(species), "vocabulary": len(vocabulary)}
    if glove_dir:
        glove_dir = Path(glove_dir)
        data = np.load(glove_dir / "our_vab_data.npy", allow_pickle=False)
        with (glove_dir / "our_vab_words.pkl").open("rb") as stream:
            words = pickle.load(stream)
        with (glove_dir / "our_vab_idx.pkl").open("rb") as stream:
            index = pickle.load(stream)
        if data.ndim != 2 or data.shape[0] != len(words) or not np.isfinite(data).all():
            raise ValueError("Invalid GloVe matrix")
        if len(set(words)) != len(words) or words[:3] != ["unk", "sos", "eos"]:
            raise ValueError("Invalid vocabulary special tokens or duplicate words")
        if index != {word: i for i, word in enumerate(words)} or set(words) != vocabulary:
            raise ValueError("Embedding vocabulary/index does not match caption corpus")
        result["embedding_shape"] = list(data.shape)
    return result


def animalml3d_model(path):
    stem = path.parent.name if path.stem == "screenshots" else path.stem
    prefix = "AnimalML3D_"
    if stem.lower().startswith(prefix.lower()):
        stem = stem[len(prefix):]
    model, separator, action = stem.partition("_")
    if not separator or not model or not action:
        raise ValueError(f"Expected an AnimalML3D model_action name: {path}")
    return field(model)


def animalml3d_species(path):
    """Use the model identifier, preserving case, rather than the caption's noun."""
    return animalml3d_model(path)


def animo4d_subject(path):
    subject, separator, _ = path.stem.partition("__")
    if not separator:
        raise ValueError(f"Expected an AniMo4D subject__motion name: {path}")
    return subject


def motion_name(dataset, path, root):
    stem = path.stem
    if dataset == "AnimalML3D" and stem == "screenshots":
        stem = path.parent.name
    if dataset == "AniMo4D" and "__" in stem:
        subject = animo4d_subject(path)
        species, gender = subject.rsplit("_", 1)
        subject = f"{gender}_{species}"
        action = stem.removesuffix("_keypoints.json").rsplit("_", 1)[-1]
        stem = f"{subject}_{action}"
    if stem.lower().startswith(dataset.lower() + "_"):
        stem = stem[len(dataset) + 1:]
    # Generic nested layouts retain parent names to avoid basename collisions.
    if path.parent != root and path.stem != "screenshots":
        stem = "_".join((*path.relative_to(root).parts[:-1], stem))
    return f"{dataset}_{stem}.txt"


def mirrored_motion_name(dataset, path, root):
    name = motion_name(dataset, path, root)
    if dataset == "DigiDogs":
        stem = path.stem
        if stem.lower().startswith("digidogs_"):
            stem = stem[len("DigiDogs_"):]
        _, separator, action = stem.partition("_")
        if not separator or not action:
            raise ValueError(f"Expected a DigiDogs species_action name: {path}")
    elif dataset == "AniMo4D":
        action = path.stem.removesuffix("_keypoints.json").rsplit("_", 1)[-1]
    elif dataset == "AnimalML3D":
        stem = path.parent.name if path.stem == "screenshots" else path.stem
        if stem.lower().startswith("animalml3d_"):
            stem = stem[len("AnimalML3D_"):]
        _, _, action = stem.partition("_")
    else:
        raise ValueError(f"Dataset has no mirrored text convention: {dataset}")
    return name[:-len(action) - 4] + "M" + action + ".txt"


def mirrored_caption(caption):
    def replace_direction(match):
        original = match.group(0)
        replacement = "right" if original.lower() == "left" else "left"
        if original.isupper():
            return replacement.upper()
        if original[0].isupper():
            return replacement.capitalize()
        return replacement

    text = re.sub(r"\b(?:left|right)\b", replace_direction, caption.text, flags=re.I)
    tokens = []
    for token in caption.tokens.split():
        word, pos = token.rsplit("/", 1)
        word = re.sub(r"^(?:left|right)$", replace_direction, word, flags=re.I)
        tokens.append(f"{word}/{pos}")
    return Caption(caption.species, text, " ".join(tokens), caption.start, caption.end)


def source_caption(dataset, path, line, nlp):
    parts = line.split("#")
    if dataset == "AnimalML3D":
        if len(parts) not in {1, 5}:
            raise ValueError(f"Unsupported source format with {len(parts)} fields")
        species = animalml3d_species(path)
        text = parts[1] if len(parts) == 5 else line
    elif len(parts) == 5:
        species, text = parts[:2]
    elif len(parts) == 6 and dataset == "AniMo4D":
        species, gender, text = parts[:3]
        if "__" in path.stem:
            # The original annotations abbreviate e.g. 'african wild dog' to 'wild dog'.
            species = path.stem.split("__", 1)[0].rsplit("_", 1)[0].replace("_", " ")
        species = f"{gender} {species}"
    elif len(parts) == 1:
        text = line
        if dataset == "DigiDogs":
            match = re.match(r"(?:DigiDogs_)?([^_]+)_", path.stem, re.I)
            if match:
                species = match[1]
            else:
                raise ValueError("Cannot infer DigiDogs species from filename")
        else:
            doc = nlp(text)
            # Retain the existing subject-based handling for plain AniMo4D text.
            nouns = [t for t in doc if t.pos_ in {"NOUN", "PROPN"}]
            if not nouns:
                raise ValueError("Cannot infer species; provide a species#caption#tokens#0.0#0.0 line")
            species = nouns[0].text.lower()
    else:
        raise ValueError(f"Unsupported source format with {len(parts)} fields")
    species, text = field(species), field(text)
    # Recompute tokens: historical AniMo4D tokens contain lemmas instead of caption words.
    return Caption(species, text, tokenize(nlp, text))


def add_manifest_fallbacks(grouped, manifest_path, nlp):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return 0
    templates_path = Path(__file__).with_name("add.txt")
    template_source = templates_path.read_text(encoding="utf-8").strip().rstrip(",")
    templates = json.loads("{" + template_source + "}")
    added = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        target = record["target_motion_id"]
        target_file = target + ".txt"
        if target_file in grouped or record["dataset"] != "AniMo4D":
            continue
        animal = record["animal_name"]
        prefix = f"AniMo4D_{animal}_"
        if not target.startswith(prefix):
            raise ValueError(f"Cannot parse AniMo4D target from manifest: {target}")
        action = target[len(prefix):]
        mirrored = record["mirrored"]
        if mirrored:
            if not action.startswith("M"):
                raise ValueError(f"Mirrored AniMo4D target lacks M prefix: {target}")
            action = action[1:]
        template = templates.get(action.lower())
        if template is None:
            raise ValueError(f"No official caption or add.txt fallback for {target}")
        gender, separator, species = animal.partition("_")
        if not separator:
            raise ValueError(f"Cannot parse AniMo4D animal_name: {animal}")
        species = species.replace("_", " ")
        text = template.format(gender=gender, species=species)
        caption = Caption(f"{gender} {species}", text, tokenize(nlp, text))
        grouped[target_file].append(
            (mirrored_caption(caption) if mirrored else caption).line()
        )
        added += 1
    return added


def deduplicate_caption_lines(lines):
    unique = []
    seen = set()
    for line in lines:
        caption = parse_caption(line)
        key = re.sub(r"\s+", " ", caption.text).strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(line)
    return unique


def normalize(sources, output, nlp, manifest_path=None):
    output = Path(output)
    manifest_path = Path(manifest_path) if manifest_path else None
    if manifest_path is None or not manifest_path.is_file():
        raise ValueError(f"Motion manifest is required: {manifest_path}")
    manifest_targets = {
        json.loads(line)["target_motion_id"]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for root in sources.values():
        root = Path(root).resolve()
        if output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve()):
            raise ValueError("Source and normalized output directories must not overlap")
    grouped = defaultdict(list)
    report = {"datasets": {}, "empty_files": [], "merged_sources": {}}
    for dataset, root in sources.items():
        root = Path(root)
        available_files = text_files(root)
        files = available_files
        if not files:
            raise ValueError(f"No selected {dataset} dog text files in {root}")
        count = 0
        used_files = set()
        names = defaultdict(list)
        for path in files:
            lines = [(i, l.strip()) for i, l in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1) if l.strip()]
            if not lines:
                report["empty_files"].append(str(path))
                continue
            name = motion_name(dataset, path, root)
            if manifest_targets is not None and name[:-4] not in manifest_targets:
                continue
            used_files.add(path)
            names[name].append(str(path))
            mirror = None
            if dataset in MIRRORED_DATASETS:
                mirror = mirrored_motion_name(dataset, path, root)
                names[mirror].append(str(path))
            for number, line in lines:
                try:
                    caption = source_caption(dataset, path, line, nlp)
                    grouped[name].append(caption.line())
                    if mirror is not None:
                        grouped[mirror].append(mirrored_caption(caption).line())
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
                count += 1
        generated = count * (2 if dataset in MIRRORED_DATASETS else 1)
        report["datasets"][dataset] = {
            "available_source_files": len(available_files),
            "selected_source_files": len(used_files),
            "filtered_source_files": len(available_files) - len(used_files),
            "source_captions": count,
            "captions": generated,
            "output_files": len(names),
        }
        report["merged_sources"].update({k: v for k, v in names.items() if len(v) > 1})
    report["manifest_fallback_files"] = (
        add_manifest_fallbacks(grouped, manifest_path, nlp) if manifest_path else 0
    )
    existing = {p.name for p in output.glob("*.txt")}
    stale = existing - grouped.keys()
    if stale:
        raise ValueError("Output contains unrelated/stale files; choose an empty --output-root")
    grouped = {name: deduplicate_caption_lines(lines) for name, lines in grouped.items()}
    for name, lines in sorted(grouped.items()):
        atomic_write(output / name, "\n".join(lines) + "\n")
    report["output_files"] = len(grouped)
    report["captions"] = sum(map(len, grouped.values()))
    return report


def validate_motion_text_alignment(texts_dir, manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return {"status": "skipped", "reason": f"motion manifest not found: {manifest_path}"}
    required = {"source_motion_path", "dataset", "animal_name", "mirrored", "target_motion_id"}
    records = []
    for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != required:
            raise ValueError(f"{manifest_path}:{number}: unexpected manifest fields")
        records.append(record)
    expected = [record["target_motion_id"] for record in records]
    if len(expected) != len(set(expected)):
        raise ValueError(f"{manifest_path}: duplicate target_motion_id values")
    actual = {path.stem for path in Path(texts_dir).glob("*.txt") if path.is_file()}
    expected_set = set(expected)
    if expected_set != actual:
        raise ValueError(
            "motion/text mismatch: "
            f"{len(expected_set - actual)} motions have no text and "
            f"{len(actual - expected_set)} texts have no motion"
        )
    return {"status": "ok", "motions": len(expected_set), "texts": len(actual)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    datasets = ["DigiDogs", "AniMo4D", "AnimalML3D"]
    parser.add_argument("--datasets", nargs="+", choices=datasets, default=datasets,
                        help="Datasets to process; defaults to all three")
    for flag, dataset in [("digi-texts", "DigiDogs"), ("animo-texts", "AniMo4D"), ("animal-texts", "AnimalML3D")]:
        default_dir = PROJECT_ROOT / "dataset" / dataset / ("motion_captions" if dataset == "AnimalML3D" else "texts")
        parser.add_argument("--" + flag, type=Path, default=default_dir)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "dataset" / "DogML3D")
    parser.add_argument("--texts-dir", type=Path, help="Override normalized text output directory")
    parser.add_argument("--glove-dir", type=Path, help="Override base GloVe output directory")
    parser.add_argument("--motion-manifest", type=Path,
                        help="Defaults to OUTPUT_ROOT/motion_manifest.jsonl when present")
    parser.add_argument("--nlp-model", default="en_core_web_sm")
    add_glove_arguments(parser)
    args = parser.parse_args(argv)
    try:
        texts = args.texts_dir or args.output_root / "texts"
        glove = args.glove_dir or args.output_root / "glove"
        nlp = load_nlp(args.nlp_model)
        sources = {"DigiDogs": args.digi_texts, "AniMo4D": args.animo_texts, "AnimalML3D": args.animal_texts}
        print("Normalizing original captions...", flush=True)
        manifest = args.motion_manifest or args.output_root / "motion_manifest.jsonl"
        report = {"normalization": normalize(
            {name: sources[name] for name in args.datasets}, texts, nlp, manifest
        )}
        report["motion_text_alignment"] = validate_motion_text_alignment(texts, manifest)
        print("Building base GloVe...", flush=True)
        report["glove"] = build_glove(texts, glove, **glove_options(args))
        report["validation"] = validate(texts, glove, nlp=nlp)
        write_report(args.output_root / "prepare_report.json", report)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
