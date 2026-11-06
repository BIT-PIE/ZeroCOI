import argparse
import json
from pathlib import Path
import random


def split_dataset(input_dir, seed=42):
    """Split manifest groups so mirrors and long-motion segments never leak across splits."""
    input_dir = Path(input_dir)
    data_dir = input_dir / "new_joint_vecs"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"action feature directory does not exist: {data_dir}")
    feature_ids = {path.stem for path in data_dir.glob("*.npy") if path.is_file()}
    if not feature_ids:
        raise ValueError(f"no npy files in action feature directory: {data_dir}")
    manifest = input_dir / "motion_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"motion manifest does not exist: {manifest}")
    records = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"source_motion_path", "dataset", "animal_name", "mirrored", "target_motion_id"}
        if set(record) != required:
            raise ValueError(f"{manifest}:{number}: unexpected fields")
        records.append(record)
    manifest_ids = {record["target_motion_id"] for record in records}
    if len(manifest_ids) != len(records):
        raise ValueError("motion manifest has duplicate target_motion_id values")
    if manifest_ids != feature_ids:
        raise ValueError(
            f"manifest/features mismatch: {len(manifest_ids - feature_ids)} missing features, "
            f"{len(feature_ids - manifest_ids)} unlisted features"
        )

    groups = {}
    for record in records:
        key = (record["dataset"], record["source_motion_path"])
        groups.setdefault(key, []).append(record["target_motion_id"])

    group_keys = sorted(groups)
    train_count = len(group_keys) * 8 // 10
    test_count = len(group_keys) // 10
    if train_count == 0:
        raise ValueError("not enough data: the training split is empty after an 8:1:1 split")
    shuffled = group_keys.copy()
    random.Random(seed).shuffle(shuffled)
    split_groups = {
        "train": shuffled[:train_count],
        "test": shuffled[train_count:train_count + test_count],
        "val": shuffled[train_count + test_count:],
    }
    splits = {name: sorted(item for key in keys for item in groups[key])
              for name, keys in split_groups.items()}
    splits["all"] = sorted(feature_ids)
    for split, split_names in splits.items():
        path = input_dir / f"{split}.txt"
        path.write_text("".join(f"{name}\n" for name in split_names), encoding="utf-8")
        print(f"{path}: {len(split_names)} actions")
    return input_dir / "train.txt"


def animal_mean_variance(data_dir, train_file, save_dir):
    """Compute statistics over all frames of the training split, reusing feature3's grouped-std rule."""
    import numpy as np

    data_dir, train_file, save_dir = Path(data_dir), Path(train_file), Path(save_dir)
    names = [line.strip() for line in train_file.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not names:
        raise ValueError(f"training split file is empty: {train_file}")
    data_list = []
    for name in names:
        path = data_dir / f"{name}.npy"
        data = np.load(path, allow_pickle=False)
        if data.ndim != 2 or data.shape[1] != 428 or data.shape[0] == 0:
            raise ValueError(f"{path}: expected a non-empty (T, 428) array, got {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError(f"{path}: contains NaN or Inf, cannot compute training-set statistics")
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    feature_groups = (
        (0, 1),      # r_velocity
        (1, 3),      # l_velocity
        (3, 4),      # root_y
        (4, 106),    # ric_data
        (106, 109),  # trans
        (109, 115),  # pose_root
        (115, 319),  # pose_rest
        (319, 424),  # local_vel
        (424, 428),  # foot_contact
    )
    for start, end in feature_groups:
        std[start:end] = std[start:end].mean()
    np.save(save_dir / "Mean.npy", mean)
    np.save(save_dir / "Std.npy", std)
    print(f"training split: {len(names)} actions, {data.shape[0]} frames")
    print(f"saved {save_dir / 'Mean.npy'} and {save_dir / 'Std.npy'}, both with shape (428,)")
    return mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Split the input_dir/new_joint_vecs data and compute feature3's Mean/Std from the training split only.",
        allow_abbrev=False,
    )
    parser.add_argument("--input-dir", required=True,
                        help="dataset root directory, e.g. dataset/DogML3D")
    parser.add_argument("--seed", type=int, default=42, help="random split seed, default 42")
    args = parser.parse_args()
    if not args.input_dir.strip():
        parser.error("--input-dir must not be empty")
    input_dir = Path(args.input_dir)
    try:
        train_file = split_dataset(input_dir, args.seed)
        animal_mean_variance(input_dir / "new_joint_vecs", train_file, input_dir)
    except (OSError, ValueError, TypeError, EOFError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
