import argparse
import os
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np


UID_SCALE_MAP: Dict[str, Tuple[float, float, float]] = {
    "26df6ae6-1e44-46f9-9210-cae7ee877639": (2.4, 2.9, 2.9),
    "91cca58f-30a6-44a8-9203-633b842e871b": (0.4, 0.4, 0.4),
    "53950b7a-298c-4907-9328-d2167a2c8d9f": (0.75, 0.9, 0.9),
    "5119667c-2467-47d8-b68d-6c66bfb90bc3": (1.6, 1.35, 1.6),
    "0d8e22dc-5ea9-48ad-863e-8dba0f7483e9": (1.5, 1.3, 1.5),
    "85e2de25-68a2-42e4-94a7-6e238ac92a2d": (0.4, 0.4, 0.4),
    "a4aa6165-a24c-4646-9589-27e9d26f8951": (1.0, 1.0, 1.0),
    "2b44bb02-b77f-490f-acbe-7fe4ecb7dcc1": (1.0, 0.75, 1.0),
    "32bac62d-d3dc-4e53-aa41-80abb3c5aa03": (1.1, 0.9, 1.4),
    "48d2dfa5-55dd-408f-a33b-4a085742e055": (1.05, 1.05, 1.05),
    "0e874d17-37e5-372a-9be9-5ca5a3b1aa00": (1.9, 1.6, 1.9),
    "a7760f97-0dea-45f1-9e2c-02148d11a2c2": (2.9, 1.75, 2.9),
    "5f86cea2-bb34-4b39-9c62-2505417ac7c7": (1.8, 2.1, 2.1),
    "7fe87256-e9b8-4599-982b-61e732d9226c": (1.0, 1.25, 1.25),
    "3e08cbe7-87b9-45d6-80a4-0dfaee2b025c": (1.05, 1.2, 1.2),
    "0acc134e-65bc-4b68-b836-c591935bdec6": (1.7, 1.4, 1.7),
    "acf6e944-1e6c-417d-a882-01e3b6727b21": (1.0, 1.25, 1.25),
    "60999d08-1e04-41cb-a830-f909d0f61ae4": (0.9, 1.1, 1.1),
    "0d309488-bbbf-4ba1-8e2c-7e6fcfb26cb3": (1.3, 1.15, 1.5),
    "c681bf3b-8279-4548-a509-83801dd79d3b": (1.1, 1.1, 1.1),
    "23c641e7-189e-4dd4-a211-1d8dd731450d": (1.0, 1.0, 1.0),
    "925bc907-2909-43e4-a988-261bf098e3be": (1.7, 1.4, 1.7),
    "85d8cf70-d34a-4b2b-b2a4-85cf526e5a5c": (2.1, 1.7, 2.1),
    "e79ec767-9e3d-4f56-ab63-38736c6f674a": (1.9, 2.3, 2.3),
    "623579a7-78dc-4f93-b4f7-e608376e66a7": (0.4, 0.4, 0.4),
    "b50f317e-90f0-4c62-81fb-d41f426b94e5": (0.6, 1.1, 0.7),
    "42d8ad99-1c6e-45d6-862a-7c7b6792daf3": (1.1, 1.25, 1.25),
    "3decf748-5de4-43a5-b349-c8c849e66cca": (1.0, 1.25, 1.25),
    "c96d8ec9-3cd3-4506-af49-856984206440": (2.9, 1.3, 2.9),
}


def read_obj_vertices(obj_path: Path) -> np.ndarray:
    vertices = []

    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    ])

    if not vertices:
        raise ValueError(f"No vertex data found in: {obj_path}")

    return np.array(vertices, dtype=np.float64)


def compute_bbox_center(vertices: np.ndarray) -> np.ndarray:
    min_v = vertices.min(axis=0)
    max_v = vertices.max(axis=0)
    return (min_v + max_v) / 2.0


def compute_height_from_vertices(vertices: np.ndarray) -> float:
    min_v = vertices.min(axis=0)
    max_v = vertices.max(axis=0)
    center = (min_v + max_v) / 2.0
    lowest_y = min_v[1]
    return float(center[1] - lowest_y)


def save_scalar_npy(value: float, output_path: Path) -> None:
    np.save(output_path, np.array(value, dtype=np.float64))


def get_uid_scale(uid: str) -> np.ndarray:
    if uid not in UID_SCALE_MAP:
        raise KeyError(f"Scale factors not found for uid: {uid}")
    return np.array(UID_SCALE_MAP[uid], dtype=np.float64)


def transform_vertex(
    vertex: np.ndarray,
    center: np.ndarray,
    scale_xyz: np.ndarray,
) -> np.ndarray:
    centered = vertex - center
    transformed = centered * scale_xyz
    return transformed


def transform_normal(
    normal: np.ndarray,
    scale_xyz: np.ndarray,
) -> np.ndarray:
    inv_scale = np.array([
        1.0 / scale_xyz[0] if scale_xyz[0] != 0 else 0.0,
        1.0 / scale_xyz[1] if scale_xyz[1] != 0 else 0.0,
        1.0 / scale_xyz[2] if scale_xyz[2] != 0 else 0.0,
    ], dtype=np.float64)

    transformed = normal * inv_scale
    norm = np.linalg.norm(transformed)
    if norm != 0:
        transformed = transformed / norm
    return transformed


def build_transform_model(
    raw_model_path: Path,
    transform_model_path: Path,
    uid: str,
    height_npy_path: Path,
) -> Dict[str, float]:
    original_vertices = read_obj_vertices(raw_model_path)
    center = compute_bbox_center(original_vertices)
    scale_xyz = get_uid_scale(uid)

    with open(raw_model_path, "r", encoding="utf-8") as f_in, open(transform_model_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.startswith("v "):
                parts = line.strip().split()
                vertex = np.array(list(map(float, parts[1:4])), dtype=np.float64)
                transformed_vertex = transform_vertex(
                    vertex=vertex,
                    center=center,
                    scale_xyz=scale_xyz,
                )
                f_out.write(
                    f"v {transformed_vertex[0]:.6f} {transformed_vertex[1]:.6f} {transformed_vertex[2]:.6f}\n"
                )
            elif line.startswith("vn "):
                parts = line.strip().split()
                normal = np.array(list(map(float, parts[1:4])), dtype=np.float64)
                transformed_normal = transform_normal(
                    normal=normal,
                    scale_xyz=scale_xyz,
                )
                f_out.write(
                    f"vn {transformed_normal[0]:.6f} {transformed_normal[1]:.6f} {transformed_normal[2]:.6f}\n"
                )
            else:
                f_out.write(line)

    transformed_vertices = (original_vertices - center) * scale_xyz
    height = compute_height_from_vertices(transformed_vertices)
    save_scalar_npy(height, height_npy_path)

    return {
        "height": height,
        "scale_x": float(scale_xyz[0]),
        "scale_y": float(scale_xyz[1]),
        "scale_z": float(scale_xyz[2]),
    }


def process_single_model(
    raw_model_path: Path,
    transform_model_path: Path,
    uid: str,
    height_npy_path: Path,
) -> Dict[str, float]:
    return build_transform_model(
        raw_model_path=raw_model_path,
        transform_model_path=transform_model_path,
        uid=uid,
        height_npy_path=height_npy_path,
    )


def process_dataset(
    base_dir: Path,
    raw_model_name: str = "raw_model.obj",
    transform_model_name: str = "transform_model.obj",
    height_npy_name: str = "height.npy",
    use_models_subfolder: bool = True,
) -> List[Tuple[str, Dict[str, float]]]:
    results = []

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_dir}")

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        uid = folder.name
        work_dir = folder / "models" if use_models_subfolder else folder
        if not work_dir.exists() or not work_dir.is_dir():
            continue

        raw_model_path = work_dir / raw_model_name
        if not raw_model_path.exists():
            continue

        if uid not in UID_SCALE_MAP:
            raise KeyError(f"Missing scale config for uid: {uid}")

        transform_model_path = work_dir / transform_model_name
        height_npy_path = work_dir / height_npy_name

        info = process_single_model(
            raw_model_path=raw_model_path,
            transform_model_path=transform_model_path,
            uid=uid,
            height_npy_path=height_npy_path,
        )

        results.append((uid, info))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model preprocessing")
    parser.add_argument("--root_dir", "--base_dir", dest="root_dir", type=Path, required=True)
    args = parser.parse_args()

    process_dataset(
        base_dir=args.root_dir,
        raw_model_name="raw_model.obj",
        transform_model_name="transform_model.obj",
        height_npy_name="height.npy",
        use_models_subfolder=False,
    )

