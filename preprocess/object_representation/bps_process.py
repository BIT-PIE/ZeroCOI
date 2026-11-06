import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import open3d as o3d

os.environ["PYOPENGL_PLATFORM"] = "egl"

from mesh_to_sdf import mesh_to_voxels, get_surface_point_cloud


OBJ_CLASS: Dict[str, List[str]] = {
    "bed": [
        "2b44bb02-b77f-490f-acbe-7fe4ecb7dcc1",
        "23c641e7-189e-4dd4-a211-1d8dd731450d",
        "48d2dfa5-55dd-408f-a33b-4a085742e055",
        "a4aa6165-a24c-4646-9589-27e9d26f8951",
        "c681bf3b-8279-4548-a509-83801dd79d3b",
    ],
    "stool": [
        "e79ec767-9e3d-4f56-ab63-38736c6f674a",
        "26df6ae6-1e44-46f9-9210-cae7ee877639",
        "5f86cea2-bb34-4b39-9c62-2505417ac7c7",
    ],
    "bowl": [
        "85e2de25-68a2-42e4-94a7-6e238ac92a2d",
        "91cca58f-30a6-44a8-9203-633b842e871b",
        "623579a7-78dc-4f93-b4f7-e608376e66a7",
    ],
    "chair": [
        "0e874d17-37e5-372a-9be9-5ca5a3b1aa00",
        "0acc134e-65bc-4b68-b836-c591935bdec6",
        "0d8e22dc-5ea9-48ad-863e-8dba0f7483e9",
        "925bc907-2909-43e4-a988-261bf098e3be",
        "5119667c-2467-47d8-b68d-6c66bfb90bc3",
    ],
    "sofa": [
        "3e08cbe7-87b9-45d6-80a4-0dfaee2b025c",
        "42d8ad99-1c6e-45d6-862a-7c7b6792daf3",
        "53950b7a-298c-4907-9328-d2167a2c8d9f",
        "60999d08-1e04-41cb-a830-f909d0f61ae4",
        "acf6e944-1e6c-417d-a882-01e3b6727b21",
    ],
    "swimming pool": [
        "c96d8ec9-3cd3-4506-af49-856984206440",
        "a7760f97-0dea-45f1-9e2c-02148d11a2c2",
        "85d8cf70-d34a-4b2b-b2a4-85cf526e5a5c",
    ],
    "table": [
        "0d309488-bbbf-4ba1-8e2c-7e6fcfb26cb3",
        "3decf748-5de4-43a5-b349-c8c849e66cca",
        "7fe87256-e9b8-4599-982b-61e732d9226c",
        "32bac62d-d3dc-4e53-aa41-80abb3c5aa03",
        "b50f317e-90f0-4c62-81fb-d41f426b94e5",
    ],
}


SAMPLE_POINTS_NUM = 1024
CENTER_ALIGN = True


def build_uuid_to_class_map(obj_class: Dict[str, List[str]]) -> Dict[str, str]:
    uuid_to_class = {}
    for class_name, uuid_list in obj_class.items():
        for uuid in uuid_list:
            uuid_to_class[uuid] = class_name
    return uuid_to_class


UUID_TO_CLASS = build_uuid_to_class_map(OBJ_CLASS)


def extract_category_from_name(name: str) -> str:
    return re.sub(r"\d+$", "", name).rstrip()


def get_category_from_uuid(folder_name: str) -> str:
    return UUID_TO_CLASS.get(folder_name, extract_category_from_name(folder_name))


def save_npy(value, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, value)


def save_scalar_npy(value: float, output_path: Path) -> None:
    save_npy(np.array(value, dtype=np.float64), output_path)


def save_class_npy(class_name: str, output_path: Path) -> None:
    save_npy(np.array(class_name, dtype="U"), output_path)


def delete_file_if_exists(file_path: Path) -> None:
    if file_path.exists():
        file_path.unlink()


def compute_height_from_obj(obj_path: Path) -> float:
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

    vertices = np.asarray(vertices, dtype=np.float64)
    min_v = vertices.min(axis=0)
    max_v = vertices.max(axis=0)
    center = (min_v + max_v) / 2.0
    lowest_y = min_v[1]
    return float(center[1] - lowest_y)


def get_original_mesh(obj_path: Path, center_align: bool = True) -> Tuple[trimesh.Trimesh, np.ndarray]:
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    center_transl_max = mesh.vertices.max(axis=0)
    center_transl_min = mesh.vertices.min(axis=0)
    center_transl = (center_transl_max + center_transl_min) / 2.0

    if not center_align:
        center_transl[1] = center_transl_min[1]

    mesh.apply_translation(-center_transl)
    return mesh, center_transl


def trimesh2open3d(trimesh_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(trimesh_mesh.vertices)
    faces = np.asarray(trimesh_mesh.faces)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh


def ensure_float32_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError(f"Unsupported type: {type(x)}")
    return x.float().to(device)


def get_one_directional_bps_feature(
    bps_set,
    point_set,
    return_bps_indices: bool = False,
    device: Optional[torch.device] = None,
    return_distance_vector: bool = False,
):
    assert len(bps_set.shape) == 3 and len(point_set.shape) == 3

    bps_set = ensure_float32_tensor(bps_set, device)
    point_set = ensure_float32_tensor(point_set, device)

    distances = torch.cdist(bps_set, point_set)
    nearest_distances, nearest_indices = torch.min(distances, dim=-1)

    batch_indices = torch.arange(
        nearest_indices.shape[0], device=point_set.device
    ).unsqueeze(-1).expand_as(nearest_indices)
    nearest_points = point_set[batch_indices, nearest_indices]
    distance_vectors = nearest_points - bps_set

    return_value = distance_vectors if return_distance_vector else nearest_distances

    if return_bps_indices:
        return return_value, nearest_indices
    return return_value


def bps_gen_ball_inside(n_bps: int = 1024, radius: float = 1.5, random_seed: int = 42) -> np.ndarray:
    np.random.seed(random_seed)
    x = np.random.normal(size=[n_bps, 3])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms
    r = np.random.uniform(size=[n_bps, 1])
    u = np.power(r, 1.0 / 3.0)
    basis_set = radius * x_unit * u
    return basis_set.astype(np.float32)


def get_default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_bps_set(
    cache_path: Path,
    radius: float = 1.5,
    n_bps: int = SAMPLE_POINTS_NUM,
    random_seed: int = 42,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = get_default_device()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not cache_path.exists():
        bps_set_np = bps_gen_ball_inside(
            n_bps=n_bps,
            radius=radius,
            random_seed=random_seed,
        )
        np.save(cache_path, bps_set_np)

    bps_set_np = np.load(cache_path)
    if bps_set_np.dtype != np.float32:
        bps_set_np = bps_set_np.astype(np.float32)

    return torch.from_numpy(bps_set_np).float().to(device)


def sample_mesh_points(
    obj_path: Path,
    sample_points_num: int = SAMPLE_POINTS_NUM,
    center_align: bool = CENTER_ALIGN,
) -> np.ndarray:
    mesh, _ = get_original_mesh(obj_path, center_align=center_align)
    mesh_o3d = trimesh2open3d(mesh)
    pcd = mesh_o3d.sample_points_uniformly(number_of_points=sample_points_num)
    return np.asarray(pcd.points)


def compute_bps_array_from_sample_points(
    sample_points: np.ndarray,
    bps_set: torch.Tensor,
) -> np.ndarray:
    bps_feat = get_one_directional_bps_feature(
        bps_set[None],
        sample_points[None],
        return_distance_vector=False,
        device=bps_set.device,
    )
    return bps_feat.squeeze(0).detach().cpu().numpy()


def compute_and_save_features(
    obj_dir: Path,
    bps_set: torch.Tensor,
    transform_model_name: str = "transform_model.obj",
    bps_name: str = "bps.npy",
    sample_points_name: str = "sample_points.npy",
    height_name: str = "height.npy",
    class_name: str = "class.npy",
) -> None:
    model_path = obj_dir / transform_model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Missing file: {model_path}")

    category = get_category_from_uuid(obj_dir.name)
    height = compute_height_from_obj(model_path)
    sample_points = sample_mesh_points(model_path)
    bps_array = compute_bps_array_from_sample_points(sample_points, bps_set)

    save_npy(bps_array, obj_dir / bps_name)
    save_npy(sample_points, obj_dir / sample_points_name)
    save_scalar_npy(height, obj_dir / height_name)
    save_class_npy(category, obj_dir / class_name)


def process_dataset(
    root_dir: Path,
    bps_cache_path: Path,
    transform_model_name: str = "transform_model.obj",
    bps_name: str = "bps.npy",
    sample_points_name: str = "sample_points.npy",
    height_name: str = "height.npy",
    class_name: str = "class.npy",
) -> None:
    bps_set = prepare_bps_set(
        cache_path=bps_cache_path,
        radius=1.5,
        n_bps=1024,
        random_seed=42,
    )

    for obj_dir in sorted(root_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        model_path = obj_dir / transform_model_name
        if not model_path.exists():
            continue

        compute_and_save_features(
            obj_dir=obj_dir,
            bps_set=bps_set,
            transform_model_name=transform_model_name,
            bps_name=bps_name,
            sample_points_name=sample_points_name,
            height_name=height_name,
            class_name=class_name,
        )


def build_npz_payload(
    bps_file: Path,
    height_file: Path,
    class_file: Path,
) -> Dict[str, np.ndarray]:
    obj_bps = np.load(bps_file, allow_pickle=False)
    obj_center_height = np.load(height_file, allow_pickle=False)
    obj_scale = np.array(1.0, dtype=np.float64)

    obj_class_arr = np.load(class_file, allow_pickle=False)
    if obj_class_arr.ndim == 0:
        obj_class = str(obj_class_arr.item())
    else:
        obj_class = str(obj_class_arr.flatten()[0])

    return {
        "obj_BPS": obj_bps,
        "obj_center_height": obj_center_height,
        "obj_scale": obj_scale,
        "obj_class": np.array(obj_class, dtype="U"),
    }


def pack_dataset_to_npz(
    root_dir: Path,
    bps_name: str = "bps.npy",
    height_name: str = "height.npy",
    class_name: str = "class.npy",
    delete_temp_files_after_pack: bool = True,
) -> None:

    for obj_dir in sorted(root_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        bps_file = obj_dir / bps_name
        height_file = obj_dir / height_name
        class_file = obj_dir / class_name

        required_files = [bps_file, height_file, class_file]
        if not all(file.exists() for file in required_files):
            continue

        data = build_npz_payload(
            bps_file=bps_file,
            height_file=height_file,
            class_file=class_file,
        )

        np.savez_compressed(obj_dir / "BPS.npz", **data)

        if delete_temp_files_after_pack:
            delete_file_if_exists(bps_file)
            delete_file_if_exists(height_file)
            delete_file_if_exists(class_file)


def run_pipeline(
    root_dir: Path,
    bps_cache_path: Path,
) -> None:
    process_dataset(
        root_dir=root_dir,
        bps_cache_path=bps_cache_path,
        transform_model_name="transform_model.obj",
        bps_name="bps.npy",
        sample_points_name="sample_points.npy",
        height_name="height.npy",
        class_name="class.npy",
    )

    pack_dataset_to_npz(
        root_dir=root_dir,
        bps_name="bps.npy",
        height_name="height.npy",
        class_name="class.npy",
        delete_temp_files_after_pack=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPS preprocessing")
    parser.add_argument("--root_dir", "--base_dir", dest="root_dir", type=Path, required=True)
    args = parser.parse_args()

    run_pipeline(
        root_dir=args.root_dir,
        bps_cache_path=args.root_dir / "bps_set.npy",
    )
