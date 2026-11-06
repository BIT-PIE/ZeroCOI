import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

from mesh_to_sdf import get_surface_point_cloud


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh is None:
        raise ValueError(f"Failed to load mesh: {mesh_path}")
    return mesh


def normalize_mesh_for_sdf(mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
    scene_centroid = mesh.bounding_box.centroid
    scene_extents = mesh.bounding_box.extents
    scene_scale = float(np.max(scene_extents) * 0.5)

    if scene_scale == 0:
        raise ValueError("Scene scale is zero")

    mesh.vertices -= scene_centroid
    mesh.vertices /= scene_scale

    return {
        "mesh": mesh,
        "centroid": scene_centroid,
        "scale": scene_scale,
    }


def compute_sdf_and_gradient(
    mesh: trimesh.Trimesh,
    voxel_resolution: int = 128,
    sign_method: str = "normal",
    surface_point_method: str = "scan",
    bounding_radius: float = 3 ** 0.5,
    scan_count: int = 100,
    scan_resolution: int = 400,
    sample_point_count: int = 10000000,
    sample_count: int = 11,
    pad: bool = False,
    check_result: bool = False,
) -> Dict[str, np.ndarray]:
    surface_point_cloud = get_surface_point_cloud(
        mesh,
        surface_point_method=surface_point_method,
        bounding_radius=bounding_radius,
        scan_count=scan_count,
        scan_resolution=scan_resolution,
        sample_point_count=sample_point_count,
        calculate_normals=(sign_method == "normal"),
    )

    sdf_grid, gradient_grid = surface_point_cloud.get_voxels(
        voxel_resolution,
        sign_method == "depth",
        sample_count=sample_count,
        pad=pad,
        check_result=check_result,
        return_gradients=True,
    )

    return {
        "grid": sdf_grid,
        "gradient_grid": gradient_grid,
        "dim": voxel_resolution,
    }


def build_sdf_payload(
    mesh_file: Path,
    voxel_resolution: int = 128,
    sign_method: str = "normal",
) -> Dict[str, np.ndarray]:
    object_mesh = load_mesh(mesh_file)
    normalized = normalize_mesh_for_sdf(object_mesh)

    sdf_result = compute_sdf_and_gradient(
        mesh=normalized["mesh"],
        voxel_resolution=voxel_resolution,
        sign_method=sign_method,
    )

    return {
        "grid": sdf_result["grid"] * normalized["scale"],
        "gradient_grid": sdf_result["gradient_grid"],
        "dim": sdf_result["dim"],
        "centroid": normalized["centroid"],
        "scale": normalized["scale"],
    }


def save_sdf_pickle(sdf_payload: Dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(sdf_payload, f)


def process_single_sdf_model(
    obj_dir: Path,
    mesh_filename: str = "transform_model.obj",
    output_filename: str = "sdf_gradient.pkl",
    voxel_resolution: int = 128,
    sign_method: str = "normal",
    overwrite: bool = False,
) -> Optional[Path]:
    mesh_file = obj_dir / mesh_filename
    output_path = obj_dir / output_filename

    if not mesh_file.exists():
        return None

    if output_path.exists() and not overwrite:
        return output_path

    sdf_payload = build_sdf_payload(
        mesh_file=mesh_file,
        voxel_resolution=voxel_resolution,
        sign_method=sign_method,
    )
    save_sdf_pickle(sdf_payload, output_path)

    return output_path


def process_sdf_dataset(
    root_dir: Path,
    mesh_filename: str = "transform_model.obj",
    output_filename: str = "sdf_gradient.pkl",
    voxel_resolution: int = 128,
    sign_method: str = "normal",
    overwrite: bool = False,
) -> List[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_dir}")

    results = []

    for obj_dir in sorted(root_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        output_path = process_single_sdf_model(
            obj_dir=obj_dir,
            mesh_filename=mesh_filename,
            output_filename=output_filename,
            voxel_resolution=voxel_resolution,
            sign_method=sign_method,
            overwrite=overwrite,
        )

        if output_path is not None:
            results.append(output_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDF preprocessing")
    parser.add_argument("--root_dir", "--base_dir", dest="root_dir", type=Path, required=True)
    args = parser.parse_args()
    
    process_sdf_dataset(
        root_dir=args.root_dir,
        mesh_filename="transform_model.obj",
        output_filename="sdf_gradient.pkl",
        voxel_resolution=128,
        sign_method="normal",
        overwrite=False,
    )
