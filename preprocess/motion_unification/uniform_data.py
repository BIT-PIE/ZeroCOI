import argparse
import json
import os
from pathlib import Path
import shutil


DATASETS = ("DigiDogs", "AniMo4D", "AnimalML3D")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mirrored DSMAL parameters, 428D features and a motion manifest.",
        allow_abbrev=False,
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--output-dir", default=Path("dataset/DogML3D"), type=Path)
    args = parser.parse_args()
    return args


import numpy as np
import torch
if __package__:
    from .fit import get_model, load_params, save_params
else:
    from fit import get_model, load_params, save_params
from data_loaders.common.quaternion import (
    qbetween,
    qbetween_np,
    qinv_np,
    qmul_np,
    qrot_np,
    quaternion_to_matrix,
)
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def cont6d_to_matrix(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat


def matrix_to_cont6d(mat: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices (column vector convention)
    into the 6D continuous representation [x, y].

    Args:
        mat: Tensor of shape (..., 3, 3), where columns are orthogonal basis vectors.

    Returns:
        cont6d: Tensor of shape (..., 6)
    """
    assert mat.shape[-2:] == (3, 3), "Input must be a rotation matrix with shape (..., 3, 3)"

    # take the first two columns as the 6D representation
    x = mat[..., 0]  # (..., 3)
    y = mat[..., 1]  # (..., 3)

    cont6d = torch.cat([x, y], dim=-1)  # (..., 6)
    return cont6d


def normalize_t(joints: torch.Tensor):
    """Compute the translation that drops the motion onto the floor and zeroes the first frame's root XZ."""
    device = joints.device
    dtype = joints.dtype
    # drop onto the floor and zero the first frame's root xz
    min_y = joints.reshape(-1, 3)[:, 1].min()  # flatten to all coordinates and take the minimum y
    root_init_x = joints[0, 0, 0]
    root_init_z = joints[0, 0, 2]
    t = torch.tensor([-root_init_x, -min_y, -root_init_z], dtype=dtype, device=device)  # total translation vector
    R = torch.eye(3, device=device, dtype=dtype)
    return R, t


def normalize_R(joints: torch.Tensor):
    """Compute the rotation that aligns the initial facing direction with +Z."""
    device = joints.device
    dtype = joints.dtype
    # rotate to face z+
    across1 = joints[0, 11] - joints[0, 7]   # vector between the left and right front shoulders in the first frame, pointing left to right
    across2 = joints[0, 21] - joints[0, 17]
    across = across1 + across2
    across = across / torch.norm(across)  # normalize
    forward_init = torch.cross(
        torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype), across, dim=-1,
    )
    forward_init = forward_init / torch.norm(forward_init)
    target = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)  # target facing direction z+
    root_quat_init = qbetween(forward_init[None, :], target[None, :])[0]  # quaternion that rotates forward_init onto target
    R = quaternion_to_matrix(root_quat_init[None, :])[0]
    t = torch.zeros(3, device=device, dtype=dtype)
    return R, t


def foot_detect(joints, threshold=0.002):
    """ Detect whether the feet touch the ground. """
    foot_idx = [10, 14, 20, 24]  # left front foot, right front foot, left back foot, right back foot
    foot_joints = joints[:, foot_idx, :]
    displacement = (foot_joints[1:] - foot_joints[:-1]) **2  # (T-1, 4, 3)
    displacement = displacement.sum(axis=-1)  # (T-1, 4)
    contacts = (displacement < threshold).astype(np.float32)
    return contacts[:, [0]], contacts[:, [1]], contacts[:, [2]], contacts[:, [3]]  # (T-1, 1)


def get_local_vel(joints, r_rot):
    """ Get the velocity of each joint in the root's local coordinate frame. """
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], joints.shape[1], axis=1), joints[1:] - joints[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)
    return local_vel


def get_rifke(joints, r_rot):
    """ Unify the joints and facing direction across all frames. """
    # per frame: subtract the root's x and z from every joint's x and z
    joints[..., 0] -= joints[:, 0:1, 0]
    joints[..., 2] -= joints[:, 0:1, 2]
    # per frame: rotate the joints into the root's local coordinate frame
    joints = qrot_np(np.repeat(r_rot[:, None], joints.shape[1], axis=1), joints)
    return joints


def get_root_data(joints, r_rot):
    """ Get the root data. """
    # root angular velocity
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    # root linear velocity x, z
    velocity = joints[1:, 0] - joints[:-1, 0]
    velocity = qrot_np(r_rot[1:], velocity)
    l_velocity = velocity[:, [0, 2]]
    # root height
    joints = get_rifke(joints, r_rot)
    root_y = joints[:, 0, 1:2]
    return joints, np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)


def get_data(joints: np.ndarray, r_rot, pose, trans) -> np.ndarray:
    """ Build the data in the unified format. """
    local_vel = get_local_vel(joints, r_rot)  # velocity of each joint in the root's local coordinate frame
    lf, rf, lb, rb = foot_detect(joints)  # whether the feet touch the ground [threshold]
    joints, root_data = get_root_data(joints, r_rot)  # root joint features; joints is modified here
    ric_data = joints[:, 1:].reshape(len(joints), -1)  # 3D positions of the non-root joints in the root's local coordinate frame
    # concatenate the data
    data = root_data  # 4
    data = np.concatenate([data, ric_data[:-1]], axis=-1)  # 34*3
    data = np.concatenate([data, trans[:-1]], axis=-1)  # 3
    data = np.concatenate([data, pose[:-1, 0]], axis=-1)  # 6
    data = np.concatenate([data, pose[:-1, 1:].reshape(len(pose)-1, -1)], axis=-1)  # 34*6
    data = np.concatenate([data, local_vel], axis=-1)  # 35*3
    data = np.concatenate([data, lf, rf, lb, rb], axis=-1)  # 4
    return data


def convert_trans_orient(trans, orient, R, t):
    """ The given T should be column matrices. """
    trans = trans @ R.T + t
    rot_mats = cont6d_to_matrix(orient)
    rot_mats = torch.matmul(R, rot_mats)
    orient = matrix_to_cont6d(rot_mats)
    return trans, orient


def swap_left_right(trans, pose):
    device = trans.device
    left_chain = [7,8,9,10,17,18,19,20,33]
    right_chain = [11,12,13,14,21,22,23,24,34]
    r = pose[:, 0, :]
    m_r = r.clone().detach()
    R_pi = euler_angles_to_matrix(torch.tensor([0, 0, torch.pi], device=device), convention='XYZ')
    R_pi = R_pi.unsqueeze(0).expand(m_r.shape[0], -1, -1)
    R_c = axis_angle_to_matrix(m_r)
    R_w = torch.bmm(R_c, R_pi)
    m_r = matrix_to_axis_angle(R_w)
    m_r[:, [0, 2]] *= -1
    m_pose = pose.clone().detach()
    tmp = m_pose[:, left_chain]
    m_pose[:, left_chain] = m_pose[:, right_chain]
    m_pose[:, right_chain] = tmp
    m_pose[:, 1:, [0,2]] *= -1
    m_pose[:, 0:1, :] = m_r.unsqueeze(1)
    m_t = trans.clone().detach()
    m_t[:, 0] = - trans[:, 0]
    return m_t, m_pose


def mirror(params):
    ''' Mirror params. '''
    pose = matrix_to_axis_angle(rotation_6d_to_matrix(params['pose']))
    params["trans"], m_pose = swap_left_right(params["trans"], pose)
    params['pose'] = matrix_to_rotation_6d(axis_angle_to_matrix(m_pose))

    return params


def convert_params(params, R, t):
    """Apply the rigid transform to the global translation and the root joint rotation."""
    params["trans"], params["pose"][:, 0] = convert_trans_orient(params["trans"], params["pose"][:, 0], R, t)
    return params


def inverse_kinematics_np(joints):
    face_joint_idx = [7, 11, 17, 21]  # shoulders of the front legs and of the back legs
    # facing vector of each frame
    l_hip, r_hip, sdr_l, sdr_r = face_joint_idx
    across1 = joints[:, r_hip] - joints[:, l_hip]
    across2 = joints[:, sdr_r] - joints[:, sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
    forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]
    # target facing direction
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    # root rotation
    root_quat = qbetween_np(forward, target)
    return root_quat


def mirror_all(datasetname, output_dir="dataset/DogML3D", device="cpu"):
    '''Save original/mirrored parameters and return paths plus manifest records.'''
    if datasetname not in DATASETS:
        raise ValueError(f"unsupported dataset: {datasetname}")
    dataset_dir = Path(output_dir) / "fitted"
    if not dataset_dir.is_dir():
        print(f"***** no fitted {datasetname} motions; skipped *****")
        return [], []
    src_paths = sorted(
        path for path in dataset_dir.glob(f"{datasetname}_*.pkl") if path.is_file()
    )
    if not src_paths:
        print(f"***** no fitted {datasetname} motions; skipped *****")
        return [], []

    # Keep the dataset_animal_action naming; check up front so that several pkls or
    # M-prefixed actions cannot overwrite each other.
    jobs = []
    used_names = set()
    for src_path in src_paths:
        params = load_params(str(src_path), False, device)
        dog_name = params.get("_animal_name")
        if not dog_name:
            raise ValueError(f"{src_path}: missing _animal_name metadata")
        prefix = f"{datasetname}_{dog_name}_"
        if not src_path.stem.startswith(prefix):
            raise ValueError(f"fitted result name does not match metadata: {src_path}")
        action_name = src_path.stem[len(prefix):]
        name = f"{datasetname}_{dog_name}_{action_name}"
        mirror_name = f"{datasetname}_{dog_name}_M{action_name}"
        for output_name in (name, mirror_name):
            if output_name in used_names:
                raise ValueError(f"output name conflict: {output_name} ({src_path})")
            used_names.add(output_name)
        jobs.append((src_path, name, mirror_name))

    target_dir = Path(output_dir) / "params_before_uniform"
    target_dir.mkdir(parents=True, exist_ok=True)
    param_paths = []
    records = []
    for src_path, name, mirror_name in jobs:
        params = load_params(str(src_path), False, device)
        source_motion_path = params.get("_source_motion_path")
        animal_name = params.get("_animal_name", src_path.parent.parent.name)
        if not source_motion_path:
            raise ValueError(f"{src_path}: missing _source_motion_path metadata")
        if params["trans"].shape[0] < 2:
            raise ValueError(f"{src_path}: an action needs at least two frames to compute velocity features")
        new_file_path = target_dir / f"{name}.pkl"
        mirror_file_path = target_dir / f"{mirror_name}.pkl"
        shutil.copy(src_path, new_file_path)
        params = mirror(params)
        save_params(str(mirror_file_path), params)
        param_paths.extend((new_file_path, mirror_file_path))
        records.extend((
            {"source_motion_path": source_motion_path, "dataset": datasetname,
             "animal_name": animal_name,
             "mirrored": False, "target_motion_id": name},
            {"source_motion_path": source_motion_path, "dataset": datasetname,
             "animal_name": animal_name,
             "mirrored": True, "target_motion_id": mirror_name},
        ))
    return param_paths, records


def cleanup_intermediates(output_dir, dataset):
    """Remove one completed dataset's fitted and pre-uniform parameter files."""
    output_dir = Path(output_dir)
    for directory in (output_dir / "fitted", output_dir / "params_before_uniform"):
        if not directory.is_dir():
            continue
        for path in directory.glob(f"{dataset}_*.pkl"):
            path.unlink()
        try:
            directory.rmdir()
        except OSError:
            pass


def remove_stale_outputs(output_dir, dataset, records):
    output_dir = Path(output_dir)
    retained = {record["target_motion_id"] for record in records}
    for folder in ("dsmal_params", "new_joints", "new_joint_vecs"):
        directory = output_dir / folder
        if not directory.is_dir():
            continue
        for path in directory.glob(f"{dataset}_*.npy"):
            if path.stem not in retained:
                path.unlink()


def uniform_all(param_paths, output_dir, model, device):
    ''' Save the normalized parameters (T, 213), coordinates (T, 35, 3) and action features (T-1, 428). '''
    dsmal_dir = Path(output_dir) / "dsmal_params"
    joints_dir = Path(output_dir) / "new_joints"
    feature_dir = Path(output_dir) / "new_joint_vecs"
    dsmal_dir.mkdir(parents=True, exist_ok=True)
    joints_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    for f_path in param_paths:
        name = Path(f_path).stem
        params = load_params(str(f_path), False, device)
        frames = params["trans"].shape[0]
        # forward pass to compute the joint coordinates
        _, joints = model(
            beta=params["beta"].repeat(frames, 1),
            betas_limbs=params["betas_limbs"].repeat(frames, 1),
            scale=torch.abs(params['scale']).reshape(1, 1).expand(frames, 1),
            trans=params["trans"],
            pose=cont6d_to_matrix(params["pose"]),
        )
        # translation
        R, t = normalize_t(joints)
        params = convert_params(params=params, R=R, t=t)
        _, joints_norm1 = model(
            beta=params["beta"].repeat(frames, 1),
            betas_limbs=params["betas_limbs"].repeat(frames, 1),
            scale=torch.abs(params['scale']).reshape(1, 1).expand(frames, 1),
            trans=params["trans"],
            pose=cont6d_to_matrix(params["pose"]),
        )
        # rotation
        R, t = normalize_R(joints_norm1)
        params = convert_params(params=params, R=R, t=t)
        _, joints_norm2 = model(
            beta=params["beta"].repeat(frames, 1),
            betas_limbs=params["betas_limbs"].repeat(frames, 1),
            scale=torch.abs(params['scale']).reshape(1, 1).expand(frames, 1),
            trans=params["trans"],
            pose=cont6d_to_matrix(params["pose"]),
        )
        # save the full T-frame parameters and coordinates after translation and rotation normalization
        save_feature1(name, dsmal_dir, params)
        joints = joints_norm2.cpu().numpy()
        np.save(joints_dir / f"{name}.npy", joints)
        # compute r_rot
        r_rot = inverse_kinematics_np(joints)  # unit quaternion
        # compute data
        pose = params["pose"].cpu().numpy()
        trans = params["trans"].cpu().numpy()
        data = get_data(joints.copy(), r_rot, pose, trans)
        save_feature3(name, feature_dir, data)
        print(f"{f_path} -> {feature_dir / (name + '.npy')} {data.shape}")


def save_feature1(name, feature1_dir, params):
    ''' Feature 1. '''
    param_path = Path(feature1_dir) / f'{name}.npy'
    pose_root = params["pose"][:, 0]  # (T, 6)
    pose_rest = params["pose"][:, 1:].reshape(params["pose"].shape[0], -1)  # (T, 34*6)
    feature = torch.cat([
        params["trans"],      # (T, 3)
        pose_root,            # (T, 6)
        pose_rest,            # (T, 34*6)
    ], dim=1)
    assert feature.shape[1] == 3 + 6 + 34*6, "feature1 dim error!"
    np.save(param_path, feature.cpu().numpy())


def save_feature3(name, feature_dir, data):
    ''' Feature 3. '''
    feature_path = Path(feature_dir) / f'{name}.npy'
    assert data.shape[1] == 4 + 34*3 + 3 + 6 + 34*6 + 35*3 + 4, "feature3 dim error!"
    np.save(feature_path, data)


def write_manifest(output_dir, dataset, records):
    output_dir = Path(output_dir)
    required = {"source_motion_path", "dataset", "animal_name", "mirrored", "target_motion_id"}
    manifest = output_dir / "motion_manifest.jsonl"
    retained = []
    if manifest.is_file():
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if set(record) != required:
                raise ValueError(f"{manifest}:{number}: unexpected manifest fields")
            if record["dataset"] != dataset:
                retained.append(record)
    combined = retained + records
    target_ids = [record["target_motion_id"] for record in combined]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("motion manifest contains duplicate target_motion_id values")
    for record in records:
        motion_id = record["target_motion_id"]
        for folder in ("dsmal_params", "new_joints", "new_joint_vecs"):
            if not (output_dir / folder / f"{motion_id}.npy").is_file():
                raise FileNotFoundError(f"missing {folder} output for {motion_id}")
    combined.sort(key=lambda record: record["target_motion_id"])
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in combined),
        encoding="utf-8",
    )
    os.replace(temporary, manifest)
    print(f"{manifest}: {len(combined)} motions")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        param_paths, records = mirror_all(args.dataset, args.output_dir, device)
        if param_paths:
            model = get_model("bite", device)
            uniform_all(param_paths, args.output_dir, model, device)
        remove_stale_outputs(args.output_dir, args.dataset, records)
        write_manifest(args.output_dir, args.dataset, records)
        cleanup_intermediates(args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
